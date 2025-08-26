import torch
from torch.utils.data import DataLoader
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.llama import LlamaTokenizer
from .config import MODEL_CONFIGS, MODEL_TEMPLATES, DEFENSE_SYSTEM_PROMPTS, PROMPT_STRATEGY

class GenericDataManager:
    def __init__(self, config, training_stage, model_type="auto", task_type="hh", tokenizer_path=None):
        self.config = config
        self.training_stage = training_stage
        self.model_type = model_type
        self.task_type = task_type
        
        # 获取模型配置
        model_config = MODEL_CONFIGS.get(model_type, MODEL_CONFIGS["auto"])
        self.model_config = model_config
        
        # 获取模型模板配置
        self.model_template = MODEL_TEMPLATES.get(model_type, MODEL_TEMPLATES["default"])
        
        # 获取防御系统提示
        self.system_prompt = DEFENSE_SYSTEM_PROMPTS.get(
            PROMPT_STRATEGY["system_prompt_key"], 
            DEFENSE_SYSTEM_PROMPTS["default"]
        )
        
        # 获取提示策略
        self.use_system_prompt = PROMPT_STRATEGY["use_system_prompt"]
        self.instruction_template = PROMPT_STRATEGY["instruction_template"]
        
        # 动态加载tokenizer
        tokenizer_class = getattr(__import__('transformers'), model_config["tokenizer_class"])
        self.tokenizer = tokenizer_class.from_pretrained(
            tokenizer_path if tokenizer_path else self.config.model_name_or_path,
            use_fast=False
        )
        
        # 应用特殊标记
        for token_name, token_value in model_config["special_tokens"].items():
            if token_value:
                setattr(self.tokenizer, token_name, token_value)
        
        # 设置通用参数
        self.padding = True
        self.max_length = self.config.block_size
        self.pad_to_multiple_of = 8
        self.return_tensors = "pt"
        self.add_special_tokens = True
        self.stop_sequences = model_config["stop_sequences"]
    
    def batch_decode(self, model_output):
        return self.tokenizer.batch_decode(model_output, skip_special_tokens=True)

    def early_truncation(self, text):
        for stop in self.stop_sequences:
            stop_ix = text.find(stop)
            if stop_ix >= 0:
                text = text[:stop_ix].strip()
        return text.strip()
    
    def train_data_collator(self, features):
        samples_num = len(features)
        training_stage = self.training_stage
        origin_state = (self.tokenizer.padding_side, self.tokenizer.truncation_side)

        # 根据任务类型设置截断策略
        if self.task_type == "hh":
            self.tokenizer.truncation_side = "left"
        else:
            self.tokenizer.truncation_side = "right"

        ps = []
        ss = []
        rs = []
        sft_index = []
        
        for feature_index, feature in enumerate(features):
            # 取前training_stage个prefix、suffix和reward
            for p, s, r in zip(feature['prefix'][:training_stage], feature['suffix'][:training_stage], feature['reward'][:training_stage]):
                # 应用任务特定的格式化
                if self.task_type == "hh":
                    # HH任务特殊处理
                    p = "".join(p)
                    p = p.replace("<|prompter|>", "\n\nHuman: ").replace("<|assistant|>", "\n\nAssistant: ").rstrip()
                else:
                    # 代码和摘要任务使用instruction模板
                    p = self.instruction_template.format(code=p)
                
                ps.append(p)
                ss.append(s)  # 保持原始suffix格式
                rs.append(r)
            
            assert feature["sft_index"] < training_stage
            sft_index.append(feature["sft_index"])

        # 计算prefix长度
        ps_input_ids = self.tokenizer(
            ps,
            add_special_tokens=self.add_special_tokens,
        )['input_ids']
        ps_lens = [len(p_input_ids)-1 for p_input_ids in ps_input_ids]
        
        # 设置tokenizer为右截断进行完整文本处理
        self.tokenizer.padding_side = "right"
        self.tokenizer.truncation_side = "right"
        
        # 拼接prefix和suffix
        texts = []
        for p, s in zip(ps, ss):
            if self.task_type == "coding":
                # 代码任务直接拼接
                texts.append(p + s)
            else:
                # 其他任务用空格分隔
                texts.append(p + " " + s)
        
        # Tokenize完整文本
        batch = self.tokenizer(
            texts,
            padding=self.padding,
            max_length=self.max_length,
            truncation=True,
            add_special_tokens=self.add_special_tokens,
            return_tensors=self.return_tensors,
        )
        
        # 生成prefix mask
        seq_len = batch["attention_mask"].shape[1]
        prefix_mask = []
        for p_len in ps_lens:
            assert seq_len > p_len
            prefix_mask.append(
                [1 if i<p_len else 0 for i in range(seq_len)]
            )
        batch["prefix_mask"] = torch.tensor(prefix_mask)
        
        # 创建labels
        batch['labels'] = batch["input_ids"].clone().detach()
        
        # 重组数据为3D格式
        for key in batch:
            if key not in ['rewards', 'sft_index']:
                batch[key] = batch[key].view(samples_num, training_stage, -1)
        
        batch['rewards'] = torch.tensor(rs).view(samples_num, training_stage)
        batch['sft_index'] = torch.tensor(sft_index)
        
        # 恢复tokenizer状态
        self.tokenizer.padding_side, self.tokenizer.truncation_side = origin_state

        return batch

    def load_train_data(self, data_collator, data_file_path, data_file_name=None, extension='json', stream=None):
        from datasets import load_dataset
        
        # 直接加载train split
        raw_datasets = load_dataset(
            extension, 
            data_dir=data_file_path, 
            data_files=data_file_name, 
            streaming=True if stream else False, 
            split="train"
        )

        # 忽略类型检查错误，实际运行时可以正常工作
        dataloader = DataLoader(
            raw_datasets,  # type: ignore
            shuffle=True,
            collate_fn=data_collator, 
            batch_size=self.config.per_device_train_batch_size
        )

        return dataloader
    
    def infer_generate(self, model, prefixes):
        origin_state = (self.tokenizer.padding_side, self.tokenizer.truncation_side)
        
        # 设置推理时的tokenizer策略
        if self.task_type == "hh":
            self.tokenizer.padding_side = "left"
            self.tokenizer.truncation_side = "left"
        else:
            self.tokenizer.padding_side = "left"
            self.tokenizer.truncation_side = "right"
        
        # 预处理prefixes - 使用instruction模板
        new_prefixes = []
        for p in prefixes:
            if self.task_type == "hh":
                p = "".join(p)
                p = p.replace("<|prompter|>", "\n\nHuman: ").replace("<|assistant|>", "\n\nAssistant: ").rstrip()
            else:
                # 使用配置的instruction模板
                p = self.instruction_template.format(code=p)
            new_prefixes.append(p)
        
        # Tokenize prefixes
        batch = self.tokenizer(
            new_prefixes,
            padding=self.padding,
            max_length=self.max_length - 128,  # 预留生成空间
            truncation=True,
            add_special_tokens=self.add_special_tokens,
            return_tensors=self.return_tensors,
        ).to(model.device)
        
        # 生成
        with torch.no_grad():
            predicted_sents = model.generate(
                **batch, 
                max_new_tokens=128 if self.task_type == "coding" else 64,
                pad_token_id=self.tokenizer.pad_token_id,
                num_beams=1,
                do_sample=False,
                num_return_sequences=1,
            )
        
        # 解码和后处理
        instant_text = self.batch_decode(predicted_sents)
        
        # 恢复tokenizer状态
        self.tokenizer.padding_side, self.tokenizer.truncation_side = origin_state
        
        # 移除prefix部分
        for index in range(len(instant_text)):
            instant_text[index] = self.early_truncation(instant_text[index])
            
        return instant_text
