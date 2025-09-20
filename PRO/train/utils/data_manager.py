from datasets import load_dataset
from datasets import Dataset
from utils.config import args
from dataclasses import dataclass
import math
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from transformers import (
    AutoConfig,
    AutoTokenizer,
    LlamaTokenizer,
    GPT2Tokenizer,
    DataCollatorWithPadding,
)

from datasets import load_dataset
from datasets import Dataset
from utils.config import args
from .templates import get_template, is_chat_template, get_chat_system_prompt  # Import the new template manager
from dataclasses import dataclass
import math
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from transformers import (
    AutoConfig,
    AutoTokenizer,
    LlamaTokenizer,
    GPT2Tokenizer,
    DataCollatorWithPadding,
)

class Coding_DataManager():
    def __init__(self, config, training_stage, tokenizer_path = None, task_type = "coding"):
        # 导入全局args配置
        from .config import args
        self.args = args  # 保存args引用
        
        if tokenizer_path is None:
            tokenizer_path = args.model_name_or_path
            task_type = getattr(args, 'task', 'coding')  # 获取任务类型
        
        self.config = config
        self.task_type = task_type  # 保存任务类型
        
        # 优先从resume_adapter_path加载checkpoint内的tokenizer，避免vocab不一致；失败则回退到基座
        preferred_paths = []
        if getattr(args, 'resume_adapter_path', None):
            preferred_paths.append(args.resume_adapter_path)
        preferred_paths.append(tokenizer_path)
        loaded = False
        for path in preferred_paths:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(path, use_fast=False)
                loaded = True
                break
            except Exception:
                continue
        if not loaded:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False)
        
        # Set pad token if it's not already set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.padding = True
        self.max_length = self.args.block_size
        self.pad_to_multiple_of = 8
        self.return_tensors = "pt"
        self.add_special_tokens = True
        self.training_stage = training_stage
        self.template_name = self.args.model_template
        # 若为字符串模板，预取字符串；若为chat模板，训练时用apply_chat_template
        if not is_chat_template(self.template_name):
            self.template = get_template(self.template_name)

    def train_data_collator(self, features):
        samples_num = len(features)
        training_stage = self.training_stage
        origin_state = (self.tokenizer.padding_side, self.tokenizer.truncation_side)

        prompts = []
        responses = []
        rewards = []
        sft_indices = []

        for feature in features:
            # Format the prompt
            instruction_text = feature.get('instruction', '')
            input_text = feature.get('input', '')
            if is_chat_template(self.template_name):
                # 构造 messages，系统提示放清洗要求，用户放纯代码
                system_prompt = get_chat_system_prompt(self.template_name)
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": input_text if input_text is not None else ""}
                ]
                # 使用 tokenizer 的原生 chat_template 进行拼接
                prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            else:
                # 字符串模板
                if self.args.model_template == 'deepseek' and (input_text is None or str(input_text).strip() == ''):
                    prompt = self.template.format(instruction=instruction_text, input="").replace("### Input:\n```\n\n```\n", "")
                else:
                    prompt = self.template.format(instruction=instruction_text, input=input_text)
            
            # The number of responses per prompt is determined by the training stage
            num_responses = min(training_stage, len(feature['output']))
            
            for i in range(num_responses):
                prompts.append(prompt)
                responses.append(feature['output'][i])
                rewards.append(feature['score'][i])
            
            # SFT index is always 0 for the highest-ranked response
            sft_indices.append(0)

        # Tokenize prompts to calculate their length for masking
        self.tokenizer.truncation_side = "left"
        prompt_tokens = self.tokenizer(
            prompts,
            max_length=self.max_length - 128,  # Reserve space for generation
            truncation=True,
            add_special_tokens=self.add_special_tokens,
        )
        prompt_lens = [len(ids) for ids in prompt_tokens['input_ids']]

        # Tokenize the full sequences (prompt + response)
        self.tokenizer.padding_side = "right"
        self.tokenizer.truncation_side = "right"
        
        full_texts = [p + r + self.tokenizer.eos_token for p, r in zip(prompts, responses)]
        
        batch = self.tokenizer(
            full_texts,
            padding=self.padding,
            max_length=self.max_length,
            truncation=True,
            add_special_tokens=self.add_special_tokens,
            return_tensors=self.return_tensors,
        )
        
        # Create prefix mask to distinguish prompt from response
        seq_len = batch["attention_mask"].shape[1]
        prefix_mask = torch.zeros_like(batch["attention_mask"])
        for i, p_len in enumerate(prompt_lens):
            prefix_mask[i, :p_len] = 1
        
        batch["prefix_mask"] = prefix_mask
        
        # Create labels for loss calculation (mask out prompt tokens)
        batch['labels'] = batch["input_ids"].clone().detach()
        batch['labels'][prefix_mask.bool()] = -100 # Use -100 to ignore prompt part in loss

        # Reshape tensors to (batch_size, num_candidates, seq_len)
        for key in batch:
            batch[key] = batch[key].view(samples_num, training_stage, -1)
        
        batch['rewards'] = torch.tensor(rewards).view(samples_num, -1)
        batch['sft_index'] = torch.tensor(sft_indices)
        
        self.tokenizer.padding_side, self.tokenizer.truncation_side = origin_state
        return batch

    def load_train_data(
        self, 
        data_collator, 
        data_file_path, 
        data_file_name=None, # This argument is kept for compatibility but not used if data_file_path is a list
        extension='json', 
        stream = None, 
    ):
        # The `load_dataset` function can handle a list of files directly
        raw_datasets = load_dataset(extension, data_files=data_file_path, streaming=bool(stream), split="train")

        dataloader = DataLoader(
            raw_datasets, 
            shuffle=not stream, # Streaming datasets cannot be shuffled
            collate_fn=data_collator, 
            batch_size=self.args.per_device_train_batch_size,
            num_workers=8,
            pin_memory=True,
            persistent_workers=True
        )
        return dataloader
    
    def infer_generate(self, model, instructions, inputs):
        origin_state = (self.tokenizer.padding_side, self.tokenizer.truncation_side)
        self.tokenizer.padding_side = "left"
        self.tokenizer.truncation_side = "left"
        
        prompts = [self.template.format(instruction=inst, input=inp) for inst, inp in zip(instructions, inputs)]

        # Truncate prompts from the left to fit within the model's context window
        prompt_tokens = self.tokenizer(
            prompts,
            max_length=self.max_length - 128, # Reserve space for generation
            truncation=True,
            add_special_tokens=self.add_special_tokens,
            return_tensors=self.return_tensors,
        ).to(model.device)

        with torch.no_grad():
            generated_tokens = model.generate(
                **prompt_tokens, 
                max_new_tokens=128,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                num_beams=1,
                do_sample=False,
                num_return_sequences=1,
            )
        
        # Decode only the newly generated part
        prompt_len = prompt_tokens['input_ids'].shape[1]
        decoded_responses = self.tokenizer.batch_decode(generated_tokens[:, prompt_len:], skip_special_tokens=True)
        
        self.tokenizer.padding_side, self.tokenizer.truncation_side = origin_state
        return decoded_responses
