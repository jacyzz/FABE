import os
import json
from utils.config import init_args
import math
from scipy.stats import poisson
from tqdm import tqdm
import numpy as np
import torch
from accelerate.logging import get_logger
from .data_manager import Coding_DataManager
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from transformers.models.auto.configuration_auto import CONFIG_MAPPING, AutoConfig
from transformers.models.auto.modeling_auto import MODEL_MAPPING, AutoModelForCausalLM
from transformers.utils.quantization_config import BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType
args = init_args()

class ProcessManager():
    def __init__(
        self,
        accelerator,
        model_path = args.model_name_or_path,
    ):
        self.accelerator = accelerator
        self.model_path = model_path
        

        # Set model config
        self.model_config = AutoConfig.from_pretrained(self.model_path)
        
        # Set datamanager - 统一使用Coding_DataManager
        if args.task == "coding":
            self.data_manager = Coding_DataManager(
                self.model_config,
                args.training_stage_num,
                task_type="coding"
            )
        elif args.task == "hh":
            # 对于HH任务，也使用Coding_DataManager
            self.data_manager = Coding_DataManager(
                self.model_config,
                args.training_stage_num,
                tokenizer_path=model_path,
                task_type="hh"
            )
        elif args.task == "summarize":
            # 对于总结任务，也使用Coding_DataManager
            self.data_manager = Coding_DataManager(
                self.model_config,
                args.training_stage_num,
                tokenizer_path=model_path,
                task_type="summarize"
            )
        else:
            # 使用Coding_DataManager支持其他任务
            self.data_manager = Coding_DataManager(
                self.model_config,
                args.training_stage_num,
                tokenizer_path=model_path,
                task_type=args.task
            )
        
        # Load base model with optional quantization
        load_kwargs = {
            "config": self.model_config,
            "torch_dtype": torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32)
        }
        
        # Add 4-bit quantization if enabled
        if hasattr(args, 'use_4bit') and args.use_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=getattr(args, 'bnb_4bit_quant_type', "nf4"),
                bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
                bnb_4bit_use_double_quant=getattr(args, 'bnb_4bit_use_double_quant', False)
            )
            load_kwargs["quantization_config"] = quantization_config
            load_kwargs["device_map"] = "auto"
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            **load_kwargs
        )
        
        # Add LoRA if enabled
        if args.use_lora:
            lora_config = LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha, 
                target_modules=args.lora_target_modules,
                lora_dropout=args.lora_dropout,
                bias="none",
                task_type=TaskType.CAUSAL_LM
            )
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()
        
        # 只扩不缩，避免与基座模型词表不一致导致 LoRA 适配器加载报错
        vocab_size_model = self.model.get_input_embeddings().num_embeddings
        vocab_size_tok = len(self.data_manager.tokenizer)
        if vocab_size_tok > vocab_size_model:
            self.model.resize_token_embeddings(vocab_size_tok)
        self.logger = get_logger(__name__)
    
    def compute_loss(self, model, batch, print_loss):
        # 添加维度检查
        # print(f"Batch shapes:")
        # for key, value in batch.items():
        #     if isinstance(value, torch.Tensor):
        #         print(f"{key}: {value.shape}")
        
        batch_size = batch["labels"].shape[0]
        temp_training_stage = batch["labels"].shape[1]
        
        # 确保批次大小大于0
        if batch_size == 0:
            raise ValueError("Empty batch detected")
        
        sub_batches = [{key: batch[key][:,time,:] for key in ["input_ids", "attention_mask"]} for time in range(temp_training_stage)]
        
        score_list = []
        suffix_mask_list = []
        for batch_index, sub_batch in enumerate(sub_batches):
            local_outputs = model(**sub_batch, return_dict=True)
            local_logits = local_outputs.logits #[batch, seq_len, token_num]
            local_mask = sub_batch["attention_mask"] & (~batch["prefix_mask"][:, batch_index, :]) #[batch, seq_len]
            local_labels = batch["labels"][:, batch_index, :]

            # Shift
            shift_logits = local_logits[..., :-1, :].contiguous() #[batch, seq_len-1, token_num]
            shift_logits = F.log_softmax(shift_logits, dim=2) #[batch, seq_len-1, token_num]
            shift_masks = local_mask[..., :-1] #[batch, seq_len-1]
            shift_labels = local_labels[..., 1:].view(batch_size, -1, 1) #[batch, seq_len-1, 1]

            # Handle -100 labels (ignore_index) to prevent CUDA errors
            vocab_size = shift_logits.size(-1)
            ignore_mask = (shift_labels == -100)
            
            # Replace -100 with 0 for gather operation (will be masked out later)
            safe_labels = torch.where(ignore_mask, torch.zeros_like(shift_labels), shift_labels)
            
            # Clamp any remaining out-of-bounds labels to valid range
            safe_labels = torch.clamp(safe_labels, 0, vocab_size - 1)

            selected_logits = torch.gather(input=shift_logits, dim=2, index=safe_labels).view(batch_size, -1) #[batch, seq_len-1]
            
            # Apply both attention mask and ignore mask
            ignore_mask_flat = ignore_mask.view(batch_size, -1)
            combined_mask = (shift_masks == 1) & (~ignore_mask_flat)
            selected_logits[~combined_mask] = 0.0 #[batch, seq_len-1]
            sentence_logits = torch.sum(selected_logits, dim=1) #[batch]
            sentence_logits = sentence_logits.view(batch_size, 1)
            score_list.append(sentence_logits)
            suffix_mask_list.append(torch.sum(combined_mask.float(), dim=1).view(batch_size, 1))
        
        sum_scores = torch.cat(score_list, dim=1) #[batch, training_stage]
        suffix_mask = torch.cat(suffix_mask_list, dim=1) #[batch, training_stage]
        scores = sum_scores / suffix_mask #[batch, training_stage]
        total_loss = 0
        for time in range(temp_training_stage - 1):
            neg_reward = batch["rewards"][:, time+1:] # [batch, training_stage-time-1]
            pos_reward = batch["rewards"][:, time] # [batch]
            
            eps = 1e-10
            neg_temperatures = pos_reward.view(-1, 1) - neg_reward # [batch, training_stage-time-1]
            pos_temperature = torch.max(neg_temperatures, dim=1).values # [batch]
            loss = torch.log(eps + torch.exp(scores[:, time] * pos_temperature) + torch.sum(torch.exp(scores[:, time+1:] * neg_temperatures), dim=1)) - scores[:, time] * pos_temperature # [batch]
            loss = torch.mean(loss).to(scores.dtype)
            
            print_loss[time].append(loss.item())
            total_loss += loss
        
        sft_index = batch["sft_index"].view(batch_size, 1)
        sft_scores = torch.gather(input = sum_scores, dim = 1, index = sft_index).view(batch_size) #[batch]
        sft_loss = torch.mean(-sft_scores).to(scores.dtype)
        sft_loss = args.sft_weight * math.pow(temp_training_stage - 1, 2) * sft_loss
        total_loss += sft_loss

        print_loss[-1].append(sft_loss.item())
        self.accelerator.backward(total_loss)

    def prepare_hfa_dataloader(self, train_file_path=None, train_file_name=None):
        # Use the file path from args if not provided
        if train_file_path is None:
            train_file_path = args.train_file_path
        
        # Handle different path formats based on user requirements:
        # 1. Directory -> load all .json/.jsonl files in the directory
        # 2. Single file -> load only that file
        # 3. Glob pattern -> load all matching files
        
        if isinstance(train_file_path, str):
            if os.path.isdir(train_file_path):
                # Directory: load all .json/.jsonl files
                import glob
                json_files = glob.glob(os.path.join(train_file_path, "*.json"))
                jsonl_files = glob.glob(os.path.join(train_file_path, "*.jsonl"))
                data_file_path = json_files + jsonl_files
                if not data_file_path:
                    raise ValueError(f"No .json or .jsonl files found in directory: {train_file_path}")
                self.logger.info(f"Directory mode: Loading {len(data_file_path)} files from {train_file_path}")
            elif os.path.isfile(train_file_path):
                # Single file: load only this file
                if not (train_file_path.endswith('.json') or train_file_path.endswith('.jsonl')):
                    self.logger.warning(f"File extension may not be supported: {train_file_path}")
                data_file_path = train_file_path
                self.logger.info(f"Single file mode: Loading {train_file_path}")
            elif '*' in train_file_path:
                # Glob pattern: load all matching files
                data_file_path = train_file_path
                self.logger.info(f"Glob pattern mode: Loading files matching {train_file_path}")
            else:
                # Path doesn't exist - assume it will be created or is valid
                data_file_path = train_file_path
                self.logger.warning(f"Path not found, using as-is: {train_file_path}")
        elif isinstance(train_file_path, list):
            # List of files (for backward compatibility)
            data_file_path = train_file_path
            self.logger.info(f"List mode: Loading {len(data_file_path)} files")
        else:
            raise ValueError(f"Invalid train_file_path type: {type(train_file_path)}")
        
        self.logger.info(f"Load training data from {data_file_path}")
        self.accelerator.print(f"Load training data from {data_file_path}")
        
        hfa_dataloader = self.data_manager.load_train_data(
            data_collator=self.data_manager.train_data_collator,
            data_file_path=data_file_path,
            data_file_name=train_file_name
        )
        
        # wrap with accelerator
        hfa_dataloader = self.accelerator.prepare(
            hfa_dataloader
        )

        return hfa_dataloader

    def init_prepare_train(self, train_file_name = None):
        # 处理训练文件路径 - 使用与 prepare_hfa_dataloader 相同的逻辑
        train_file_path = args.train_file_path
        
        # 根据路径类型确定训练文件列表和数据长度
        if isinstance(train_file_path, str):
            if os.path.isdir(train_file_path):
                # 目录：获取所有 .json/.jsonl 文件
                import glob
                json_files = glob.glob(os.path.join(train_file_path, "*.json"))
                jsonl_files = glob.glob(os.path.join(train_file_path, "*.jsonl"))
                train_files = json_files + jsonl_files
                if not train_files:
                    raise ValueError(f"No .json or .jsonl files found in directory: {train_file_path}")
                # 使用第一个文件来估算数据长度
                try:
                    with open(train_files[0], 'r', encoding='utf-8') as f:
                        dataset_length = sum(1 for _ in f)
                except:
                    dataset_length = 1000  # 默认值
            elif os.path.isfile(train_file_path):
                # 单个文件
                train_files = [train_file_path]
                try:
                    with open(train_file_path, 'r', encoding='utf-8') as f:
                        dataset_length = sum(1 for _ in f)
                except:
                    dataset_length = 1000
            elif '*' in train_file_path:
                # Glob模式
                import glob
                train_files = glob.glob(train_file_path)
                if train_files:
                    try:
                        with open(train_files[0], 'r', encoding='utf-8') as f:
                            dataset_length = sum(1 for _ in f)
                    except:
                        dataset_length = 1000
                else:
                    train_files = []
                    dataset_length = 1000
            else:
                # 假设是有效路径
                train_files = [train_file_path]
                dataset_length = 1000
        elif isinstance(train_file_path, list):
            # 文件列表（向后兼容）
            train_files = train_file_path
            if train_files:
                try:
                    with open(train_files[0], 'r', encoding='utf-8') as f:
                        dataset_length = sum(1 for _ in f)
                except:
                    dataset_length = 1000
            else:
                dataset_length = 1000
        else:
            train_files = []
            dataset_length = 1000

        # get the placeholder dataloader
        placeholder_dataloader = self.data_manager.load_train_data(
            data_file_path = args.train_file_path,
            data_file_name = train_file_name,
            data_collator = self.data_manager.train_data_collator
        )
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.learning_rate)
        
        # Scheduler and math around the number of training steps.
        if args.max_train_steps is None:
            num_update_steps_per_epoch_per_train_file = math.ceil(
                math.ceil(
                    dataset_length / args.per_device_train_batch_size
                ) / args.gradient_accumulation_steps
            )
            args.max_train_steps = len(train_files) * num_update_steps_per_epoch_per_train_file * args.num_train_epochs
        

        model, optimizer, _ = self.accelerator.prepare(
            self.model, optimizer, placeholder_dataloader
        )
        # 不要将self.model设置为None，保持引用

        total_batch_size = args.per_device_train_batch_size * self.accelerator.num_processes * args.gradient_accumulation_steps
        self.logger.info("***** Running training *****", main_process_only=True)
        self.logger.info(f"  Num examples = {len(train_files) * dataset_length}", main_process_only=True)
        self.logger.info(f"  Num training stages = {args.training_stage_num}", main_process_only=True)
        self.logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}", main_process_only=True)
        self.logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}", main_process_only=True)
        self.logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}", main_process_only=True)
        
        return model, optimizer, dataset_length

    def train(self):
        # 使用与 prepare_hfa_dataloader 和 init_prepare_train 相同的逻辑
        train_file_path = args.train_file_path
        
        # 根据路径类型确定训练文件列表
        if isinstance(train_file_path, str):
            if os.path.isdir(train_file_path):
                # 目录：获取所有 .json/.jsonl 文件
                import glob
                json_files = glob.glob(os.path.join(train_file_path, "*.json"))
                jsonl_files = glob.glob(os.path.join(train_file_path, "*.jsonl"))
                train_files = json_files + jsonl_files
                if not train_files:
                    raise ValueError(f"No .json or .jsonl files found in directory: {train_file_path}")
            elif os.path.isfile(train_file_path):
                # 单个文件
                train_files = [train_file_path]
            elif '*' in train_file_path:
                # Glob模式
                import glob
                train_files = glob.glob(train_file_path)
            else:
                # 假设是有效路径
                train_files = [train_file_path] if train_file_path else []
        elif isinstance(train_file_path, list):
            # 文件列表（向后兼容）
            train_files = train_file_path
        else:
            train_files = []
        
        train_file_name = None  # 不再需要单独的 train_file_name
            
        model, optimizer, dataset_length = self.init_prepare_train(
            train_file_name = train_file_name
        )
        training_stage = args.training_stage_num
        if self.accelerator.is_main_process:
            writer = SummaryWriter(args.log_path)

        self.accelerator.wait_for_everyone()
        # Train!
        progress_bar = tqdm(
            range(
                len(train_files) * math.ceil(
                    math.ceil(
                        math.ceil(
                            dataset_length / args.per_device_train_batch_size
                        ) / self.accelerator.num_processes
                    ) / args.gradient_accumulation_steps
                ) * args.num_train_epochs
            ),
            disable=not self.accelerator.is_local_main_process
        )
        completed_steps = 0
        best_step = -1
        last_dev_reward = float('-inf')
        for epoch in range(args.num_train_epochs):
            if self.accelerator.is_main_process:
                self.logger.info(f"Epoch {epoch} starts")
                self.accelerator.print(f"\nEpoch {epoch} starts")
            train_file_path = args.train_file_path

            for train_file_index, train_file_name in enumerate(train_files):
                if_get_new_dataloader = False
                if len(train_files) == 1:
                    if epoch == 0:
                        if_get_new_dataloader = True
                    else:
                        pass
                else:
                    if_get_new_dataloader = True
                
                torch.cuda.empty_cache()
                if if_get_new_dataloader:
                    hfa_dataloader = None
                    hfa_dataloader = self.prepare_hfa_dataloader(train_file_path, train_file_name)
                
                print_loss = [[] for i in range(training_stage)]
                for step, batch in enumerate(hfa_dataloader):
                    # 跳过None的batch（过长的样本）
                    if batch is None:
                        continue
                    
                    model.train()
                    with self.accelerator.accumulate(model):
                        self.compute_loss(model, batch, print_loss)
                        optimizer.step()
                        optimizer.zero_grad()

                    if self.accelerator.sync_gradients:
                        completed_steps += 1
                        progress_bar.update(1)
                        self.accelerator.wait_for_everyone()
                        if self.accelerator.is_main_process:
                            print_loss = [sum(l) for l in print_loss]
                            print_loss = [l / self.accelerator.gradient_accumulation_steps for l in print_loss]
                            total_loss = sum(print_loss)
                            print_loss_info = "\nstage_{}_loss: {:.4f}".format(training_stage, total_loss)
                            
                            print_loss_info += "".join(
                                [" | rank_{}_loss: {:.4f}".format(n+1, print_loss[n]) for n in range(training_stage-1)]
                            )
                            print_loss_info += " | sft_loss: {:.4f}".format(print_loss[training_stage-1])
                            
                            for n in range(training_stage-1):
                                writer.add_scalar("stage_{}/rank_{}_loss".format(training_stage, n+1), print_loss[n], completed_steps)
                            writer.add_scalar("stage_{}/sft_loss".format(training_stage), print_loss[training_stage-1], completed_steps)
                            
                            self.logger.info(f"Step {completed_steps} | " + print_loss_info)
                            writer.add_scalar("stage_{}/loss".format(training_stage), total_loss, completed_steps) # record on tensorboard                      
                            # 按步保存（独立于验证逻辑）
                            if completed_steps % args.checkpointing_step == 0 or (step == len(hfa_dataloader)-1 and train_file_index == len(train_files)-1):
                                model_to_save = self.accelerator.unwrap_model(model)
                                self.save_checkpoint(model_to_save, self.data_manager.tokenizer, os.path.join(args.output_dir, 'step_{}'.format(completed_steps)))
                                model_to_save = None

                        print_loss = [[] for i in range(training_stage)]
                        self.accelerator.wait_for_everyone()
                torch.cuda.empty_cache()
            self.accelerator.wait_for_everyone()
            if self.accelerator.sync_gradients and self.accelerator.is_main_process:
                model_to_save = self.accelerator.unwrap_model(model)
                self.save_checkpoint(model_to_save,self.data_manager.tokenizer,os.path.join(args.output_dir, 'epoch_{}'.format(epoch)))
                self.logger.info(f"Epoch {epoch} checkpoint has been saved.")
                model_to_save = None
        if self.accelerator.is_main_process:
            writer.close()
        
        return self.accelerator.unwrap_model(model)

    def save_checkpoint(self, model, tokenizer, path):
        if path is not None and path != '':
            os.makedirs(path, exist_ok=True)
            tokenizer.save_pretrained(path)
            # 额外写入训练元数据，便于推理时自动对齐模板等设置
            try:
                meta = {
                    "model_template": getattr(args, 'model_template', 'default'),
                    "task": getattr(args, 'task', 'coding'),
                    "training_stage_num": getattr(args, 'training_stage_num', 1),
                    "sft_weight": getattr(args, 'sft_weight', 1.0)
                }
                with open(os.path.join(path, 'training_meta.json'), 'w', encoding='utf-8') as f:
                    json.dump(meta, f, ensure_ascii=False, indent=2)
            except Exception as e:
                self.logger.warning(f"Failed to write training_meta.json: {e}")
            if args.use_lora:
                model.save_pretrained(
                    path,
                    is_main_process=self.accelerator.is_main_process,
                    save_function=self.accelerator.save,
                )
            else:
                model.save_pretrained(
                    path,
                    is_main_process=self.accelerator.is_main_process,
                    save_function=self.accelerator.save,
                )
            
            # 实现save_total_limit功能
            if hasattr(args, 'save_total_limit') and args.save_total_limit > 0:
                self._cleanup_checkpoints(args.output_dir, args.save_total_limit)
        else:
            self.logger.error('No save path!', main_process_only=True)

    def _cleanup_checkpoints(self, output_dir, save_total_limit):
        """清理超过限制数量的checkpoint"""
        import glob
        import shutil
        
        # 获取所有step_*和epoch_*目录
        step_checkpoints = glob.glob(os.path.join(output_dir, 'step_*'))
        epoch_checkpoints = glob.glob(os.path.join(output_dir, 'epoch_*'))
        
        # 过滤掉符号链接和非目录文件
        step_checkpoints = [cp for cp in step_checkpoints if os.path.isdir(cp) and not os.path.islink(cp)]
        epoch_checkpoints = [cp for cp in epoch_checkpoints if os.path.isdir(cp) and not os.path.islink(cp)]
        
        # 按修改时间排序，最新的在前
        step_checkpoints.sort(key=os.path.getmtime, reverse=True)
        epoch_checkpoints.sort(key=os.path.getmtime, reverse=True)
        
        # 删除超过限制的step checkpoints
        if len(step_checkpoints) > save_total_limit:
            for checkpoint_to_remove in step_checkpoints[save_total_limit:]:
                self.logger.info(f"Removing old checkpoint: {checkpoint_to_remove}")
                shutil.rmtree(checkpoint_to_remove)
        
        # 删除超过限制的epoch checkpoints  
        if len(epoch_checkpoints) > save_total_limit:
            for checkpoint_to_remove in epoch_checkpoints[save_total_limit:]:
                self.logger.info(f"Removing old checkpoint: {checkpoint_to_remove}")
                shutil.rmtree(checkpoint_to_remove)
