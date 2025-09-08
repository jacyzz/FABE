
import argparse
import json
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
import os

# ----------------- 临时路径解决方案 -----------------
# 为了能够从 FABE/inference 目录导入位于 FABE/PRO/train/utils 的模块
# 我们需要将 PRO 项目的根目录添加到 Python 路径中。
# 一个更长期的解决方案是创建一个共享的 'utils' 包。
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
pro_train_utils_path = os.path.join(project_root, 'PRO', 'train')
if pro_train_utils_path not in sys.path:
    sys.path.insert(0, pro_train_utils_path)
# ----------------------------------------------------

# 现在可以导入模板工具
from utils.templates import get_template

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="统一批量推理脚本")
    parser.add_argument("--base_model_path", type=str, required=True, help="基础模型的路径")
    parser.add_argument("--lora_model_path", type=str, required=True, help="LoRA 适配器权重路径")
    parser.add_argument("--input_file", type=str, required=True, help="输入数据文件路径 (.jsonl)")
    parser.add_argument("--output_file", type=str, required=True, help="输出结果文件路径 (.jsonl)")
    parser.add_argument("--model_template", type=str, required=True, help="用于推理的提示模板名称 (例如 'deepseek')")
    parser.add_argument("--batch_size", type=int, default=8, help="推理的批量大小")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="生成的最大新令牌数")
    return parser.parse_args()

def load_model_and_tokenizer(base_model_path, lora_model_path):
    """加载模型和分词器，并合并 LoRA 权重"""
    print("正在加载基础模型和分词器...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    
    # 加载基础模型 (使用 bfloat16 以提高效率)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    print("正在加载并合并 LoRA 适配器...")
    # 加载并合并 LoRA 权重
    model = PeftModel.from_pretrained(model, lora_model_path)
    model = model.merge_and_unload()
    
    model.eval()  # 设置为评估模式
    print("模型加载并合并完成。")
    return model, tokenizer

def process_batch(batch, model, tokenizer, template, max_new_tokens):
    """处理一个批次的数据进行推理"""
    # 从数据中提取 instruction 和 input
    instructions = [item.get('instruction', '') for item in batch]
    inputs = [item.get('input', '') for item in batch]

    # 使用模板格式化输入
    prompts = [template.format_prompt(instruction=inst, input=inp) for inst, inp in zip(instructions, inputs)]

    # 分词
    inputs_tokenized = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)

    # 生成
    with torch.no_grad():
        outputs_tokenized = model.generate(
            **inputs_tokenized,
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )

    # 解码
    # 我们只解码生成的部分，而不是整个序列
    input_lengths = inputs_tokenized.input_ids.shape[1]
    generated_tokens = outputs_tokenized[:, input_lengths:]
    results = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

    return results

def main():
    args = parse_arguments()

    # 获取提示模板
    template = get_template(args.model_template)
    if template is None:
        print(f"错误: 无法找到名为 '{args.model_template}' 的模板。")
        sys.exit(1)

    # 加载模型
    model, tokenizer = load_model_and_tokenizer(args.base_model_path, args.lora_model_path)

    # 读取输入数据
    try:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f]
    except FileNotFoundError:
        print(f"错误: 输入文件 '{args.input_file}' 未找到。")
        sys.exit(1)

    # 批量推理并写入结果
    with open(args.output_file, 'w', encoding='utf-8') as f_out:
        for i in range(0, len(data), args.batch_size):
            batch = data[i:i + args.batch_size]
            print(f"正在处理批次 {i // args.batch_size + 1}/{(len(data) + args.batch_size - 1) // args.batch_size}...")
            
            generated_texts = process_batch(batch, model, tokenizer, template, args.max_new_tokens)
            
            # 将生成结果与原始数据合并并写入文件
            for original_item, generated_text in zip(batch, generated_texts):
                result_item = original_item.copy()
                result_item['generated_output'] = generated_text
                f_out.write(json.dumps(result_item, ensure_ascii=False) + '\n')
    
    print(f"推理完成，结果已保存到 '{args.output_file}'。")

if __name__ == "__main__":
    main()
