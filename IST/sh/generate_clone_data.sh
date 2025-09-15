#!/bin/bash

# =====================================================================================
#
# 脚本说明:
#
#   此脚本使用 universal_data_transformer.py 从 clone-dectect 数据集生成
#   通用的、与模型无关的训练数据。
#
#   - 输入: /home/nfs/u2023-zlb/datasets/clone-dectect/filtered/ 目录下的 .jsonl 文件
#   - 输出: /home/nfs/u2023-zlb/datasets/universal_clone_detect_format/ 目录下的 .jsonl 文件
#   - 核心逻辑:
#     1. 读取输入文件中的 "func1" 字段作为原始代码。
#     2. 使用 IST (Implicit Style Transformer) 对代码应用多种风格转换，生成排名数据。
#     3. 将结果保存为通用的 {"instruction": "...", "input": "...", "output": [...], "score": [...]} 格式。
#
# =====================================================================================

# --- 配置参数 ---

# 输入数据集目录
# 包含 train-*.jsonl, test-*.jsonl, validation-*.jsonl 文件
INPUT_DATA_DIR="/home/nfs/u2023-zlb/datasets/clone-dectect/filtered/"

# 输出目录
# 用于存放生成的通用格式数据
OUTPUT_DATA_DIR="/home/nfs/u2023-zlb/datasets/universal_clone_detect_format/"

# 确保输出目录存在
mkdir -p $OUTPUT_DATA_DIR

echo "开始生成通用训练数据..."
echo "输入目录: $INPUT_DATA_DIR"
echo "输出目录: $OUTPUT_DATA_DIR"

# --- 执行数据转换 ---

python /home/nfs/u2023-zlb/FABE/IST/universal_data_transformer.py \
    --input_dir "$INPUT_DATA_DIR" \
    --output_dir "$OUTPUT_DATA_DIR" \
    --language "java" \
    --instruction "Please refactor the following code to improve its structure and style while maintaining the original functionality." \
    --code_field "func1" \
    --code_field2 "func2" \
    --output_format "universal" \
    --rank_len 4 \
    --verbose

# --- 完成 ---

echo "数据生成完成！"
echo "通用格式的训练数据已保存到: $OUTPUT_DATA_DIR"

