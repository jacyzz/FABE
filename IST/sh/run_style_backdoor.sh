#!/bin/bash

# Simple Style Backdoor Dataset Generator Runner
# 使用IST风格转换生成后门清洗数据集

# 默认参数
INPUT_PATH="/home/nfs/share-yjy/dachuang2025/data/Code_Search/CodeSearchNet/python/train.jsonl"
OUTPUT_PATH="/home/nfs/share-yjy/dachuang2025/data/fabe/PRO/csn_train.jsonl"
LANGUAGE="python"
METHOD="partial_fix"
DATANAME="CSN"
LIMIT=0
INSTRUCTION=""

cd ..

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -i) INPUT_PATH="$2"; shift 2 ;;
        -o) OUTPUT_PATH="$2"; shift 2 ;;
        -l) LANGUAGE="$2"; shift 2 ;;
        -m) METHOD="$2"; shift 2 ;;
        -n) DATANAME="$2"; shift 2 ;;
        -s) LIMIT="$2"; shift 2 ;;
        --instruction) INSTRUCTION="$2"; shift 2 ;;
        -h) show_help; exit 0 ;;
        *) echo "Unknown option: $1"; show_help; exit 1 ;;
    esac
done

# 检查输入文件
if [ ! -f "$INPUT_PATH" ]; then
    echo "Error: Input file not found: $INPUT_PATH"
    exit 1
fi

# 创建输出目录
mkdir -p "$(dirname "$OUTPUT_PATH")"

# 构建命令
CMD="conda run -n IST python style_backdoor_dataset.py"
CMD="$CMD --input_path '$INPUT_PATH'"
CMD="$CMD --output_path '$OUTPUT_PATH'"
CMD="$CMD --language '$LANGUAGE'"
CMD="$CMD --method '$METHOD'"
CMD="$CMD --dataname '$DATANAME'"
CMD="$CMD --limit $LIMIT"

if [ -n "$INSTRUCTION" ]; then
    CMD="$CMD --instruction '$INSTRUCTION'"
fi

# 显示配置
echo "=== Style Backdoor Dataset Generator ==="
echo "Input:    $INPUT_PATH"
echo "Output:   $OUTPUT_PATH"
echo "Language: $LANGUAGE"
echo "Method:   $METHOD"
echo "Dataname: $DATANAME"
echo "Limit:    $LIMIT"
echo "Command:  $CMD"
echo "========================================="

# 执行命令
echo "Running..."
eval "$CMD"

# 检查结果
if [ $? -eq 0 ] && [ -f "$OUTPUT_PATH" ]; then
    SAMPLES=$(wc -l < "$OUTPUT_PATH")
    echo "✅ Success! Generated $SAMPLES samples in $OUTPUT_PATH"
else
    echo "❌ Failed to generate dataset"
    exit 1
fi