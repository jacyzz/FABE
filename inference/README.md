# FABE批量推理框架 (FABE Batch Inference Framework)

## 概述

FABE是一个批量推理框架，专门用于通过大型语言模型（LLM）批量处理数据集记录，根据自定义指令/模板转换目标字段（如代码），然后将转换后的字段写回，同时保留原始结构。

## 主要特性

- **多种输入格式**: 支持单个JSON/JSONL文件或包含此类文件的目录
- **多模型支持**: 通过可插拔提供者支持多种模型（echo、Ollama、OpenAI兼容HTTP、Hugging Face本地模型、ModelScope模型）
- **自定义指令/模板**: 支持Jinja2模板和简单的后处理（如去除代码围栏）
- **智能缓存**: 基于内容哈希进行缓存，避免重复处理
- **并发控制和重试**: 支持并发处理和基本的重试机制
- **批处理优化**: 支持批量推理，大幅提高大规模数据集处理效率
- **模板自动应用**: 自动检测和应用模型特定的chat_template

## 快速开始

### 1. 安装

```bash
# 创建虚拟环境（可选）
python -m venv venv
source venv/bin/activate

# 安装依赖
pip install -e .
```

### 2. 干运行（检查将要处理的记录）

```bash
fabe-infer run \
  --input path/to/data.jsonl \
  --field code \
  --instruction "请重构代码以提高可读性。" \
  --provider echo \
  --model local
```

### 3. 实际运行（使用Ollama）

```bash
fabe-infer run \
  --input path/to/dir \
  --glob "*.jsonl" \
  --field code \
  --instruction examples/prompts/refactor_python.j2 \
  --provider ollama \
  --model deepseek-coder:6.7b \
  --output outputs/ \
  --concurrency 4 \
  --batch-size 8
```

### 4. 使用ModelScope模型

```bash
fabe-infer run \
  --input dataset.jsonl \
  --field code \
  --instruction prompts/refactor.j2 \
  --provider modelscope \
  --model model_name_or_path \
  --batch-size 16 \
  --concurrency 4 \
  --device cuda
```

## 详细参数说明

### 输入/输出参数 (I/O)

- `--input`: 输入文件或目录路径（必需）
- `--glob`: 当输入为目录时的文件匹配模式（如 `*.jsonl`）
- `--output`: 输出目录路径（非inplace模式时使用）
- `--inplace`: 是否就地修改输入文件（默认: false）
- `--field`: 需要转换的目标字段名（默认: "code"）
- `--backup-field`: 备份原始内容的字段名（默认: "original_code"）

### 提示/模板参数 (Prompt)

- `--instruction`: 指令文本或模板文件路径（必需）
- `--template-var`: 模板变量，格式为k=v，可重复使用（如 `--template-var language=python --template-var goal=refactor`）
- `--strip-fences`: 是否提取代码块或去除围栏（默认: true）
- `--auto-extract-code`: 是否自动提取代码（默认: true，与strip-fences相同）

### 模型参数 (Model)

- `--provider`: 模型提供者（默认: "echo"）
  - `echo`: 测试用，返回带标题的原始代码
  - `ollama`: 需要本地Ollama服务（http://localhost:11434）
  - `openai`: OpenAI兼容的HTTP端点
  - `hf`: Hugging Face本地模型
  - `modelscope`: ModelScope模型（支持自动下载）
- `--model`: 模型名称或路径（必需）
- `--base-url`: 远程服务的base URL（Ollama/OpenAI兼容）
- `--api-key`: 远程服务的API密钥（OpenAI兼容）
- `--device`: 设备类型（默认: "cpu"，可选: "cuda"）
- `--local-files-only`: 是否仅使用本地文件（默认: true）
- `--max-new-tokens`: 生成的最大token数（默认: 128）
- `--temperature`: 采样温度（默认: 0.0）
- `--top-p`: nucleus采样参数（可选）
- `--do-sample`: 是否进行采样（默认: 根据温度自动判定）
- `--use-chat-template`: 是否使用模型的chat_template（默认: true）
- `--system-prompt`: 系统提示词（默认: "You are a coding assistant. Output only the final code."）

### 执行参数 (Execution)

- `--concurrency`: 并发任务数（默认: 4）
- `--batch-size`: 批处理大小（默认: 8，仅支持批处理的提供者有效）
- `--retry`: 失败重试次数（默认: 3）
- `--cache-dir`: 缓存目录（默认: ".cache"）
- `--resume`: 是否从缓存恢复（默认: true）
- `--dry-run`: 是否仅检查而不实际处理（默认: false）

## 配置文件和示例

### 使用YAML配置文件

```yaml
# examples/config.yaml
provider: modelscope
model: your-model-name
input_path: ../data
input_glob: "*.jsonl"
field: code
output_path: ./outputs
instruction: ./prompts/refactor_python.j2
template_vars:
  language: python
  goal: improve readability and remove dead code

concurrency: 4
batch_size: 16
retry: 3
cache_dir: .cache
resume: true
strip_fences: true
auto_extract_code: true
```

```bash
fabe-infer run --config examples/config.yaml
```

### 示例模板

```jinja2
# examples/prompts/refactor_python.j2
You are a helpful senior Python engineer. 
Goal: {{ goal | default('improve readability') }}.
Language: {{ language | default('python') }}.

Rules:
- Output ONLY the final code, no explanations.
- Keep functionality unchanged unless obviously dead code.
- Use clear naming and add minimal comments if essential.

Transform the following code accordingly.
```

## 提供者详细说明

### Echo提供者
- **用途**: 测试和调试
- **行为**: 返回原始代码加上简单标题
- **示例**: `--provider echo --model local`

### Ollama提供者
- **要求**: 本地运行Ollama服务（默认: http://localhost:11434）
- **示例**: `--provider ollama --model deepseek-coder:6.7b --base-url http://localhost:11434`

### OpenAI兼容提供者
- **要求**: OpenAI兼容的API端点
- **示例**: `--provider openai --model gpt-3.5-turbo --base-url https://api.openai.com/v1 --api-key your-api-key`

### Hugging Face本地提供者
- **要求**: 本地Hugging Face模型路径
- **示例**: `--provider hf --model /path/to/model --device cuda`

### ModelScope提供者
- **要求**: ModelScope模型ID或本地路径（支持自动下载）
- **特性**: 自动应用模型特定的chat_template，针对大规模数据集优化
- **示例**: `--provider modelscope --model model_name --device cuda --batch-size 16`

## 高级用法

### 批量处理大规模数据集

对于20万+记录的大规模数据集，推荐使用以下配置：

```bash
fabe-infer run \
  --input large_dataset.jsonl \
  --field code \
  --instruction prompts/refactor.j2 \
  --provider modelscope \
  --model your-large-model \
  --batch-size 16 \
  --concurrency 2 \
  --device cuda \
  --max-new-tokens 256 \
  --cache-dir .cache_large \
  --resume
```

### 使用Shell脚本

```bash
# 使用优化后的shell脚本
sh/run_infer.sh \
  -m /path/to/model \
  -i dataset.jsonl \
  -o outputs \
  -d cuda \
  -P modelscope \
  -S 16 \
  -C 4 \
  -n 256 \
  -t 0.1
```

### 环境变量

- `MODEL_PATH`: 默认模型路径
- `DEVICE`: 默认设备（cpu/cuda）
- `LOCAL_FILES_ONLY`: 是否仅使用本地文件（1/0）
- `MAX_NEW_TOKENS`: 默认生成token数
- `TEMPERATURE`: 默认温度
- `DRY_RUN`: 是否干运行（1/0）

## 注意事项

1. **JSON格式**: 支持对象列表或单个对象
2. **JSONL格式**: 每行一个JSON对象
3. **空字段处理**: 缺失或空的目标字段会被跳过并记录日志
4. **输出结构**: 输出目录会镜像输入目录结构
5. **缓存机制**: 基于提供者名称、模型名称、指令和内容哈希进行缓存
6. **内存管理**: 处理大规模数据集时会定期进行垃圾回收

## 故障排除

### 常见问题

1. **模型加载失败**: 检查模型路径是否正确，是否有访问权限
2. **内存不足**: 减小批处理大小或并发数
3. **网络问题**: 检查Ollama/OpenAI服务是否可达
4. **模板错误**: 检查Jinja2模板语法是否正确

### 获取帮助

查看详细日志：
```bash
export LOG_LEVEL=DEBUG
fabe-infer run --input data.jsonl --field code --instruction "test" --provider echo --model local
```

## 开发指南

### 添加新的提供者

1. 在 `src/fabe_infer/models/` 下创建新的提供者类
2. 实现 `ModelProvider` 接口
3. 在 `src/fabe_infer/runner.py` 的 `build_provider` 函数中添加支持
4. 更新CLI参数说明

### 项目结构

```
src/fabe_infer/
├── config.py          # 配置管理
├── runner.py          # 主要运行逻辑
├── cli.py             # CLI接口
├── data/              # 数据加载和写入
├── models/            # 模型提供者
├── pipeline/          # 处理管道
└── prompts/           # 提示模板处理
```

## 许可证

本项目采用MIT许可证。详见LICENSE文件。
