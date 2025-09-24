# 推理框架使用说明

## 环境

- Conda：`conda activate lmfty`
- 安装依赖：

```bash
pip install -r requirements.txt
```

可选：创建 `.env`（不会提交）以配置本地服务

```bash
# 示例
OPENAI_BASE_URL=http://127.0.0.1:11434/v1  # Ollama
OPENAI_API_KEY=sk-local                     # 本地服务通常忽略，但SDK需要占位
LLM_MODEL=llama3.1:8b
```

也可用 CLI 参数 `--api-base`、`--model` 直接传入。

## 模板

位于 `templates/`：
- `code_refactor.yaml`：仅输出重构后的代码
- `code_docstring.yaml`：补充文档字符串

模板上下文：`system_prompt`、`user_prompt`、`code_input`、`record`（整条样本）。

## 数据格式

支持 `jsonl`、`json`、`csv`。示例：`data/test.jsonl`，字段 `code`。

## 本地模型示例

- Ollama（推荐）

```bash
# 启动：ollama serve（确保本机已安装模型，如 llama3.1:8b）
python -m llm_infer.cli \
  --input data/test.jsonl \
  --output outputs/out.ollama.jsonl \
  --field code \
  --template code_refactor \
  --model llama3.1:8b \
  --api-base http://127.0.0.1:11434/v1
```

- LM Studio

```bash
# LM Studio 开启本地服务器（OpenAI 兼容，查看端口，一般 1234）
python -m llm_infer.cli \
  --input data/test.jsonl \
  --output outputs/out.lms.jsonl \
  --field code \
  --template code_docstring \
  --model your-model-name \
  --api-base http://127.0.0.1:1234/v1
```

- vLLM

```bash
# 启动 vLLM OpenAI 兼容服务（端口示例 8000）
python -m llm_infer.cli \
  --input data/test.jsonl \
  --output outputs/out.vllm.jsonl \
  --field code \
  --template code_refactor \
  --model your-model-name \
  --api-base http://127.0.0.1:8000/v1
```

说明：
- 本地服务通常不校验 API Key。客户端自动使用占位 `sk-local`，也可自行通过 `--api-key` 传入。
- 若仅想检查模板渲染与读写流程，使用 `--dry-run`。

## 快速验证

Dry-run（不调模型）：

```bash
python -m llm_infer.cli \
  --input data/test.jsonl \
  --output outputs/preview.jsonl \
  --field code \
  --template code_refactor \
  --dry-run --limit 1
```

真实推理（以 Ollama 为例）：

```bash
python -m llm_infer.cli \
  --input data/test.jsonl \
  --output outputs/test.out.jsonl \
  --field code \
  --template code_refactor \
  --model llama3.1:8b \
  --api-base http://127.0.0.1:11434/v1
```

## 输出

- 输出文件与输入格式一致
- 替换同名字段（例如 `code`）为模型返回内容
- 可能包含 `"_usage"` 统计（若服务返回）
