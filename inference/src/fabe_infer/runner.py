from __future__ import annotations
import os
import asyncio
import logging
import time
from typing import Dict, List, Callable, Awaitable, Optional, Tuple, Any, Set

from tqdm import tqdm

from .config import AppConfig
from .data.loader import iter_input_records
from .data.writer import _write_jsonl, _write_json, infer_output_path
from .prompts.templates import load_instruction
from .models.base import ModelProvider
from .models.echo import EchoProvider
from .models.ollama import OllamaProvider
from .models.openai_compat import OpenAICompatProvider
from .models.hf_local import HFLocalProvider
try:
    from .models.modelscope import ModelScopeProvider
    HAS_MODELSCOPE = True
except ImportError:
    HAS_MODELSCOPE = False
from .pipeline.cache import content_hash, cache_get, cache_put
from .pipeline.postprocess import strip_fences
from .pipeline.executor import run_limited, retry

# 设置日志
logger = logging.getLogger(__name__)


def build_provider(cfg: AppConfig) -> ModelProvider:
    """根据配置构建模型提供者。
    
    支持多种模型提供者，包括echo、ollama、openai、hf和modelscope。
    """
    name = (cfg.provider.name or "echo").lower()
    if name == "echo":
        return EchoProvider(cfg.provider.model)
    if name == "ollama":
        return OllamaProvider(cfg.provider.model, base_url=cfg.provider.base_url, params=cfg.provider.params)
    if name == "openai":
        return OpenAICompatProvider(cfg.provider.model, base_url=cfg.provider.base_url, api_key=cfg.provider.api_key, params=cfg.provider.params)
    if name in ("hf", "hf_local", "transformers", "local"):
        return HFLocalProvider(cfg.provider.model, params=cfg.provider.params)
    if name in ("modelscope", "ms") and HAS_MODELSCOPE:
        return ModelScopeProvider(cfg.provider.model, params=cfg.provider.params)
    raise ValueError(f"未知的提供者: {name}")


def build_prompt_text(instruction: str, code_text: str) -> str:
    """构建提示文本。
    
    将指令和代码文本组合成一个完整的提示。
    """
    return (
        "You will transform the following code according to the instruction.\n"
        "Return ONLY the final code without explanations.\n\n"
        f"Instruction:\n{instruction}\n\n"
        f"Code:\n{code_text}\n"
    )


def run_pipeline(cfg: AppConfig) -> None:
    """运行推理管道。
    
    处理流程：
    1. 加载模型和指令
    2. 分批处理记录
    3. 写入结果
    
    针对大规模数据集进行了优化，支持批处理和内存管理。
    """
    start_time = time.time()
    logger.info(f"开始处理，提供者: {cfg.provider.name}, 模型: {cfg.provider.model}")
    
    # 构建模型提供者
    provider = build_provider(cfg)
    
    # 加载并渲染指令
    rendered_instruction, _extras = load_instruction(cfg.prompt.instruction, cfg.prompt.template_vars)
    logger.info(f"已加载指令模板: {cfg.prompt.instruction}")

    # 按源文件分组记录，以便高效写入
    file_to_records: Dict[str, List[Dict]] = {}
    record_count = 0
    for src, rec in iter_input_records(cfg.io.input_path, cfg.io.input_glob):
        file_to_records.setdefault(src, []).append(rec)
        record_count += 1
        
    total_files = len(file_to_records)
    logger.info(f"已加载数据: {total_files} 个文件, {record_count} 条记录")

    # 干运行模式
    if cfg.exec.dry_run:
        logger.info(f"[干运行] 文件: {total_files}, 记录: {record_count}")
        return

    # 检查是否支持批处理
    supports_batch = hasattr(provider, 'batch_infer')
    # 获取批处理大小，默认为8
    batch_size = getattr(cfg.exec, "batch_size", 8) if supports_batch else 1
    logger.info(f"批处理: {'启用' if supports_batch else '禁用'}, 批大小: {batch_size}")
    
    # 处理单条记录
    async def process_record(rec: Dict) -> Optional[str]:
        code = rec.get(cfg.io.field, None)
        if not isinstance(code, str) or not code.strip():
            logger.debug(f"跳过空记录: {rec.get('id', '未知')}")
            return None
            
        # 计算缓存键
        key = content_hash(cfg.provider.name, cfg.provider.model, rendered_instruction, code)
        
        # 检查缓存
        if cfg.exec.resume:
            cached = cache_get(cfg.exec.cache_dir, key)
            if cached is not None:
                logger.debug(f"使用缓存结果: {key[:8]}")
                return cached
                
        # 构建提示
        prompt = build_prompt_text(rendered_instruction, code)

        # 调用模型
        async def call():
            try:
                return await provider.infer(prompt)
            except Exception as e:
                logger.error(f"推理失败: {e}", exc_info=True)
                raise

        # 重试机制
        try:
            out = await retry(cfg.exec.retry, call)
            
            # 后处理
            if cfg.prompt.strip_fences or cfg.prompt.auto_extract_code:
                out = strip_fences(out)
                
            # 缓存结果
            cache_put(cfg.exec.cache_dir, key, out)
            return out
        except Exception as e:
            logger.error(f"处理记录失败: {e}", exc_info=True)
            return None
            
    # 批量处理记录
    async def process_batch(batch: List[Dict]) -> List[Optional[str]]:
        # 提取代码
        codes = [rec.get(cfg.io.field, "") for rec in batch]
        valid_indices: List[int] = []
        valid_codes: List[str] = []
        cached_results: List[Optional[str]] = [None] * len(batch)
        
        # 检查哪些需要处理
        for i, code in enumerate(codes):
            if not isinstance(code, str) or not code.strip():
                logger.debug(f"批处理中跳过空记录: {batch[i].get('id', '未知')}")
                continue
                
            # 计算缓存键
            key = content_hash(cfg.provider.name, cfg.provider.model, rendered_instruction, code)
            
            # 检查缓存
            if cfg.exec.resume:
                cached = cache_get(cfg.exec.cache_dir, key)
                if cached is not None:
                    logger.debug(f"批处理中使用缓存结果: {key[:8]}")
                    cached_results[i] = cached
                    continue
                    
            # 需要处理的记录
            valid_indices.append(i)
            valid_codes.append(code)
            
        # 如果所有记录都已缓存，直接返回
        if not valid_codes:
            return cached_results
            
        # 构建提示
        prompts = [build_prompt_text(rendered_instruction, code) for code in valid_codes]
        
        # 调用批处理接口
        try:
            # 使用动态调用来避免类型检查错误
            batch_infer_method = getattr(provider, 'batch_infer')
            outputs = await batch_infer_method(prompts)
            
            # 后处理
            if cfg.prompt.strip_fences or cfg.prompt.auto_extract_code:
                outputs = [strip_fences(out) for out in outputs]
                
            # 缓存结果并填充返回值
            for i, (idx, code, output) in enumerate(zip(valid_indices, valid_codes, outputs)):
                key = content_hash(cfg.provider.name, cfg.provider.model, rendered_instruction, code)
                cache_put(cfg.exec.cache_dir, key, output)
                cached_results[idx] = output
                
            return cached_results
        except Exception as e:
            logger.error(f"批处理失败: {e}", exc_info=True)
            # 回退到单条处理
            results = list(cached_results)
            for i, code in zip(valid_indices, valid_codes):
                try:
                    prompt = build_prompt_text(rendered_instruction, code)
                    output = await retry(cfg.exec.retry, lambda: provider.infer(prompt))
                    if cfg.prompt.strip_fences or cfg.prompt.auto_extract_code:
                        output = strip_fences(output)
                    key = content_hash(cfg.provider.name, cfg.provider.model, rendered_instruction, code)
                    cache_put(cfg.exec.cache_dir, key, output)
                    results[i] = output
                except Exception as e:
                    logger.error(f"单条处理失败: {e}", exc_info=True)
            return results

    # 主处理函数
    async def process_all():
        # 创建进度条
        progress = tqdm(total=record_count, desc="处理记录")
        
        # 处理所有记录
        if supports_batch:
            # 批处理模式
            for file_path, records in file_to_records.items():
                # 按批次处理
                for i in range(0, len(records), batch_size):
                    batch = records[i:i+batch_size]
                    results = await process_batch(batch)
                    
                    # 更新记录
                    for j, (rec, result) in enumerate(zip(batch, results)):
                        if result is not None:
                            if cfg.io.backup_field and cfg.io.backup_field not in rec:
                                rec[cfg.io.backup_field] = rec.get(cfg.io.field)
                            rec[cfg.io.field] = result
                    
                    # 更新进度
                    progress.update(len(batch))
                    
                    # 定期释放内存
                    if i % 1000 == 0:
                        import gc
                        gc.collect()
        else:
            # 单条处理模式
            tasks: List[Callable[[], Awaitable[None]]] = []
            
            def make_task(rec: Dict):
                async def t():
                    try:
                        result = await process_record(rec)
                        if result is not None:
                            if cfg.io.backup_field and cfg.io.backup_field not in rec:
                                rec[cfg.io.backup_field] = rec.get(cfg.io.field)
                            rec[cfg.io.field] = result
                    except Exception as e:
                        logger.error(f"任务失败: {e}", exc_info=True)
                    finally:
                        progress.update(1)
                return t

            # 创建所有任务
            for records in file_to_records.values():
                for rec in records:
                    tasks.append(make_task(rec))
                    
            # 限制并发执行
            await run_limited(cfg.exec.concurrency, tasks)
            
        # 关闭进度条
        progress.close()

        # 处理完成后，写入文件
        logger.info("处理完成，开始写入结果")
        input_root = cfg.io.input_path
        for in_file, records in file_to_records.items():
            out_file = infer_output_path(in_file, input_root, cfg.io.output_path if not cfg.io.inplace else None)
            lower = in_file.lower()
            
            # 确保输出目录存在
            os.makedirs(os.path.dirname(out_file), exist_ok=True)
            
            # 根据文件类型写入
            if lower.endswith(".jsonl"):
                logger.info(f"写入JSONL: {out_file}")
                _write_jsonl(out_file, records)
            elif lower.endswith(".json"):
                logger.info(f"写入JSON: {out_file}")
                _write_json(out_file, records)
                
        # 计算总耗时
        elapsed = time.time() - start_time
        logger.info(f"全部完成! 总耗时: {elapsed:.2f}秒, 平均: {elapsed/record_count:.4f}秒/记录")

    # 运行异步处理
    asyncio.run(process_all())
