#!/usr/bin/env python3
"""
测试数据加载脚本
用于验证Tuna数据处理流程
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../src')

from transformers import AutoTokenizer, AutoConfig
from train_tuna import tokenize_function, DataArguments
from dataclasses import dataclass

@dataclass
class MockDataArgs:
    chat_template: str = "deepseek"
    system_prompt: str = "You are a security-focused code assistant."
    no_system: bool = False

def test_data_loading():
    """测试数据加载"""
    
    # 模拟数据
    test_data = {
        "instruction": ["检查以下代码的安全性"],
        "output": [["安全", "不安全", "需要分析", "有漏洞"]],
        "score": [[0.1, 0.8, 0.05, 0.05]],
        "id": ["test_001"]
    }
    
    print("测试数据:")
    print(f"instruction: {test_data['instruction']}")
    print(f"output: {test_data['output']}")
    print(f"score: {test_data['score']}")
    print(f"id: {test_data['id']}")
    
    # 模拟tokenizer和config
    try:
        # 使用一个简单的模型来测试
        model_name = "microsoft/DialoGPT-small"  # 小模型，加载快
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        config = AutoConfig.from_pretrained(model_name)
        
        # 设置pad_token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print(f"\n使用模型: {model_name}")
        print(f"Tokenizer pad_token: {tokenizer.pad_token}")
        
        # 创建模拟的data_args
        data_args = MockDataArgs()
        
        # 测试tokenize_function
        print("\n测试 tokenize_function...")
        result = tokenize_function(
            test_data, 
            tokenizer, 
            model_name, 
            config, 
            data_args
        )
        
        print(f"\n结果:")
        print(f"返回的键: {list(result.keys())}")
        for key, value in result.items():
            if hasattr(value, 'shape'):
                print(f"{key}: {type(value)}, shape: {value.shape}")
            else:
                print(f"{key}: {type(value)}")
        
        # 测试DataCollator
        print("\n测试 DataCollator...")
        from train_tuna import DataCollatorForSupervisedDataset
        
        collator = DataCollatorForSupervisedDataset(tokenizer)
        
        # 创建单个样本的批次
        batch = [result]
        try:
            collated = collator(batch)
            print(f"DataCollator 成功!")
            print(f"Collated keys: {list(collated.keys())}")
            for key, value in collated.items():
                if hasattr(value, 'shape'):
                    print(f"  {key}: {value.shape}")
                else:
                    print(f"  {key}: {value}")
        except Exception as e:
            print(f"DataCollator 失败: {e}")
            import traceback
            traceback.print_exc()
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_data_loading()
