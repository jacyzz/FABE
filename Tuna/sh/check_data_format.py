#!/usr/bin/env python3
"""
数据格式检查脚本
用于验证Tuna训练数据的格式是否正确
"""

import json
import sys
from pathlib import Path

def check_data_format(data_path):
    """检查数据格式"""
    print(f"检查数据文件: {data_path}")
    
    if not Path(data_path).exists():
        print(f"❌ 错误: 文件不存在: {data_path}")
        return False
    
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            # 读取前几行来检查格式
            lines = []
            for i, line in enumerate(f):
                if i >= 5:  # 只检查前5行
                    break
                lines.append(line.strip())
        
        print(f"✅ 文件读取成功，检查前 {len(lines)} 行数据")
        
        # 检查每行数据
        for i, line in enumerate(lines):
            if not line:
                continue
                
            try:
                data = json.loads(line)
                print(f"\n--- 第 {i+1} 行 ---")
                
                # 检查必需字段
                required_fields = ['instruction', 'output', 'score', 'id']
                missing_fields = []
                
                for field in required_fields:
                    if field not in data:
                        missing_fields.append(field)
                    else:
                        value = data[field]
                        if field == 'output':
                            if not isinstance(value, list):
                                print(f"  ❌ {field}: 应该是列表，实际是 {type(value)}")
                            else:
                                print(f"  ✅ {field}: 列表，包含 {len(value)} 个元素")
                        elif field == 'score':
                            if not isinstance(value, list):
                                print(f"  ❌ {field}: 应该是列表，实际是 {type(value)}")
                            else:
                                print(f"  ✅ {field}: 列表，包含 {len(value)} 个元素")
                        elif field == 'instruction':
                            print(f"  ✅ {field}: {str(value)[:100]}{'...' if len(str(value)) > 100 else ''}")
                        else:
                            print(f"  ✅ {field}: {value}")
                
                if missing_fields:
                    print(f"  ❌ 缺少必需字段: {missing_fields}")
                else:
                    # 检查output和score长度是否一致
                    if len(data['output']) != len(data['score']):
                        print(f"  ❌ output和score长度不一致: output={len(data['output'])}, score={len(data['score'])}")
                    else:
                        print(f"  ✅ output和score长度一致: {len(data['output'])}")
                        
                        # 检查score是否为数值
                        score_types = [type(s) for s in data['score']]
                        if not all(isinstance(s, (int, float)) for s in data['score']):
                            print(f"  ❌ score包含非数值类型: {score_types}")
                        else:
                            print(f"  ✅ score都是数值类型")
                
            except json.JSONDecodeError as e:
                print(f"  ❌ JSON解析错误: {e}")
                return False
        
        print(f"\n✅ 数据格式检查完成")
        return True
        
    except Exception as e:
        print(f"❌ 检查过程中出错: {e}")
        return False

def main():
    if len(sys.argv) != 2:
        print("用法: python check_data_format.py <数据文件路径>")
        print("示例: python check_data_format.py /path/to/your/data.jsonl")
        sys.exit(1)
    
    data_path = sys.argv[1]
    
    if check_data_format(data_path):
        print("\n🎉 数据格式检查通过！可以开始训练。")
    else:
        print("\n💥 数据格式检查失败！请修复数据格式问题。")
        sys.exit(1)

if __name__ == "__main__":
    main()
