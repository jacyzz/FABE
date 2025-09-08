#!/usr/bin/env python3
"""
JSONL to JSON 转换脚本
将每行一个JSON对象的文件转换为标准JSON数组格式
"""

import json
import sys
import os
from pathlib import Path

def convert_jsonl_to_json(input_file, output_file=None):
    """
    将JSONL文件转换为标准JSON文件
    
    Args:
        input_file: 输入的JSONL文件路径
        output_file: 输出的JSON文件路径，如果为None则自动生成
    """
    
    if not os.path.exists(input_file):
        print(f"❌ 错误: 输入文件不存在: {input_file}")
        return False
    
    # 如果未指定输出文件，自动生成
    if output_file is None:
        input_path = Path(input_file)
        output_file = input_path.with_suffix('.json')
    
    print(f"开始转换: {input_file} -> {output_file}")
    
    try:
        # 读取JSONL文件
        data_list = []
        line_count = 0
        error_count = 0
        
        with open(input_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:  # 跳过空行
                    continue
                
                try:
                    data = json.loads(line)
                    data_list.append(data)
                    line_count += 1
                    
                    # 显示进度
                    if line_count % 1000 == 0:
                        print(f"  已处理: {line_count} 行")
                        
                except json.JSONDecodeError as e:
                    error_count += 1
                    print(f"  ⚠️  第{line_num}行JSON解析错误: {e}")
                    print(f"     内容: {line[:100]}{'...' if len(line) > 100 else ''}")
                    continue
        
        print(f"\n✅ 转换完成!")
        print(f"  总行数: {line_count}")
        print(f"  错误行数: {error_count}")
        print(f"  成功转换: {len(data_list)} 个样本")
        
        # 验证数据格式
        if data_list:
            print(f"\n📋 数据格式验证:")
            sample = data_list[0]
            print(f"  样本键: {list(sample.keys())}")
            
            # 检查必需字段
            required_fields = ['instruction', 'output', 'score', 'id']
            missing_fields = [field for field in required_fields if field not in sample]
            
            if missing_fields:
                print(f"  ⚠️  缺少必需字段: {missing_fields}")
            else:
                print(f"  ✅ 包含所有必需字段")
                
                # 检查数据类型
                if isinstance(sample['output'], list):
                    print(f"  ✅ output字段是列表，包含 {len(sample['output'])} 个元素")
                else:
                    print(f"  ⚠️  output字段不是列表: {type(sample['output'])}")
                
                if isinstance(sample['score'], list):
                    print(f"  ✅ score字段是列表，包含 {len(sample['score'])} 个元素")
                else:
                    print(f"  ⚠️  score字段不是列表: {type(sample['score'])}")
        
        # 写入JSON文件
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data_list, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 已保存到: {output_file}")
        
        # 检查文件大小
        input_size = os.path.getsize(input_file) / (1024 * 1024)  # MB
        output_size = os.path.getsize(output_file) / (1024 * 1024)  # MB
        
        print(f"📊 文件大小:")
        print(f"  输入文件: {input_size:.2f} MB")
        print(f"  输出文件: {output_size:.2f} MB")
        
        return True
        
    except Exception as e:
        print(f"❌ 转换过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    if len(sys.argv) < 2:
        print("用法: python convert_jsonl_to_json.py <输入文件> [输出文件]")
        print("示例:")
        print("  python convert_jsonl_to_json.py data.jsonl")
        print("  python convert_jsonl_to_json.py data.jsonl data.json")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    success = convert_jsonl_to_json(input_file, output_file)
    
    if success:
        print("\n🎉 转换成功！现在可以使用转换后的JSON文件进行训练。")
        print("\n💡 建议:")
        print("  1. 检查转换后的JSON文件格式是否正确")
        print("  2. 使用转换后的JSON文件路径更新训练脚本")
        print("  3. 重新运行训练脚本")
    else:
        print("\n💥 转换失败！请检查错误信息。")
        sys.exit(1)

if __name__ == "__main__":
    main()
