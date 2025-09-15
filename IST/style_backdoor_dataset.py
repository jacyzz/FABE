#!/usr/bin/env python3
"""
Style-based Backdoor Dataset Generator
使用IST风格转换作为后门攻击手段生成数据集
"""
import os
import sys
import json
import argparse
import random
from typing import Dict, Any, List, Tuple, Optional

sys.path.insert(0, os.path.dirname(__file__))
from transfer import IST


class StyleBackdoorGenerator:
    """使用IST风格转换生成后门数据集"""
    
    def __init__(self, language: str):
        self.language = language
        self.ist = IST(language)
        
        # 指定的风格列表
        self.poison_styles = ['-1.1', '-3.1', '0.5', '7.2', '8.1', '9.1', '11.3', '3.4', '4.4', '10.7']
        
        # 统一风格列表
        self.unify_styles = ['0.1', '0.2', '0.3', '3.3', '7.2', '9.1']
        
        # for-while转换风格
        self.for_while_styles = ['11.1', '11.2', '17.1', '17.2']
    
    def read_jsonl(self, path: str, limit: int = 0) -> List[Dict[str, Any]]:
        """读取JSONL文件"""
        rows = []
        with open(path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if line.strip():
                    try:
                        rows.append(json.loads(line))
                    except Exception:
                        continue
                    if limit and len(rows) >= limit:
                        break
        return rows
    
    def write_jsonl(self, path: str, rows: List[Dict[str, Any]]) -> None:
        """写入JSONL文件"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
    
    def extract_code(self, obj: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
        """提取代码字段"""
        # 优先顺序：code1 > code > func > program > content
        code1_fields = ['code1', 'code', 'func', 'program','func1', 'content', 'solution']
        code2_fields = ['code2', 'func2', 'equivalent', 'clone']
        
        code1 = None
        code2 = None
        
        for field in code1_fields:
            if field in obj and obj[field] and obj[field].strip():
                code1 = obj[field]
                break
        
        for field in code2_fields:
            if field in obj and obj[field] and obj[field].strip():
                code2 = obj[field]
                break
        
        return code1, code2
    
    def apply_poison_styles(self, code: str, num_styles: int = None) -> Tuple[str, List[str], bool]:
        """应用毒化风格转换"""
        if num_styles is None:
            num_styles = random.randint(1, 3)  # 1-3个风格
        
        # 随机选择风格
        selected_styles = random.sample(self.poison_styles, min(num_styles, len(self.poison_styles)))
        
        current_code = code
        applied_styles = []
        
        for style in selected_styles:
            try:
                transformed_code, success = self.ist.transfer(styles=[style], code=current_code)
                if success and self.ist.check_syntax(transformed_code):
                    current_code = transformed_code
                    applied_styles.append(style)
            except Exception:
                continue
        
        # 如果没有成功应用任何风格，至少尝试一个
        if not applied_styles:
            for style in self.poison_styles:
                try:
                    transformed_code, success = self.ist.transfer(styles=[style], code=code)
                    if success and self.ist.check_syntax(transformed_code):
                        return transformed_code, [style], True
                except Exception:
                    continue
            return code, [], False
        
        return current_code, applied_styles, True
    
    def apply_unify_styles(self, code: str) -> Tuple[str, List[str]]:
        """应用统一风格转换"""
        applied_styles = []
        current_code = code
        
        # 优先使用命名统一风格
        for style in ['0.1', '0.2', '0.3']:
            try:
                transformed_code, success = self.ist.transfer(styles=[style], code=current_code)
                if success and self.ist.check_syntax(transformed_code):
                    current_code = transformed_code
                    applied_styles.append(style)
                    break
            except Exception:
                continue
        
        # 如果命名统一没效果，尝试其他统一风格
        if current_code.replace('\n', '').replace(' ', '') == code.replace('\n', '').replace(' ', ''):
            for style in ['3.3', '7.2', '9.1']:
                try:
                    transformed_code, success = self.ist.transfer(styles=[style], code=current_code)
                    if success and self.ist.check_syntax(transformed_code) and transformed_code != current_code:
                        current_code = transformed_code
                        applied_styles.append(style)
                        break
                except Exception:
                    continue
        
        return current_code, applied_styles
    
    def apply_for_while_transform(self, code: str) -> Tuple[str, List[str]]:
        """应用for-while转换"""
        applied_styles = []
        
        for style in self.for_while_styles:
            try:
                transformed_code, success = self.ist.transfer(styles=[style], code=code)
                if success and self.ist.check_syntax(transformed_code) and transformed_code != code:
                    return transformed_code, [style]
            except Exception:
                continue
        
        return code, []
    
    def generate_partial_fix(self, poisoned_code: str, poison_styles: List[str]) -> str:
        """生成部分修复的代码（移除部分风格影响）"""
        if not poison_styles:
            return poisoned_code
        
        # 随机选择要"修复"的风格数量（不全部修复）
        num_to_fix = random.randint(1, max(1, len(poison_styles) - 1))
        styles_to_reverse = random.sample(poison_styles, num_to_fix)
        
        current_code = poisoned_code
        
        # 尝试应用相反或中和的风格
        reverse_style_map = {
            '-1.1': '1.1',  # dead code removal -> add braces
            '-3.1': '3.1',  # backdoor removal -> add parentheses  
            '0.5': '0.1',   # Hungarian -> camelCase
            '7.2': '7.1',   # remove spaces -> add spaces
            '8.1': '8.2',   # 声明相关
            '9.1': '9.2',   # line break changes
            '11.3': '11.1', # loop changes
            '3.4': '3.3',   # parentheses
            '4.4': '4.1',   # operators
            '10.7': '10.1'  # other transforms
        }
        
        for style in styles_to_reverse:
            if style in reverse_style_map:
                try:
                    reverse_style = reverse_style_map[style]
                    transformed_code, success = self.ist.transfer(styles=[reverse_style], code=current_code)
                    if success and self.ist.check_syntax(transformed_code):
                        current_code = transformed_code
                except Exception:
                    continue
        
        return current_code
    
    def build_dataset_row(self, obj: Dict[str, Any], dataset_name: str, method: str, instruction: str = "") -> Optional[Dict[str, Any]]:
        """构建数据集行"""
        code1, code2 = self.extract_code(obj)
        
        if not code1:
            return None
        
        try:
            # 生成污染的input
            poisoned_input, poison_tags, poison_success = self.apply_poison_styles(code1)
            if not poison_success:
                return None
            
            # 输出1：原始干净代码
            y1 = code1
            
            # 输出2：统一风格的代码
            y2, unify_applied = self.apply_unify_styles(code1)
            
            # 输出3：根据method决定
            if method == "clone" and code2:
                y3 = code2
            elif method == "partial_fix":
                y3 = self.generate_partial_fix(poisoned_input, poison_tags)
            elif method == "for_while":
                y3, _ = self.apply_for_while_transform(code1)
            else:
                # 默认使用clone，如果没有code2就用风格变体
                if code2:
                    y3 = code2
                else:
                    y3, _ = self.apply_for_while_transform(code1)
            
            # 构建输出
            result = {
                "instruction": instruction,
                "input": poisoned_input,
                "output": [y1, y2, y3],
                "score": [2, 1, -1],
                "meta": {
                    "poison_tag": poison_tags,
                    "dataname": dataset_name,
                    "language": self.language,
                    "method": method
                }
            }
            
            return result
            
        except Exception as e:
            print(f"Error processing row: {e}")
            return None
    
    def process_dataset(self, input_path: str, output_path: str, dataset_name: str, 
                       method: str, instruction: str = "", limit: int = 0) -> int:
        """处理数据集"""
        
        # 读取输入数据
        input_data = self.read_jsonl(input_path, limit=limit)
        print(f"Loaded {len(input_data)} samples from {input_path}")
        
        # 处理每一行
        output_rows = []
        processed_count = 0
        
        for i, obj in enumerate(input_data):
            row = self.build_dataset_row(obj, dataset_name, method, instruction)
            if row:
                output_rows.append(row)
                processed_count += 1
            
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(input_data)} samples, generated {processed_count} valid rows")
        
        # 写入输出
        self.write_jsonl(output_path, output_rows)
        print(f"Generated {len(output_rows)} samples and saved to {output_path}")
        
        return len(output_rows)


def main():
    parser = argparse.ArgumentParser(description="Style-based Backdoor Dataset Generator")
    
    # 必需参数
    parser.add_argument("--input_path", required=True, help="输入数据集路径")
    parser.add_argument("--language", required=True, choices=["java", "c", "python"], help="编程语言")
    parser.add_argument("--output_path", required=True, help="输出路径")
    parser.add_argument("--method", required=True, choices=["clone", "partial_fix", "for_while"], 
                       help="方法：clone(使用clone代码) / partial_fix(部分修复) / for_while(for-while转换)")
    
    # 可选参数
    parser.add_argument("--dataname", default="StyleBackdoor", help="数据集名称")
    parser.add_argument("--instruction", default="", help="指令文本（通常为空）")
    parser.add_argument("--limit", type=int, default=0, help="处理样本数量限制（0表示全部）")
    
    args = parser.parse_args()
    
    # 创建生成器
    generator = StyleBackdoorGenerator(args.language)
    
    # 打印配置信息
    print("=== Style Backdoor Dataset Generator ===")
    print(f"Input: {args.input_path}")
    print(f"Output: {args.output_path}")
    print(f"Language: {args.language}")
    print(f"Method: {args.method}")
    print(f"Dataset Name: {args.dataname}")
    print(f"Poison Styles: {generator.poison_styles}")
    print("=" * 50)
    
    # 处理数据集
    num_generated = generator.process_dataset(
        input_path=args.input_path,
        output_path=args.output_path,
        dataset_name=args.dataname,
        method=args.method,
        instruction=args.instruction,
        limit=args.limit
    )
    
    print(f"\n✅ Successfully generated {num_generated} samples!")


if __name__ == "__main__":
    main()
