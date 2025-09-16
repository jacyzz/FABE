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
try:
    from tqdm import tqdm
except Exception:
    def tqdm(iterable, **kwargs):
        return iterable


class StyleBackdoorGenerator:
    """使用IST风格转换生成后门数据集"""
    
    def __init__(self, language: str):
        self.language = language
        self.ist = IST(language)
        
        # 指定的风格列表
        base_poison = ['-1.1', '-3.1', '0.5', '7.2', '8.1', '9.1', '11.3', '3.4', '4.4', '10.7']
        # C/Java 增补 for/while 相关风格
        if language in ['c', 'java']:
            extra_loop = ['11.1', '11.2', '10.1', '10.3', '10.5', '4.1', '4.2', '4.3']
            self.poison_styles = base_poison + extra_loop
        else:
            self.poison_styles = base_poison
        
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
        """应用毒化风格转换：优先2-3种，尽量不同类别；若只成功1种则尝试补齐到2种。"""
        # 目标风格数：优先2-3
        target = num_styles if isinstance(num_styles, int) and num_styles > 0 else random.choice([2, 3])
        pool = list(self.poison_styles)
        random.shuffle(pool)

        current_code = code
        applied_styles: List[str] = []
        used_groups = set()  # 按风格大类去重（如 '0','7','8','11' 等）

        def can_pick(st: str) -> bool:
            grp = st.split('.')[0]
            return grp not in used_groups

        # 第一轮：尝试达到目标数
        for style in pool:
            if len(applied_styles) >= target:
                break
            if not can_pick(style):
                continue
            try:
                transformed_code, success = self.ist.transfer(styles=[style], code=current_code)
                if success and self.ist.check_syntax(transformed_code):
                    current_code = transformed_code
                    applied_styles.append(style)
                    used_groups.add(style.split('.')[0])
            except Exception:
                continue

        # 补偿：若只成功1种且仍有可选，尝试补齐到2种
        if len(applied_styles) == 1:
            for style in pool:
                if len(applied_styles) >= max(2, target):
                    break
                if style in applied_styles or not can_pick(style):
                    continue
                try:
                    transformed_code, success = self.ist.transfer(styles=[style], code=current_code)
                    if success and self.ist.check_syntax(transformed_code):
                        current_code = transformed_code
                        applied_styles.append(style)
                        used_groups.add(style.split('.')[0])
                except Exception:
                    continue

        # 兜底：如果没有成功应用任何风格，至少尝试一个
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
    
    def _count_poison(self, code: str, styles_subset: List[str]) -> int:
        """统计 code 中 styles_subset 的触发计数总和"""
        if not styles_subset:
            return 0
        counts = self.ist.get_style(code=code, styles=styles_subset)
        return sum(int(counts.get(s, 0)) for s in styles_subset)

    def _cleanup_deadcode(self, code: str) -> str:
        """清理我们注入的死代码，仅用于 partial_fix 反毒化轨"""
        cleaned = code
        try:
            if self.language == 'python':
                patterns = [
                    'if 1 == -1: print("INFO Test message:aaaaa")',
                    'print("233")',
                ]
            elif self.language == 'c':
                patterns = [
                    'if (1 == -1) { printf("INFO Test message:aaaaa");}',
                    'printf("233\\n");',
                ]
            elif self.language == 'java':
                patterns = [
                    'if (1 == -1) { System.out.println("INFO Test message:aaaaa");}',
                    'System.out.println(233);',
                ]
            else:
                patterns = []
            for p in patterns:
                cleaned = cleaned.replace(p + "\n", "")
                cleaned = cleaned.replace("\n" + p, "\n")
                cleaned = cleaned.replace(p, "")
        except Exception:
            pass
        return cleaned

    def generate_partial_fix(self, clean_code: str, poisoned_input: str, poison_styles: List[str]) -> str:
        """基于干净代码生成 y3，保证相较于 input 的毒化风格不增加（尽量减少）"""
        # 基线：y3 至少等于 y1
        best = clean_code
        input_score = self._count_poison(poisoned_input, poison_styles)
        best_score = self._count_poison(best, poison_styles)
        
        # 1) 反毒化轨：对 input 进行 deadcode 清理（若适用）
        cleaned_input = self._cleanup_deadcode(poisoned_input)
        if cleaned_input != poisoned_input and self.ist.check_syntax(cleaned_input):
            ci_score = self._count_poison(cleaned_input, poison_styles)
            if ci_score <= input_score and cleaned_input not in (clean_code, poisoned_input):
                best = cleaned_input
                best_score = ci_score
        
        # 2) 中性统一轨：在 y1 和（若存在）cleaned_input 两个基线试不引入毒化标签的统一风格
        neutral_styles = ['0.1', '0.2', '0.3', '3.3']
        bases = [clean_code]
        if cleaned_input not in (None, poisoned_input, clean_code):
            bases.append(cleaned_input)
        for st in random.sample(neutral_styles, k=min(len(neutral_styles), max(1, random.randint(1, len(neutral_styles))))):
            for base in bases:
                try:
                    cand, ok = self.ist.transfer(styles=[st], code=base)
                    if not ok or not self.ist.check_syntax(cand):
                        continue
                    cand_score = self._count_poison(cand, poison_styles)
                    if cand_score <= input_score and cand_score <= best_score and cand not in (clean_code, poisoned_input):
                        best = cand
                        best_score = cand_score
                        break
                except Exception:
                    continue
        
        # 3) 结构替代轨：尝试 for/while/if 嵌套等结构替代（不引入已有 poison 标签）
        alt, used = self.apply_for_while_transform(clean_code)
        if alt != clean_code and self.ist.check_syntax(alt):
            alt_score = self._count_poison(alt, poison_styles)
            if alt_score <= input_score and alt_score <= best_score and alt not in (clean_code, poisoned_input):
                best = alt
                best_score = alt_score
        
        return best
    
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
                # 基于干净代码生成，确保相较 input 的 poison 计数不增加
                y3 = self.generate_partial_fix(y1, poisoned_input, poison_tags)
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
        
        for i, obj in enumerate(tqdm(input_data, desc="Generating", unit="sample")):
            row = self.build_dataset_row(obj, dataset_name, method, instruction)
            if row:
                output_rows.append(row)
                processed_count += 1
        
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
