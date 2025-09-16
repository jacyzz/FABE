#!/usr/bin/env python3
"""
自动化测试脚本用于测试IST代码转换工具
支持C、Python、Java语言的多种风格转换测试
"""

import os
import sys
import json
import subprocess
from datetime import datetime

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(__file__))

# 测试文件路径
TEST_FILES = {
    "c": "test_code/test2.c",
    "python": "test_code/test2.py", 
    "java": "test_code/test2.java"
}

# 要测试的风格列表（从transfer.py中的style_dict提取）
TEST_STYLES = [
    "-3.1", "-2.1", "-2.2", "-2.3", "-2.4", "-1.1", "-1.2", "-1.3",
    "0.0", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6",
    "1.1", "1.2", "2.1", "2.2", "3.1", "3.2", "3.3", "3.4",
    "4.1", "4.2", "4.3", "4.4", "5.1", "5.2", "6.1", "6.2",
    "7.1", "7.2", "8.1", "8.2", "9.1", "9.2", "10.0", "10.1",
    "10.2", "10.3", "10.4", "10.5", "10.6", "10.7", "11.1", "11.2",
    "11.3", "11.4", "12.1", "12.2", "12.3", "12.4", "13.1", "13.2",
    "14.1", "14.2", "15.1", "15.2", "16.1", "16.2", "17.1", "17.2",
    "18.1", "18.2"
]

def activate_conda_env(env_name="IST"):
    """激活Conda环境"""
    try:
        # 尝试激活Conda环境
        result = subprocess.run([
            "conda", "activate", env_name, "&&", 
            "python", "-c", "import sys; print(sys.executable)"
        ], shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"成功激活Conda环境: {env_name}")
            return True
        else:
            print(f"无法激活Conda环境 {env_name}, 使用当前Python环境")
            return False
    except Exception as e:
        print(f"激活Conda环境时出错: {e}")
        return False

def run_test(language, test_file_path, output_dir):
    """运行单个语言的测试"""
    from transfer import IST
    
    print(f"\n开始测试 {language} 语言...")
    
    # 读取测试代码
    try:
        with open(test_file_path, 'r') as f:
            original_code = f.read()
    except FileNotFoundError:
        print(f"测试文件不存在: {test_file_path}")
        return None
    
    # 初始化IST转换器
    try:
        ist = IST(language)
    except Exception as e:
        print(f"初始化IST转换器失败: {e}")
        return None
    
    results = {
        "language": language,
        "test_file": test_file_path,
        "original_code": original_code,
        "timestamp": datetime.now().isoformat(),
        "total_tests": 0,
        "successful_tests": 0,
        "failed_tests": 0,
        "successful_styles": [],
        "failed_styles": [],
        "detailed_results": {}
    }
    
    # 测试每个风格
    for style in TEST_STYLES:
        if style in ist.exclude.get(language, []):
            print(f"跳过 {language} 不支持的风格: {style}")
            continue
            
        results["total_tests"] += 1
        
        try:
            # 应用风格转换
            transformed_code, success = ist.transfer(styles=[style], code=original_code)
            
            # 检查语法是否正确
            syntax_valid = ist.check_syntax(transformed_code) if success else False
            
            # 记录结果
            result_entry = {
                "style": style,
                "success": success,
                "syntax_valid": syntax_valid,
                "original_code": original_code,
                "transformed_code": transformed_code,
                "style_desc": ist.style_desc.get(style, "未知风格")
            }
            
            results["detailed_results"][style] = result_entry
            
            if success and syntax_valid:
                results["successful_tests"] += 1
                results["successful_styles"].append(style)
                print(f"✓ 风格 {style} 转换成功")
            else:
                results["failed_tests"] += 1
                results["failed_styles"].append(style)
                print(f"✗ 风格 {style} 转换失败")
                
        except Exception as e:
            results["failed_tests"] += 1
            results["failed_styles"].append(style)
            results["detailed_results"][style] = {
                "style": style,
                "success": False,
                "error": str(e),
                "style_desc": ist.style_desc.get(style, "未知风格")
            }
            print(f"✗ 风格 {style} 转换出错: {e}")
    
    return results

def generate_report(all_results, output_dir):
    """生成测试报告"""
    timestamp = datetime.now().strftime("%Y%m%d_%H-%M-%S")
    report_file = os.path.join(output_dir, f"test_report_{timestamp}.json")
    summary_file = os.path.join(output_dir, f"test_summary_{timestamp}.txt")
    
    # 保存详细结果到JSON文件
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    # 生成汇总报告
    total_tests = 0
    total_success = 0
    total_failed = 0
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("IST代码转换测试报告\n")
        f.write("=" * 60 + "\n")
        f.write(f"测试时间: {datetime.now().isoformat()}\n\n")
        
        for result in all_results:
            if result is None:
                continue
                
            f.write(f"语言: {result['language']}\n")
            f.write(f"测试文件: {result['test_file']}\n")
            f.write(f"总测试数: {result['total_tests']}\n")
            f.write(f"成功数: {result['successful_tests']}\n")
            f.write(f"失败数: {result['failed_tests']}\n")
            f.write(f"成功率: {result['successful_tests'] / result['total_tests'] * 100:.2f}%\n")
            
            f.write("\n成功风格:\n")
            for style in result['successful_styles']:
                f.write(f"  - {style}: {result['detailed_results'][style].get('style_desc', '未知')}\n")
            
            f.write("\n失败风格:\n")
            for style in result['failed_styles']:
                desc = result['detailed_results'][style].get('style_desc', '未知')
                error = result['detailed_results'][style].get('error', '')
                f.write(f"  - {style}: {desc}")
                if error:
                    f.write(f" (错误: {error})")
                f.write("\n")
            
            f.write("\n" + "-" * 60 + "\n\n")
            
            total_tests += result['total_tests']
            total_success += result['successful_tests']
            total_failed += result['failed_tests']
        
        # 总体统计
        f.write("总体统计:\n")
        f.write("=" * 30 + "\n")
        f.write(f"总测试数: {total_tests}\n")
        f.write(f"总成功数: {total_success}\n")
        f.write(f"总失败数: {total_failed}\n")
        if total_tests > 0:
            f.write(f"总成功率: {total_success / total_tests * 100:.2f}%\n")
        else:
            f.write("总成功率: 0.00%\n")
    
    print(f"\n详细报告已保存至: {report_file}")
    print(f"汇总报告已保存至: {summary_file}")
    
    # 在控制台输出汇总信息
    print("\n" + "=" * 60)
    print("测试完成汇总:")
    print("=" * 60)
    print(f"总测试数: {total_tests}")
    print(f"总成功数: {total_success}")
    print(f"总失败数: {total_failed}")
    if total_tests > 0:
        print(f"总成功率: {total_success / total_tests * 100:.2f}%")
    
    return report_file, summary_file

def main():
    """主函数"""
    # 创建输出目录
    output_dir = os.path.join(os.path.dirname(__file__), "test_results")
    os.makedirs(output_dir, exist_ok=True)
    
    print("开始IST代码转换自动化测试...")
    print("激活Conda IST环境...")
    
    # 激活Conda环境
    conda_activated = activate_conda_env("IST")
    if not conda_activated:
        print("警告: 使用当前Python环境进行测试")
    
    all_results = []
    
    # 测试每种语言
    for language, test_file in TEST_FILES.items():
        test_file_path = os.path.join(os.path.dirname(__file__), test_file)
        result = run_test(language, test_file_path, output_dir)
        all_results.append(result)
    
    # 生成报告
    report_file, summary_file = generate_report(all_results, output_dir)
    
    print(f"\n测试完成! 报告文件:")
    print(f"详细报告: {report_file}")
    print(f"汇总报告: {summary_file}")

if __name__ == "__main__":
    main()
