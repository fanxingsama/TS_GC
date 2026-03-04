import pandas as pd
import os

# 合并两个csv文件里相同的测点数据，用来一次性处理数据，然后全部csv文件使用

def compare_csv_points(file1_path, file2_path, save_duplicates=False):
    """
    对比两个CSV文件中的测点。
    
    参数:
    save_duplicates: 如果为 True，则分别保存两个文件中重复测点的数据到两个 CSV 文件。
    """
    
    # 检查文件是否存在
    if not os.path.exists(file1_path) or not os.path.exists(file2_path):
        print("错误：输入文件路径不存在，请检查。")
        return
    
    try:
        # 1. 读取CSV文件
        df1 = pd.read_csv(file1_path, header=0)
        df2 = pd.read_csv(file2_path, header=0)
        
        # 2. 找到共有的测点（交集）
        points1 = set(df1.columns)
        points2 = set(df2.columns)
        duplicate_points = sorted(list(points1 & points2))
        
        # 3. 执行保存逻辑
        if save_duplicates:
            if duplicate_points:
                # 获取原始文件名（不含扩展名）
                name1 = os.path.splitext(os.path.basename(file1_path))[0]
                name2 = os.path.splitext(os.path.basename(file2_path))[0]
                
                # 分别保存
                file1_out = f"{name1}_重复数据.csv"
                file2_out = f"{name2}_重复数据.csv"
                
                df1[duplicate_points].to_csv(file1_out, index=False, encoding='utf-8-sig')
                df2[duplicate_points].to_csv(file2_out, index=False, encoding='utf-8-sig')
                
                print(f"\n✅ 数据已保存：")
                print(f"   - 文件1重复部分 -> {file1_out}")
                print(f"   - 文件2重复部分 -> {file2_out}")
            else:
                print("\nℹ️  未发现重复测点，跳过保存步骤。")

        # 4. 生成报告
        only_in_1 = sorted(list(points1 - points2))
        only_in_2 = sorted(list(points2 - points1))
        
        report = [
            "=" * 60, "测点对比结果报告", "=" * 60,
            f"文件1: {file1_path}", f"文件2: {file2_path}",
            f"\n[统计]",
            f"- 共有测点数: {len(duplicate_points)}",
            f"- 仅文件1特有: {len(only_in_1)}",
            f"- 仅文件2特有: {len(only_in_2)}",
            f"\n[共有测点详情]:",
            ", ".join(duplicate_points) if duplicate_points else "无"
        ]
        
        result_text = "\n".join(report)
        print("\n" + result_text)

    except Exception as e:
        print(f"处理过程中出错: {str(e)}")

def main():
    # 示例用法
    # True -> 保存两个CSV文件中的重复测点数据
    # False -> 仅输出对比报告
    compare_csv_points("TIC_1101低波动.csv", "TIC_1101正常.csv", save_duplicates=True)

if __name__ == "__main__":
    main()