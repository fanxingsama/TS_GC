import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from matplotlib import rcParams

rcParams['font.family'] = 'SimHei'
rcParams['axes.unicode_minus'] = False

# === 配置区域 ===
csv_file_path = '增压机故障/增压机出口阀关闭_重复数据.csv'  # 在这里修改你的CSV文件路径
output_folder = './增压机故障/出口阀关闭无SIS图片保存'   # 图片保存的文件夹名称
points_to_plot = None            # 选择画多少个点
# ================

def visualize_sensors(file_path, save_dir, limit=1000):
    # 1. 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"错误：找不到文件 {file_path}")
        return

    # 2. 创建保存目录
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"已创建文件夹：{save_dir}")

    try:
        print("正在读取CSV文件...")
        df = pd.read_csv(file_path, nrows=limit, encoding='utf-8') 
        
        # 尝试将所有列转换为数值型，无法转换的变为NaN
        df = df.apply(pd.to_numeric, errors='coerce')
        
        # 4. 设置绘图风格和字体（防止中文乱码）
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial'] # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
        plt.style.use('default') # 使用一种比较好看的图表风格

        columns = df.columns
        total_cols = len(columns)
        print(f"共发现 {total_cols} 个测点，准备开始绘图...")

        # 5. 循环绘图
        for i, col_name in enumerate(columns):
            # 跳过全为空的列
            if df[col_name].isna().all():
                print(f"跳过空列: {col_name}")
                continue

            plt.figure(figsize=(10, 4)) # 设置图片大小 (宽, 高)
            
            # 绘制折线图
            plt.plot(df.index, df[col_name], linewidth=1.5)
            
            plt.title(f"{col_name}", fontsize=12)
            # plt.xlabel("pot")
            plt.ylabel("Value")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            # 6. 处理文件名 (非常重要：把不能作为文件名的字符替换掉)
            # 例如：有的测点叫 "P-101/A"，斜杠会导致保存路径错误，替换为 "_"
            safe_name = str(col_name).replace('/', '_').replace('\\', '_').replace(':', '').replace('*', '')
            save_path = os.path.join(save_dir, f"{safe_name}.png")

            plt.savefig(save_path, dpi=100) # dpi=100 保证清晰度且文件不会太大
            plt.close() # 关闭画布，释放内存（非常重要，否则画几百张图内存就爆了）

            # 打印进度
            if (i + 1) % 10 == 0:
                print(f"进度: {i + 1}/{total_cols} 已保存 -> {safe_name}.png")

        print(f"\n✅ 所有图片绘制完成！请查看文件夹: {os.path.abspath(save_dir)}")

    except Exception as e:
        print(f"发生错误: {e}")

if __name__ == "__main__":
    visualize_sensors(csv_file_path, output_folder, points_to_plot)