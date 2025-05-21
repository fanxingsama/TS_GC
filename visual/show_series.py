from matplotlib import rcParams
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

rcParams['font.family'] = 'SimHei'
rcParams['axes.unicode_minus'] = False
def visualize_time_series(file_path, selected_columns=None, sequence_length=None):
    """
    可视化多变量时序序列CSV文件
    
    参数:
    file_path (str): CSV文件路径
    selected_columns (list): 需要可视化的列名列表，如果为None则提示用户选择
    sequence_length (int): 要显示的序列长度，如果为None则显示全部
    """
    try:
        # 读取CSV文件
        df = pd.read_csv(file_path)
        
        # 获取所有列名（序列名）
        all_columns = df.columns.tolist()
        
        # 如果没有指定要可视化的列，让用户交互选择
        if selected_columns is None:
            print("可用的序列:")
            for i, col in enumerate(all_columns):
                print(f"{i+1}. {col}")
            
            # 获取用户输入
            selection = input("请输入要可视化的序列编号（用逗号分隔，例如：1,3,5）: ")
            try:
                # 将用户输入转换为索引列表
                indices = [int(idx.strip()) - 1 for idx in selection.split(',')]
                selected_columns = [all_columns[i] for i in indices if 0 <= i < len(all_columns)]
                
                if not selected_columns:
                    print("未选择有效的序列，将显示所有序列")
                    selected_columns = all_columns
            except:
                print("输入格式不正确，将显示所有序列")
                selected_columns = all_columns
        
        # 如果没有指定序列长度，让用户交互输入
        if sequence_length is None:
            length_input = input(f"请输入要显示的序列长度（最大 {len(df)}，按Enter显示全部）: ")
            try:
                if length_input.strip():
                    sequence_length = int(length_input)
                    if sequence_length <= 0 or sequence_length > len(df):
                        print(f"输入的长度无效，将显示全部长度: {len(df)}")
                        sequence_length = len(df)
                else:
                    sequence_length = len(df)
            except:
                print(f"输入格式不正确，将显示全部长度: {len(df)}")
                sequence_length = len(df)
        
        # 截取数据
        df_subset = df.iloc[:sequence_length]
        
        # 为每个选定的序列创建单独的图表
        num_plots = len(selected_columns)
        fig, axes = plt.subplots(num_plots, 1, figsize=(12, 3 * num_plots))
        
        # 如果只有一列，axes不是列表，需要转换为列表以便统一处理
        if num_plots == 1:
            axes = [axes]
        
        # 绘制每个序列
        for i, col in enumerate(selected_columns):
            time_index = np.arange(len(df_subset))
            axes[i].plot(time_index, df_subset[col], 'b-', linewidth=1.5)
            axes[i].set_title(f"时序序列: {col}")
            axes[i].set_xlabel("时间步")
            axes[i].set_ylabel("值")
            axes[i].grid(True)
        
        plt.tight_layout()
        
        # 显示图表
        plt.show()
        
        return True
        
    except Exception as e:
        print(f"错误: {str(e)}")
        return False

def main():
    file_path = '../data/simu_data/series_data.csv'
    # file_path = '../data/fMRI/timeseries9.csv'
    
    parser = argparse.ArgumentParser(description='多变量时序序列可视化工具')
    parser.add_argument('--columns', type=str, help='要可视化的列（用逗号分隔）')
    parser.add_argument('--length', type=int, help='要显示的序列长度')
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"错误: 文件 '{file_path}' 不存在")
        return
    
    # 解析列参数
    selected_columns = None
    if args.columns:
        selected_columns = [col.strip() for col in args.columns.split(',')]
    
    # 可视化时序序列
    visualize_time_series(file_path, selected_columns, args.length)

if __name__ == "__main__":
    main()