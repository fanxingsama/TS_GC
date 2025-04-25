import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import re  # 导入正则表达式模块
import numpy as np  # 导入numpy用于数据处理
import math  # 导入math用于计算子图排列

# 设置字体为SimHei，这是Windows系统常用的中文字体
rcParams['font.family'] = 'SimHei'
rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 读取CSV文件
file_path = '1744883745.csv'  # 替换为你的文件路径
data = pd.read_csv(file_path)

# 打印列名，方便用户选择
print("可用的列名：")
print(data.columns.tolist())

# 预处理列名，忽略大小写、下划线和后缀
def preprocess_column_name(col_name):
    # 去掉后缀（如.PV）
    col_name = re.sub(r'\.(PV|OUT|SV)$', '', col_name)
    # 去掉下划线
    col_name = col_name.replace('_', '')
    # 转换为小写
    col_name = col_name.lower()
    return col_name

# 预处理所有列名
preprocessed_columns = {preprocess_column_name(col): col for col in data.columns}

# 用户输入多个列名，用逗号分隔
user_input = input("请输入要可视化的列名，多个列名用逗号分隔（例如：fic1123,fic2234），或输入 * 显示前8列：").strip().lower()

matched_columns = []

# 检查是否为通配符 "*"
if user_input == "*":
    # 如果输入是 "*"，则取数据框的前8列
    matched_columns = data.columns[:8].tolist()
    print("将显示前8列数据：")
    for col in matched_columns:
        print(f"- {col}")
else:
    # 否则按用户输入的列名进行匹配
    column_inputs = [name.strip() for name in user_input.split(',')]
    
    # 检查每个输入列名是否匹配
    for col_input in column_inputs:
        if col_input in preprocessed_columns:
            target_column = preprocessed_columns[col_input]
            matched_columns.append(target_column)
            print(f"匹配到的列名：{target_column}")
        else:
            print(f"警告：未找到匹配的列名 '{col_input}'！")

# 如果没有匹配到任何列，则退出
if not matched_columns:
    print("错误：未找到任何匹配的列名！")
    exit()

# 限制最多只能选择8个序列
if len(matched_columns) > 8:
    print(f"警告：选择的序列超过8个，将只显示前8个序列")
    matched_columns = matched_columns[:8]

# 用户输入要显示的数据点数量
try:
    num_points = int(input(f"请输入要显示的数据点数量（最大 {len(data)}）："))
    if num_points <= 0:
        print("错误：数据点数量必须大于0！")
    else:
        # 限制数据点数量不超过数据长度
        num_points = min(num_points, len(data))

        # 计算子图布局
        n_cols = min(4, len(matched_columns))  # 每行最多4个子图
        n_rows = math.ceil(len(matched_columns) / n_cols)  # 根据序列数量计算需要的行数
        
        # 创建图表，子图之间留出足够的空间
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows), squeeze=False)
        
        # 为每个匹配的列创建单独的子图
        for i, column in enumerate(matched_columns):
            # 计算当前子图的行和列索引
            row = i // n_cols
            col = i % n_cols
            
            # 提取数据
            selected_data = data[column].iloc[:num_points]
            
            # 获取数据的基本统计信息用于设置y轴范围
            data_mean = selected_data.mean()
            data_std = selected_data.std()
            y_min = data_mean - 6 * data_std
            y_max = data_mean + 6 * data_std
            
            # 绘制子图
            axes[row, col].plot(selected_data, linestyle='-', color='blue')
            axes[row, col].set_title(column)
            axes[row, col].set_xlabel("时间点")
            axes[row, col].set_ylabel("值")
            axes[row, col].grid(True, linestyle='--', alpha=0.7)
            axes[row, col].set_ylim(y_min, y_max)
        
        # 隐藏未使用的子图
        for i in range(len(matched_columns), n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            fig.delaxes(axes[row, col])
        
        # 调整布局，确保子图之间有足够的空间，标题和标签不重叠
        plt.tight_layout(pad=3.0)
        plt.suptitle("多序列时序数据可视化", fontsize=16, y=1.02)
        
        # 显示图表
        plt.show()
        
except ValueError:
    print("错误：请输入一个有效的整数！")