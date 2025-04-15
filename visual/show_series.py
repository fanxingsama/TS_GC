import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import re  # 导入正则表达式模块

# 设置字体为SimHei，这是Windows系统常用的中文字体
rcParams['font.family'] = 'SimHei'
rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 读取CSV文件
file_path = 'data/fMRI/timeseries9.csv'  # 替换为你的文件路径
data = pd.read_csv(file_path)

# 打印列名，方便用户选择
print("可用的列名：")
print(data.columns.tolist())
# 预处理列名，忽略大小写、下划线和后缀
def preprocess_column_name(col_name):
    # 去掉后缀（如.PV）
    col_name = re.sub(r'\.PV$', '', col_name)
    # 去掉下划线
    col_name = col_name.replace('_', '')
    # 转换为小写
    col_name = col_name.lower()
    return col_name

# 预处理所有列名
preprocessed_columns = {preprocess_column_name(col): col for col in data.columns}

# 用户输入列名
user_input = input("请输入要可视化的列名（例如：fic1123）：").strip().lower()

# 检查用户输入是否匹配
if user_input in preprocessed_columns:
    target_column = preprocessed_columns[user_input]
    print(f"匹配到的列名：{target_column}")
else:
    print(f"错误：未找到匹配的列名 '{user_input}'！")
    exit()

# 用户输入要显示的数据点数量
try:
    num_points = int(input(f"请输入要显示的数据点数量（最大 {len(data)}）："))
    if num_points <= 0:
        print("错误：数据点数量必须大于0！")
    else:
        # 限制数据点数量不超过数据长度
        num_points = min(num_points, len(data))

        # 提取数据
        selected_data = data[target_column].iloc[:num_points]
        
        # 获取数据的基本统计信息
        data_mean = selected_data.mean()
        data_std = selected_data.std()
        
        # 计算适当的y轴范围，例如均值±3倍标准差
        y_min = data_mean - 4 * data_std
        y_max = data_mean + 4 * data_std

        # 可视化
        plt.figure(figsize=(10, 6))
        plt.plot(selected_data, linestyle='-')
        plt.title(f"时序数据可视化 : {target_column}")
        
        # 设置y轴范围
        plt.ylim(y_min, y_max)
        
        
        plt.show()
except ValueError:
    print("错误：请输入一个有效的整数！")