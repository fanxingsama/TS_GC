import pandas as pd

# 输入文件路径
input_file = 'train.csv'  # 原始大文件的路径
output_prefix = 'small_train_'  # 输出文件的前缀
num_splits = 10  # 切分成10个小文件

# 读取CSV文件
df = pd.read_csv(input_file)

# 获取表头
header = df.columns.tolist()

# 计算每个小文件的行数
rows_per_file = len(df) // num_splits

# 切分文件
for i in range(num_splits):
    start_row = i * rows_per_file
    end_row = (i + 1) * rows_per_file if i < num_splits - 1 else len(df)
    
    # 选择当前片段的数据
    subset = df.iloc[start_row:end_row]
    
    # 写入到新的CSV文件
    output_file = f'{output_prefix}{i + 1}.csv'
    subset.to_csv(output_file, index=False, header=header)

print(f"数据已成功切分成{num_splits}个小文件，每个文件包含表头。")