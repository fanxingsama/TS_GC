import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
import re
from pathlib import Path

# 将时间序列数据处理为适合时间序列预测模型训练
def create_data(data_path, input_window, output_window):
    prediction_input_df = pd.read_csv(data_path)
    series_names = prediction_input_df.columns.tolist() # 提取序列名称
    
    data_np = prediction_input_df[series_names].values.astype(np.float32) 
    num_timesteps, num_series = data_np.shape
    
    # 数据归一化
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_np = scaler.fit_transform(data_np)

    # 构建输入输出数据
    X_list, Y_list = [], []
    for i in range(num_timesteps - input_window - output_window + 1):
        X_list.append(data_np[i : i + input_window, :])
        Y_list.append(data_np[i + input_window : i + input_window + output_window, :])
    
    X_data = torch.tensor(np.array(X_list), dtype=torch.float32)
    Y_data = torch.tensor(np.array(Y_list), dtype=torch.float32)
    
    return X_data, Y_data, num_series, series_names

def get_latest_run_id():
    base_path = Path('saved')
    if not base_path.exists():
        return None
    
    # 获取所有符合格式的目录名
    pattern = re.compile(r'^\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$')
    timestamps = [d.name for d in base_path.iterdir() 
                 if d.is_dir() and pattern.match(d.name)]
    
    return max(timestamps) if timestamps else None