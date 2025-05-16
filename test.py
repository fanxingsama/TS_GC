from pathlib import Path
import joblib
from sklearn import preprocessing
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from model.Granger_causalFormer import PredictModel
from data_loader import TimeSeriesDataloader
import pandas as pd
import os

# 设置 Matplotlib 中文显示
rcParams['font.family'] = 'Microsoft YaHei'
rcParams['axes.unicode_minus'] = False


def load_model(config, best_params, device):
    """
    根据最佳参数加载模型
    """
    # 从最佳参数中提取模型参数
    d_model = best_params['d_model']
    n_head = best_params['n_head']
    n_layers = best_params['n_layers']
    ffn_hidden = best_params['ffn_hidden']
    dropout = best_params['dropout']
    tau = best_params['tau']
    
    # GrangerTCN 参数
    tcn_channels = best_params['tcn_channels']
    tcn_kernel_size = best_params['tcn_kernel_size']
    tcn_dropout = best_params['tcn_dropout']
    
    # 创建并返回模型
    model = PredictModel(
        config=config,
        d_model=d_model,
        n_head=n_head,
        n_layers=n_layers,
        tcn_channels=tcn_channels,
        tcn_kernel_size=tcn_kernel_size,
        tcn_dropout=tcn_dropout,
        ffn_hidden=ffn_hidden,
        drop_prob=dropout,
        tau=tau
    ).to(device)
    
    return model

run_id = "05-15_11-01-59"  # 
png_save_path = Path('saved') / run_id


data_path = 'data/fMRI/timeseries9.csv'
gc_dir = 'data/fMRI/sim9_gt_processed.csv'
BATCH_SIZE = 64
DATA_SEED = 42
INPUT_WINDOW = 20
OUTPUT_WINDOW = 1
FEATURE_DIM = 1
OUTPUT_DIM = 1

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 加载数据
timeseriesDataLoader = TimeSeriesDataloader(data_dir=data_path, gc_dir=gc_dir, batch_size=BATCH_SIZE, 
                                        DATA_SEED=DATA_SEED, input_window=INPUT_WINDOW, output_window=OUTPUT_WINDOW)

# 获取数据加载器和序列数量
_, _, test_loader = timeseriesDataLoader.split_sampler()
series_num = timeseriesDataLoader.series_num

# 加载模型最佳参数
best_params_file = png_save_path / "best_params.pkl"
best_params = joblib.load(best_params_file)
config = {
    'data_loader': {
        'args': {
            'input_window': INPUT_WINDOW,
            'output_window': OUTPUT_WINDOW,
            'feature_dim': FEATURE_DIM,
            'output_dim': OUTPUT_DIM,
            'series_num': series_num
        }
    },
    'device': device.type
}
model = load_model(config, best_params, device) # 构建模型

# test = model.encoder.layers[0].attention.tcn_processors[1]
# for name, param in model.named_parameters():
#     print(name)

for i in range(5):
    for (name, param) in model.named_parameters():
        # 检查参数是否属于当前TCN
        if f"encoder.layers.0.attention.tcn_processors.{i}" in name:
            print('++++++++++',name)
    print('-------------')