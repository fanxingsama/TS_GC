# config.py
import torch
from util import create_data
import os


DATA_PATH = os.path.join('data', 'simu_data', 'series_data.csv')
GC_PATH = os.path.join('data', 'simu_data', 'granger_causality.csv')

# DATA_PATH = os.path.join('data', 'data_fz_2000.csv')
# GC_PATH = None

DATA_SEED = 42
BATCH_SIZE = 4096

# 训练参数
INPUT_WINDOW = 30
OUTPUT_WINDOW = 3
FEATURE_DIM = 64
OUTPUT_DIM = 1
EPOCHS = 10000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载数据
X_DATA, Y_DATA, SERIES_NUM, SERIES_NAME = create_data(DATA_PATH, INPUT_WINDOW, OUTPUT_WINDOW)
