# config.py
import torch
from util import *


DATA_PATH = 'data\\simu_data\\series_data.csv'
GC_PATH = 'data\\simu_data\\granger_causality.csv'

# DATA_PATH = 'data/data_1.csv'
# GC_PATH = None

# 训练参数
INPUT_WINDOW = 5
OUTPUT_WINDOW = 1
FEATURE_DIM = 1
OUTPUT_DIM = 1
EPOCHS = 20000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


X_DATA, Y_DATA, SERIES_NUM, SERIES_NAME = create_data(DATA_PATH, INPUT_WINDOW, OUTPUT_WINDOW)

