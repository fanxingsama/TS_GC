# config.py
import torch
from util import *


DATA_PATH = 'data\\simu_data\\series_data1000.csv'
GC_PATH = 'data\\simu_data\\granger_causality1000.csv'

# DATA_PATH = 'data\\fMRI\\timeseries9.csv'
# GC_PATH = 'data\\fMRI\\sim9_gt_processed.csv'

# 训练参数
INPUT_WINDOW = 5
OUTPUT_WINDOW = 1
FEATURE_DIM = 1
OUTPUT_DIM = 1
EPOCHS = 20000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


X_DATA, Y_DATA, SERIES_NUM = create_data(DATA_PATH, INPUT_WINDOW, OUTPUT_WINDOW)

