# config.py
import torch
from util.util import create_data
import os


# DATA_PATH = os.path.join('data', 'virtual','simu_data', 'series_data2.csv')
# GC_PATH = os.path.join('data', 'virtual', 'simu_data', 'granger_causality2.csv')

# DATA_PATH = os.path.join('data', 'data_use.csv')
# GC_PATH = None

DATA_PATH = os.path.join('data', 'virtual','fMRI', 'timeseries6.csv')
GC_PATH = os.path.join('data', 'virtual', 'fMRI', 'sim6_gt_processed.csv')

# DATA_PATH = os.path.join('data', 'potential_var.csv')
# GC_PATH = None

INPUT_WINDOW = 5
OUTPUT_WINDOW = 1

# 训练参数
LR = 0.01      
LASSO_PARAM = 0.02
RIDGE_PARAM = 0.001
PENALTY_TYPE = 'H' # 'GSGL'和'GL'
KERNAL_SIZE = 5    
DROUP_OUT = 0     
TEMPORAL_LAYERS = 2
FEATURE_DIM = 32
OUTPUT_DIM = 1
EPOCHS = 10000
APPLY_MASK = False
LOSS_FUNCTION = torch.nn.MSELoss()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 加载数据
X_DATA, Y_DATA, SERIES_NUM, SERIES_NAME = create_data(DATA_PATH, INPUT_WINDOW, OUTPUT_WINDOW)