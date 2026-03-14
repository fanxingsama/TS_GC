# config.py
import torch
from util.util import create_data, get_latest_run_id
import os
from pathlib import Path

run_id = get_latest_run_id(base_path_str='Find_root/PCA_saved')
model_path = Path('saved') / run_id
# DATA_PATH = os.path.join('Find_root', 'PCA_saved', run_id, 'potential_var.csv')
# GC_PATH = None

# DATA_PATH = os.path.join('data','virtual', 'time_series_linear.csv')
# GC_PATH = os.path.join('data','virtual', 'causal_relations.csv')

DATA_PATH = os.path.join('data','casual', 'RRP_data.csv')
GC_PATH = os.path.join('data','casual', 'RRP_causal_true.csv')

INPUT_WINDOW = 5
OUTPUT_WINDOW = 1

# 训练参数
LR = 0.01      
LASSO_PARAM = 0.040
RIDGE_PARAM = 0.001
PENALTY_TYPE = 'H' # 'GSGL'和'GL'和'H'
KERNAL_SIZE = 5    
DROUP_OUT = 0     
TEMPORAL_LAYERS = 2
FEATURE_DIM = 32
OUTPUT_DIM = 1
EPOCHS = 10000
APPLY_MASK = False
LOSS_FUNCTION = torch.nn.MSELoss()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 新增：消融实验配置 (Ablation Settings) ---
USE_TEMPORAL = True   # 是否使用多尺度时间层
USE_SPATIAL = True    # 是否使用全局空间池化层
USE_RESIDUAL = True   # 是否在最终特征中融合第一层原始特征的残差

# --- 新增：自因果关系处理 (Self-Causality Handling) ---
IGNORE_SELF_CAUSALITY = False  # False表示考虑自因果，True表示忽略自因果

# 加载数据
X_DATA, Y_DATA, SERIES_NUM, SERIES_NAME = create_data(DATA_PATH, INPUT_WINDOW, OUTPUT_WINDOW)