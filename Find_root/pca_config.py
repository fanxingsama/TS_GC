import os

class Config:
    # ==========================================
    # 1. 路径与目录配置 (Paths Configuration)
    # ==========================================
    # 基础数据目录 (假设与脚本同级的 data 文件夹)
    DATA_BASE_DIR = os.path.join('..', 'data', '异常传播', '主提升管异常')
    
    # 训练与测试数据路径
    TRAIN_DATA_PATH = os.path.join(DATA_BASE_DIR, '反再全测点正常数据.csv')
    TEST_DATA_PATH = os.path.join(DATA_BASE_DIR, '提升管阀门开口全开.csv')
    
    # 模型与结果保存路径
    MODEL_SAVE_PATH = 'pca_pipeline.pkl'
    RESULT_SAVE_DIR = 'PCA_saved'

    # ==========================================
    # 2. 数据预处理参数 (Preprocessing Parameters)
    # ==========================================
    WAVELET_TYPE = 'sym8' # sym8/db2/db3
    WAVELET_LEVEL = 4 # 1/2/3
    WAVELET_THRESHOLD_COEFF = 2.0

    # ==========================================
    # 3. RNSPCA 模型超参数 (Model Parameters)
    # ==========================================
    N_COMPONENTS = 6       # 主成分数量
    SPARSITY_K = 4         # 每个主成分保留的非零元素数量
    SIGMA = 1.0            # 核宽度参数 (代码内部会自动乘以 1.4826)
    USE_WINDOW = True      # 是否启用滑动窗口 (时序嵌入)
    WINDOW_SIZE = 5        # 滑动窗口大小 (结合你原本在 train_PAC.py 中的设置)
    ALPHA = 0.01           # 显著性水平 (用于计算全局异常阈值)

    # ==========================================
    # 4. 故障诊断与可视化参数 (Diagnosis Parameters)
    # ==========================================
    DIAGNOSE_TOP_K = 10    # 提取排名前 K 的异常变量
    STAT_TYPE = 'SPE'      # 诊断统计量标准，可选 'T2' 或 'SPE'