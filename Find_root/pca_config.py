import os

class Config:
    # ==========================================
    # 1. 路径与目录配置 (Paths Configuration)
    # ==========================================
    DATA_BASE_DIR = os.path.join('..', 'data', '异常传播', '主提升管异常')
    TRAIN_DATA_PATH = os.path.join(DATA_BASE_DIR, '反再全测点正常数据.csv')
    TEST_DATA_PATH = os.path.join(DATA_BASE_DIR, '提升管阀门开口全开.csv')
    
    MODEL_SAVE_PATH = 'pca_pipeline.pkl'
    RESULT_SAVE_DIR = 'PCA_saved'

    # ==========================================
    # 2. 数据预处理参数 (Preprocessing Parameters)
    # ==========================================
    USE_WAVELET_DENOISING = True  
    WAVELET_TYPE = 'db3' 
    WAVELET_LEVEL = 3 
    WAVELET_THRESHOLD_COEFF = 2.0

    # ==========================================
    # 3. RNSPCA 模型超参数 (Model Parameters)
    # ==========================================
    N_COMPONENTS = 8       
    SPARSITY_K = 4         
    SIGMA = 1.0            
    USE_WINDOW = True      
    WINDOW_SIZE = 5        
    ALPHA = 0.001           

    # ==========================================
    # 4. 故障诊断与可视化参数 (Diagnosis Parameters)
    # ==========================================
    DIAGNOSE_TOP_K = 10    
    # 已移除 STAT_TYPE = 'SPE' 参数，当前系统默认仅计算和使用 SPE