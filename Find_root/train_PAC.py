import pandas as pd
import pickle
import os
from RSPCA import WaveletDenoiser, RobustScaler, RNSPCA
from pca_config import Config

def train_from_csv(config):
    df_train = pd.read_csv(config.TRAIN_DATA_PATH)
    
    # 从 config 引入预处理参数
    denoiser = WaveletDenoiser(
        wavelet=config.WAVELET_TYPE, 
        level=config.WAVELET_LEVEL,
        threshold_coeff=config.WAVELET_THRESHOLD_COEFF
    )
    scaler = RobustScaler()
    
    # 从 config 引入模型参数
    model = RNSPCA(
        n_components=config.N_COMPONENTS, 
        sparsity_k=config.SPARSITY_K, 
        sigma=config.SIGMA,
        window_size=config.WINDOW_SIZE, 
        use_window=config.USE_WINDOW,
        alpha=config.ALPHA
    )

    X_denoised = denoiser.transform(df_train)
    scaler.fit(X_denoised)
    X_scaled = scaler.transform(X_denoised)
    
    print(">>> 正在训练 RNSPCA 模型 (这可能需要一点时间)...")
    model.fit_offline(X_scaled)

    # 保存模型
    with open(config.MODEL_SAVE_PATH, 'wb') as f:
        pickle.dump({'denoiser': denoiser, 'scaler': scaler, 'model': model}, f)
    print(f">>> 建模完成！模型已保存至: {config.MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train_from_csv(Config)