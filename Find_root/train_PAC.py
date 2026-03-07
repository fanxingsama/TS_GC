import pandas as pd
import pickle
from RSPCA import WaveletDenoiser, RobustScaler, RNSPCA
from pca_config import Config

def train_from_csv(config):
    df_train = pd.read_csv(config.TRAIN_DATA_PATH)
    
    # --- 新增：根据配置决定是否使用小波降噪 ---
    if config.USE_WAVELET_DENOISING:
        print(f">>> 启用小波降噪 (Wavelet: {config.WAVELET_TYPE}, Level: {config.WAVELET_LEVEL})...")
        denoiser = WaveletDenoiser(
            wavelet=config.WAVELET_TYPE, 
            level=config.WAVELET_LEVEL,
            threshold_coeff=config.WAVELET_THRESHOLD_COEFF
        )
        X_preprocessed = denoiser.transform(df_train)
    else:
        print(">>> 禁用小波降噪，直接使用原始数据进行标准化...")
        denoiser = None
        X_preprocessed = df_train.values # 转换为 numpy 数组保持一致格式
    # ----------------------------------------

    scaler = RobustScaler()
    model = RNSPCA(
        n_components=config.N_COMPONENTS, 
        sparsity_k=config.SPARSITY_K, 
        sigma=config.SIGMA,
        window_size=config.WINDOW_SIZE, 
        use_window=config.USE_WINDOW,
        alpha=config.ALPHA
    )

    scaler.fit(X_preprocessed)
    X_scaled = scaler.transform(X_preprocessed)
    
    print(">>> 正在训练 RNSPCA 模型 (这可能需要一点时间)...")
    model.fit_offline(X_scaled)

    # 保存模型 (如果禁用降噪，denoiser 保存为 None)
    with open(config.MODEL_SAVE_PATH, 'wb') as f:
        pickle.dump({'denoiser': denoiser, 'scaler': scaler, 'model': model}, f)
    print(f">>> 建模完成！模型已保存至: {config.MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train_from_csv(Config)