import pandas as pd
import pickle
from pca_core import WaveletDenoiser, RobustScaler, RNSPCA

def train_from_csv(file_path, model_path='pca_pipeline.pkl'):
    df_train = pd.read_csv(file_path)
    denoiser = WaveletDenoiser(wavelet='sym8', level=3)
    scaler = RobustScaler()
    model = RNSPCA(n_components=6, sparsity_k=4)

    X_denoised = denoiser.transform(df_train)
    scaler.fit(X_denoised)
    X_scaled = scaler.transform(X_denoised)
    
    print(">>> 正在训练 RNSPCA 模型 (这可能需要一点时间)...")
    model.fit_offline(X_scaled)

    # 保存模型
    with open(model_path, 'wb') as f:
        pickle.dump({'denoiser': denoiser, 'scaler': scaler, 'model': model}, f)
    print(f">>> 建模完成！模型已保存至: {model_path}")

if __name__ == "__main__":
    # 请修改为你的实际正常数据路径
    train_from_csv('normal_重复.csv')