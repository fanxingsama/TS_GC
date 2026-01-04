import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pywt
import os

# ==========================================
# 增强版小波去噪器 (支持力度调节)
# ==========================================
class WaveletDenoiser:
    """
    增强版小波去噪器
    wavelet: 小波基名称 (推荐 'db8' 或 'sym8' 处理平滑信号)
    level: 分解层数 (越高去噪越狠, 建议 3-5)
    threshold_coeff: 阈值系数 (越大去噪越狠, 1.0 为标准值)
    """
    def __init__(self, wavelet='sym8', level=3, threshold_coeff=2.0):
        self.wavelet = wavelet
        self.level = level
        self.threshold_coeff = threshold_coeff

    def transform(self, X):
        data = X.values if isinstance(X, pd.DataFrame) else X
        denoised_data = np.zeros_like(data)
        for i in range(data.shape[1]):
            # 1. 小波分解
            coeffs = pywt.wavedec(data[:, i], self.wavelet, level=self.level)
            # 2. 对每一层高频系数进行阈值处理
            new_coeffs = [coeffs[0]] # 保留低频近似部分
            for j in range(1, len(coeffs)):
                sigma = np.median(np.abs(coeffs[j])) / 0.6745
                uthresh = sigma * np.sqrt(2 * np.log(len(data))) * self.threshold_coeff
                new_coeffs.append(pywt.threshold(coeffs[j], value=uthresh, mode='soft'))
            # 3. 小波重构
            res = pywt.waverec(new_coeffs, self.wavelet)
            denoised_data[:, i] = res[:len(data)]
        return denoised_data

# ==========================================
# SNR 计算函数
# ==========================================
def calculate_snr(original, denoised):
    """
    计算信噪比 (SNR)
    公式: 10 * log10(信号能量 / 噪声能量)
    """
    noise = original - denoised
    # 避免分母为0
    noise_power = np.sum(noise**2)
    if noise_power == 0:
        return float('inf')
    signal_power = np.sum(denoised**2)
    snr = 10 * np.log10(signal_power / noise_power)
    return snr

def run_visualization(csv_path, num_features=3):
    if not os.path.exists(csv_path):
        print(f"错误：找不到文件 {csv_path}")
        return

    # 1. 数据加载
    df = pd.read_csv(csv_path)
    df_numeric = df.select_dtypes(include=[np.number])
    
    # 2. 执行增强去噪 (此处参数调高以观察明显变化)
    # 尝试调整 threshold_coeff: 1.5 -> 3.0 观察变化
    denoiser = WaveletDenoiser(wavelet='sym8', level=3, threshold_coeff=2.0)
    denoised_array = denoiser.transform(df_numeric)
    df_denoised = pd.DataFrame(denoised_array, columns=df_numeric.columns)

    # 3. 绘图与 SNR 显示
    cols_to_plot = df_numeric.columns[:num_features]
    plt.figure(figsize=(12, 4 * num_features))
    
    for i, col in enumerate(cols_to_plot):
        orig_signal = df_numeric[col].values
        denoised_signal = df_denoised[col].values
        
        # 计算该特征的 SNR
        snr_val = calculate_snr(orig_signal, denoised_signal)
        
        plt.subplot(num_features, 1, i + 1)
        plt.plot(orig_signal, label='Raw (Original)', color='green', alpha=0.7)
        plt.plot(denoised_signal, label='Denoised (Wavelet)', color='crimson', linewidth=1.5)
        
        # 在标题中实时显示 SNR
        plt.title(f'Feature: {col} | Denoising SNR: {snr_val:.2f} dB')
        plt.legend(loc='upper right')
        plt.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.savefig('wavelet_snr_comparison.png')
    print(f">>> 带有SNR指标的可视化已保存。")
    print(f">>> 当前参数: Level={denoiser.level}, Coeff={denoiser.threshold_coeff}")

if __name__ == "__main__":
    run_visualization('normal_重复.csv') # 替换为你的文件名