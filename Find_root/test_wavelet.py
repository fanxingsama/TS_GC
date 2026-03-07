import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from RSPCA import WaveletDenoiser

def test_wavelet_denoising(file_path, target_var_name):
    """
    测试不同小波降噪参数对原始时间序列的影响，并绘制对比图。
    
    :param file_path: CSV 数据文件路径
    :param target_var_name: 要可视化的测点名称 (列名字符串)
    """
    print(f">>> 正在读取数据: {file_path}")
    df = pd.read_csv(file_path)
    
    # --- 新增：检查测点名是否存在并获取索引 ---
    if target_var_name not in df.columns:
        available_cols = df.columns.tolist()
        raise ValueError(f"\n❌ 找不到测点名: '{target_var_name}'\n"
                         f"💡 数据集中可用的测点名有: {available_cols[:10]} ... (共 {len(available_cols)} 个)")
        
    var_index = df.columns.get_loc(target_var_name)
    original_signal = df[target_var_name].values
    # ----------------------------------------
    
    # ==========================================
    # 定义要对比的小波参数组
    # ==========================================
    configs = [
        {'name': 'Original (sym8, level=5)', 'wavelet': 'sym8', 'level': 5, 'color': '#2ca02c'},  # 原配置：绿色
        {'name': 'Tuned (db3, level=3)', 'wavelet': 'db3', 'level': 3, 'color': '#ff7f0e'}        # 建议配置：橙色
    ]
    
    results = {}

    print(">>> 正在应用小波降噪...")
    for cfg in configs:
        denoiser = WaveletDenoiser(wavelet=cfg['wavelet'], level=cfg['level'])
        denoised_data = denoiser.transform(df)
        results[cfg['name']] = {
            'signal': denoised_data[:, var_index], # 使用获取到的列索引提取数据
            'color': cfg['color']
        }
    
    print(">>> 正在生成对比图表...")
    # ==========================================
    # 绘图逻辑：子图1为全局视图，子图2为局部放大视图
    # ==========================================
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), dpi=150)
    
    time_steps = np.arange(len(original_signal))
    # anomaly_injection_point = 165 # 异常注入点
    
    # --- 子图 1: 全局视图 ---
    ax1 = axes[0]
    ax1.plot(time_steps, original_signal, label='Raw Signal', color='#1f77b4', alpha=0.5, linewidth=1.5)
    for name, data in results.items():
        ax1.plot(time_steps, data['signal'], label=name, color=data['color'], alpha=0.8, linewidth=1.5)
        
    # ax1.axvline(x=anomaly_injection_point, color='crimson', linestyle='--', linewidth=1.5, label='Anomaly Injection (t=165)')
    ax1.set_title(f"Global View: Wavelet Denoising Comparison (Feature: {target_var_name})", fontsize=14)
    ax1.set_ylabel("Sensor Value", fontsize=12)
    ax1.legend(loc='upper left')
    ax1.grid(True, linestyle=':', alpha=0.5)

    # --- 子图 2: 局部放大视图 (聚焦异常发生前后 130 ~ 200 之间) ---
    ax2 = axes[1]
    zoom_start, zoom_end = 130, 200 
    
    ax2.plot(time_steps, original_signal, label='Raw Signal', color='#1f77b4', alpha=0.5, marker='.', markersize=6)
    for name, data in results.items():
        ax2.plot(time_steps, data['signal'], label=name, color=data['color'], alpha=0.8, marker='.', markersize=6)
        
    # ax2.axvline(x=anomaly_injection_point, color='crimson', linestyle='--', linewidth=2, label='Anomaly Injection (t=165)')
    ax2.set_xlim(zoom_start, zoom_end)
    ax2.set_title("Zoomed-in View: Inspecting Pre-Anomaly Gibbs Ringing (t=130 to 200)", fontsize=14)
    ax2.set_xlabel("Time Step", fontsize=12)
    ax2.set_ylabel("Sensor Value", fontsize=12)
    ax2.legend(loc='upper left')
    ax2.grid(True, linestyle=':', alpha=0.5)

    plt.tight_layout()
    
    # 保存图片，为了防止文件名有非法字符，做了简单的替换
    safe_var_name = str(target_var_name).replace('/', '_').replace('\\', '_')
    save_path = f"wavelet_comparison_{safe_var_name}.png"
    plt.savefig(save_path, bbox_inches='tight')
    print(f">>> 对比图表已保存至: {save_path}")
    
    plt.show()

if __name__ == "__main__":
    DATA_PATH = os.path.join('..', 'data', '异常传播', '主提升管异常', '提升管阀门开口全开.csv')
    
    # 在这里输入你想要观察的具体测点名称
    # 例如：'FI-101' 或 '主提升管温度'
    TARGET_SENSOR = 'TIA2877.PV' 
    
    test_wavelet_denoising(DATA_PATH, target_var_name=TARGET_SENSOR)