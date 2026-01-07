import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from RSPCA import WaveletDenoiser, RobustScaler, RNSPCA

def diagnose_from_csv(file_path, model_path='pca_pipeline.pkl', output_img='diagnostic_report.png'):
    """
    读取异常 CSV 并进行诊断
    """
    # 1. 加载模型
    try:
        with open(model_path, 'rb') as f:
            pipeline = pickle.load(f)
    except FileNotFoundError:
        print("错误：未找到模型文件，请先运行 model_trainer.py")
        return

    denoiser = pipeline['denoiser']
    scaler = pipeline['scaler']
    model = pipeline['model']
    # 查看模型里存储的正常基准值
    print("正常工况 T2 基准值 (前5个变量):", model.normal_baseline_T2[:5])
    print("正常工况 SPE 基准值 (前5个变量):", model.normal_baseline_SPE[:5])

    # 2. 读取测试数据
    print(f">>> 正在读取待检测数据: {file_path}")
    df_test = pd.read_csv(file_path)
    # 获取变量名称（用于图表展示）
    feature_names = df_test.columns.tolist()

    # 3. 数据预处理与计算
    X_test_denoised = denoiser.transform(df_test)
    X_test_scaled = scaler.transform(X_test_denoised)
    
    # 触发诊断计算
    diag_res = model.trigger_diagnose(X_test_scaled)
    dcc_scores = diag_res['dcc_norm']
    
    # 4. 筛选 Top 10 贡献变量
    top_indices = np.argsort(dcc_scores)[::-1][:10]
    
    print("\n" + "="*40)
    print(f"{'排名':<6} | {'变量索引':<10} | {'变量名称':<20} | {'贡献得分'}")
    print("-"*40)
    for i, idx in enumerate(top_indices):
        name = feature_names[idx] if idx < len(feature_names) else f"Var_{idx}"
        print(f"{i+1:<8} | {idx:<12} | {name:<22} | {dcc_scores[idx]:.4f}")
    print("="*40)

    # 5. 可视化保存
    plot_results(diag_res, dcc_scores, top_indices, feature_names, output_img, model.lmvt)

def plot_results(diag_res, dcc_scores, top_indices, feature_names, output_img, lmvt):
    n_vars = len(dcc_scores)
    plt.figure(figsize=(12, 6))
    x_idx = np.arange(n_vars)
    
    # # 子图1: T2 贡献度对比
    # plt.subplot(3, 1, 1)
    # plt.bar(x_idx - 0.2, diag_res['before_T2'], 0.4, label='Normal Baseline', color='gray', alpha=0.5)
    # plt.bar(x_idx + 0.2, diag_res['after_T2'], 0.4, label='Fault Current', color='royalblue')
    # plt.title('Hotelling $T^2$ Contribution (Baseline vs Fault)')
    # plt.ylabel('Contribution')
    # plt.legend()

    # # 子图2: SPE 贡献度对比
    # plt.subplot(2, 1, 1)
    # plt.bar(x_idx - 0.2, diag_res['before_SPE'], 0.4, label='Normal Baseline', color='gray', alpha=0.5)
    # plt.bar(x_idx + 0.2, diag_res['after_SPE'], 0.4, label='Fault Current', color='seagreen')
    # plt.title('SPE Contribution (Baseline vs Fault)')
    # plt.ylabel('Contribution')
    # plt.legend()

    # # 子图3: DCC 根因诊断图
    # plt.subplot(2, 1, 2)
    colors = ['crimson' if i in top_indices else 'gold' for i in range(n_vars)]
    plt.bar(x_idx, dcc_scores, color=colors)
    plt.axhline(y=lmvt, color='black', linestyle='--', label=f'Threshold (LMVT={lmvt:.3f})')
    
    # 如果变量名不太长，显示在横轴
    if n_vars <= 30:
        plt.xticks(x_idx, feature_names, rotation=45, ha='right')
    
    plt.title('Root Cause Diagnosis (DCC Score) - Top 10 in Red')
    plt.xlabel('Sensors / Features')
    plt.ylabel('Normalized Score')
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_img)
    print(f"\n>>> 诊断可视化报告已保存至: {output_img}")

if __name__ == "__main__":
    # 请修改为你的实际异常数据路径
    diagnose_from_csv('low_fault_重复.csv')