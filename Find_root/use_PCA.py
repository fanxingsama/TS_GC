import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import os
from datetime import datetime
from RSPCA import WaveletDenoiser, RobustScaler, RNSPCA

# 
def diagnose_from_csv(file_path, model_path='pca_pipeline.pkl', top_k=5):
    root_save_dir = 'PCA_saved'
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    save_dir = os.path.join(root_save_dir, current_time)
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f">>> 已创建结果保存文件夹: {save_dir}")
        
    output_img_path = os.path.join(save_dir, 'diagnostic_report.png')
    output_csv_path = os.path.join(save_dir, 'potential_var.csv')

    # 加载模型
    with open(model_path, 'rb') as f:
            pipeline = pickle.load(f)
    model = pipeline['model']

    #数据预处理
    df_test = pd.read_csv(file_path)
    feature_names = df_test.columns.tolist()
    denoiser = pipeline['denoiser']
    scaler = pipeline['scaler']
    X_test_denoised = denoiser.transform(df_test)
    X_test_scaled = scaler.transform(X_test_denoised)
    
    # 诊断计算
    diag_res = model.trigger_diagnose(X_test_scaled)
    dcc_scores = diag_res['dcc_norm']
    
    # 异常变量筛选
    sorted_indices = np.argsort(dcc_scores)[::-1]
    save_indices = sorted_indices[:top_k]
    
    print("\n" + "="*45)
    print(f"【诊断报告】 提取贡献度最高的 Top {top_k} 变量")
    print(f"{'排名':<6} | {'变量索引':<10} | {'变量名称':<20} | {'贡献得分'}")
    print("-"*45)
    
    # 遍历并打印出贡献度最高的 Top-K 异常变量的详细信息
    for i, idx in enumerate(save_indices):
        name = feature_names[idx] if idx < len(feature_names) else f"Var_{idx}"
        print(f"{i+1:<8} | {idx:<12} | {name:<20} | {dcc_scores[idx]:.4f}")
    print("="*45)

    df_potential = df_test.iloc[:, save_indices]
    
    # 保存 CSV
    df_potential.to_csv(output_csv_path, index=False)
    print(f">>> 潜在异常变量数据已分离并保存至: {output_csv_path}")
    print(f"    包含变量索引: {save_indices}")
    plot_results(dcc_scores, save_indices, feature_names, output_img_path, top_k)

# 可视化结果
def plot_results(dcc_scores, highlight_indices, feature_names, output_img, top_k):
    n_vars = len(dcc_scores)
    plt.figure(figsize=(12, 6))
    x_idx = np.arange(n_vars)
    
    # 颜色逻辑：top_k 的变量显示为红色，其余显示为灰色
    colors = ['crimson' if i in highlight_indices else 'lightgray' for i in range(n_vars)]
    
    plt.bar(x_idx, dcc_scores, color=colors, alpha=0.9)
    
    # 如果变量名不太长，显示在横轴
    if n_vars <= 30:
        plt.xticks(x_idx, feature_names, rotation=45, ha='right', fontsize=9)
    
    plt.title(f'Root Cause Diagnosis - Top {top_k} Variables Highlighted')
    plt.xlabel('Sensors / Features')
    plt.ylabel('Normalized Contribution Score')
    plt.grid(axis='y', linestyle=':', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_img)
    print(f">>> 诊断可视化报告已保存至: {output_img}")
    plt.close()

if __name__ == "__main__":
    # 在这里传入你想提取的变量数量，例如 top_k=5
    diagnose_from_csv('high_fault_重复.csv', top_k=8)