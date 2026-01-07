import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import os
from datetime import datetime
from RSPCA import WaveletDenoiser, RobustScaler, RNSPCA

def diagnose_from_csv(file_path, model_path='pca_pipeline.pkl', top_k=5):
    """
    读取异常 CSV 并进行诊断，手动指定提取前 top_k 个异常变量
    top_k: 指定要提取并保存的潜在异常变量数量
    """
    # ==========================================
    # 0. 文件夹与路径设置
    # ==========================================
    root_save_dir = 'PCA_saved'
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    save_dir = os.path.join(root_save_dir, current_time)
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f">>> 已创建结果保存文件夹: {save_dir}")
        
    output_img_path = os.path.join(save_dir, 'diagnostic_report.png')
    output_csv_path = os.path.join(save_dir, 'potential_var.csv')

    # ==========================================
    # 1. 加载模型
    # ==========================================
    try:
        with open(model_path, 'rb') as f:
            pipeline = pickle.load(f)
    except FileNotFoundError:
        print("错误：未找到模型文件，请先运行 train_PAC.py")
        return

    denoiser = pipeline['denoiser']
    scaler = pipeline['scaler']
    model = pipeline['model']
    
    print(f"当前模型配置: Window Size = {model.window_size}")

    # ==========================================
    # 2. 读取测试数据
    # ==========================================
    print(f">>> 正在读取待检测数据: {file_path}")
    df_test = pd.read_csv(file_path)
    feature_names = df_test.columns.tolist()

    if len(df_test) < model.window_size:
        print(f"错误：测试数据行数 ({len(df_test)}) 少于模型窗口大小 ({model.window_size})")
        return

    # ==========================================
    # 3. 数据预处理与计算
    # ==========================================
    X_test_denoised = denoiser.transform(df_test)
    X_test_scaled = scaler.transform(X_test_denoised)
    
    # 触发诊断计算
    diag_res = model.trigger_diagnose(X_test_scaled)
    dcc_scores = diag_res['dcc_norm']
    
    # ==========================================
    # 4. 筛选异常变量 (完全由 top_k 控制)
    # ==========================================
    # 按贡献度从大到小排序，取前 top_k 个索引
    sorted_indices = np.argsort(dcc_scores)[::-1]
    save_indices = sorted_indices[:top_k]
    
    print("\n" + "="*45)
    print(f"【诊断报告】 提取贡献度最高的 Top {top_k} 变量")
    print(f"{'排名':<6} | {'变量索引':<10} | {'变量名称':<20} | {'贡献得分'}")
    print("-"*45)
    
    for i, idx in enumerate(save_indices):
        name = feature_names[idx] if idx < len(feature_names) else f"Var_{idx}"
        print(f"{i+1:<8} | {idx:<12} | {name:<20} | {dcc_scores[idx]:.4f}")
    print("="*45)

    # ==========================================
    # 5. 提取并保存 CSV
    # ==========================================
    # 从原始数据 df_test 中提取这些列
    df_potential = df_test.iloc[:, save_indices]
    
    # 保存 CSV
    df_potential.to_csv(output_csv_path, index=False)
    print(f">>> 潜在异常变量数据已分离并保存至: {output_csv_path}")
    print(f"    包含变量索引: {save_indices}")

    # ==========================================
    # 6. 可视化保存
    # ==========================================
    # 传入 save_indices 以便在图中高亮显示这些变量
    plot_results(dcc_scores, save_indices, feature_names, output_img_path, top_k)


def plot_results(dcc_scores, highlight_indices, feature_names, output_img, top_k):
    n_vars = len(dcc_scores)
    plt.figure(figsize=(12, 6))
    x_idx = np.arange(n_vars)
    
    # 颜色逻辑：top_k 的变量显示为红色，其余显示为灰色
    colors = ['crimson' if i in highlight_indices else 'lightgray' for i in range(n_vars)]
    
    plt.bar(x_idx, dcc_scores, color=colors, alpha=0.9)
    
    # 不再绘制 LMVT 阈值线，保持画面清爽
    
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