import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import os
from datetime import datetime
from RSPCA import WaveletDenoiser, RobustScaler, RNSPCA
from pca_config import Config

def diagnose_from_csv(config):
    file_path = config.TEST_DATA_PATH
    model_path = config.MODEL_SAVE_PATH
    top_k = config.DIAGNOSE_TOP_K
    stat_type = config.STAT_TYPE
    root_save_dir = config.RESULT_SAVE_DIR
    
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    save_dir = os.path.join(root_save_dir, f"{current_time}_{stat_type}")
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f">>> 已创建结果保存文件夹: {save_dir}")
        
    output_img_path = os.path.join(save_dir, 'diagnostic_report.png')
    output_line_path = os.path.join(save_dir, 'global_anomaly_trend.png')
    output_csv_path = os.path.join(save_dir, 'potential_var.csv')
    output_log_path = os.path.join(save_dir, 'diagnosis_log.txt')

    # 加载模型
    with open(model_path, 'rb') as f:
        pipeline = pickle.load(f)
    model = pipeline['model']

    # 数据预处理
    df_test = pd.read_csv(file_path)
    feature_names = df_test.columns.tolist()
    denoiser = pipeline['denoiser']
    scaler = pipeline['scaler']
    X_test_denoised = denoiser.transform(df_test)
    X_test_scaled = scaler.transform(X_test_denoised)
    
    # 1. 计算系统全局异常监测趋势
    stat_scores, threshold = model.predict_global_anomaly(X_test_scaled, stat_type=stat_type)
    plot_global_anomaly(stat_scores, threshold, output_line_path, stat_type)
    
    # 2. 诊断计算 (变量贡献度)
    diag_res = model.trigger_diagnose(X_test_scaled, stat_type=stat_type)
    dcc_scores = diag_res['dcc_norm']
    
    # 异常变量筛选
    sorted_indices = np.argsort(dcc_scores)[::-1]
    save_indices = sorted_indices[:top_k]
    
    # --- 终端输出 ---
    print("\n" + "="*45)
    print(f"【诊断报告 - 基于 {stat_type} 标准】 提取 Top {top_k} 变量")
    print(f"{'排名':<6} | {'变量索引':<10} | {'变量名称':<20} | {'贡献得分'}")
    print("-"*45)
    for i, idx in enumerate(save_indices):
        name = feature_names[idx] if idx < len(feature_names) else f"Var_{idx}"
        print(f"{i+1:<8} | {idx:<12} | {name:<20} | {dcc_scores[idx]:.4f}")
    print("="*45)

    # --- 写入日志文件 ---
    log_lines = []
    log_lines.append("==================================================")
    log_lines.append("            【诊断日志报告 / Diagnostic Log】")
    log_lines.append("==================================================")
    log_lines.append(f"生成时间: {current_time}")
    log_lines.append(f"数据文件: {file_path}")
    log_lines.append(f"模型文件: {model_path}\n")
    
    log_lines.append("【模型与诊断参数】")
    log_lines.append(f"- 诊断统计量 (Stat Type): {stat_type}")
    log_lines.append(f"- 保留主成分数 (n_components): {model.n_components}")
    log_lines.append(f"- 稀疏度控制 (sparsity_k): {model.sparsity_k}")
    log_lines.append(f"- 高斯核带宽 (sigma): {model.sigma}")
    log_lines.append(f"- 显著性水平 (alpha): {model.alpha}")
    log_lines.append(f"- 滑动窗口启用 (use_window): {model.use_window}")
    if model.use_window:
        log_lines.append(f"- 滑动窗口大小 (window_size): {model.window_size}")
    log_lines.append(f"- 系统全局异常阈值: {threshold:.4f}\n")
    
    log_lines.append(f"【Top {top_k} 异常贡献测点】")
    log_lines.append(f"{'排名':<4} | {'变量索引':<8} | {'变量名称':<25} | {'贡献得分'}")
    log_lines.append("-" * 60)
    for i, idx in enumerate(save_indices):
        name = feature_names[idx] if idx < len(feature_names) else f"Var_{idx}"
        log_lines.append(f"{i+1:<6} | {idx:<12} | {name:<25} | {dcc_scores[idx]:.6f}")
    
    log_lines.append("==================================================")
    
    with open(output_log_path, 'w', encoding='utf-8') as log_file:
        log_file.write("\n".join(log_lines))
    print(f">>> 详细诊断日志已保存至: {output_log_path}")

    # 保存分离出的潜在异常数据 CSV
    df_potential = df_test.iloc[:, save_indices]
    df_potential.to_csv(output_csv_path, index=False)
    print(f">>> 潜在异常变量数据已分离并保存至: {output_csv_path}")
    
    # 绘图
    plot_results(dcc_scores, save_indices, feature_names, output_img_path, top_k, stat_type)

def plot_global_anomaly(stat_scores, threshold, output_img, stat_type):
    # (此函数代码无需修改，保持原样)
    plt.figure(figsize=(12, 6))
    time_steps = np.arange(len(stat_scores))
    label_name = 'Global Anomaly Score ($S^2$/$T^2$)' if stat_type == 'T2' else 'Global Anomaly Score (SPE)'
    plt.plot(time_steps, stat_scores, color='#1f77b4', linewidth=1.5, label=label_name, zorder=4)
    plt.axhline(y=threshold, color='crimson', linestyle='--', linewidth=2, label=f'Control Limit ({threshold:.2f})', zorder=5)
    is_anomaly = np.nan_to_num(stat_scores) > threshold
    diff = np.diff(is_anomaly.astype(int))
    change_points = np.where(diff != 0)[0] + 1
    split_indices = [0] + list(change_points) + [len(stat_scores) - 1]
    for i in range(len(split_indices) - 1):
        start = split_indices[i]
        end = split_indices[i+1]
        if is_anomaly[start]:
            plt.axvspan(start, end, color='lightcoral', alpha=0.2, label='Anomaly Zone' if i < 2 else "")
        else:
            plt.axvspan(start, end, color='lightskyblue', alpha=0.2, label='Normal Zone' if i < 2 else "")
    plt.title(f'System Global Anomaly Monitoring ({stat_type}) with Background Zoning', fontsize=14)
    plt.xlabel('Time Step', fontsize=12)
    plt.ylabel(f'Anomaly Score ({stat_type})', fontsize=12)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='upper left')
    plt.grid(True, linestyle=':', alpha=0.4)
    exceed_idx = np.where(is_anomaly)[0]
    if len(exceed_idx) > 0:
        plt.scatter(time_steps[exceed_idx], stat_scores[exceed_idx], color='red', s=8, zorder=6)
    plt.tight_layout()
    plt.savefig(output_img, dpi=300)
    print(f">>> 全局异常趋势背景分区图已保存至: {output_img}")
    plt.close()

def plot_results(dcc_scores, highlight_indices, feature_names, output_img, top_k, stat_type):
    # (此函数代码无需修改，保持原样)
    n_vars = len(dcc_scores)
    plt.figure(figsize=(12, 6))
    x_idx = np.arange(n_vars)
    colors = ['crimson' if i in highlight_indices else 'lightgray' for i in range(n_vars)]
    plt.bar(x_idx, dcc_scores, color=colors, alpha=0.9)
    if n_vars <= 30:
        plt.xticks(x_idx, feature_names, rotation=45, ha='right', fontsize=9)
    plt.title(f'Root Cause Diagnosis ({stat_type}) - Top {top_k} Variables Highlighted', fontsize=14)
    plt.xlabel('Sensors / Features', fontsize=12)
    plt.ylabel('Normalized Contribution Score', fontsize=12)
    plt.grid(axis='y', linestyle=':', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_img, dpi=300)
    print(f">>> 诊断可视化报告已保存至: {output_img}")
    plt.close()

if __name__ == "__main__":
    diagnose_from_csv(Config)