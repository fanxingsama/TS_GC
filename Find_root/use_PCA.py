import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import os
from datetime import datetime
from RSPCA import RobustScaler, RNSPCA
from pca_config import Config

def diagnose_from_csv(config):
    file_path = config.TEST_DATA_PATH
    model_path = config.MODEL_SAVE_PATH
    top_k = config.DIAGNOSE_TOP_K
    root_save_dir = config.RESULT_SAVE_DIR
    
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    save_dir = os.path.join(root_save_dir, f"{current_time}")
    
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
    denoiser = pipeline.get('denoiser') 
    scaler = pipeline['scaler']

    # 数据预处理
    df_test = pd.read_csv(file_path)
    feature_names = df_test.columns.tolist()
    
    if denoiser is not None:
        print(">>> 检测到模型包含小波降噪器，正在进行降噪...")
        X_test_preprocessed = denoiser.transform(df_test)
    else:
        print(">>> 模型未使用小波降噪，直接进行标准化...")
        X_test_preprocessed = df_test.values

    X_test_scaled = scaler.transform(X_test_preprocessed)
    
    # 1. 计算系统全局异常监测趋势 (SPE)
    stat_scores, threshold = model.predict_global_anomaly(X_test_scaled)
    plot_global_anomaly(stat_scores, threshold, output_line_path)
    
    # 2. 诊断计算 (变量贡献度)
    diag_res = model.trigger_diagnose(X_test_scaled)
    dcc_scores = diag_res['dcc_norm']
    
    # 异常变量筛选
    sorted_indices = np.argsort(dcc_scores)[::-1]
    save_indices = sorted_indices[:top_k]
    
    # ============== 新增: 绘制筛选出的各个异常变量随时间的分数及阈值 ==============
    # 获取测试集上所有变量的时间序列异常分数
    var_spe_series = model.get_variable_spe_series(X_test_scaled)
    
    # 创建专门存放单变量趋势图的文件夹
    var_trend_dir = os.path.join(save_dir, 'variable_trends')
    os.makedirs(var_trend_dir, exist_ok=True)
    
    for rank, idx in enumerate(save_indices):
        name = feature_names[idx] if idx < len(feature_names) else f"Var_{idx}"
        series = var_spe_series[:, idx]
        
        # 获取该变量对应的异常阈值
        var_thresholds = getattr(model, 'var_SPE_thresholds', None) 
        
        plt.figure(figsize=(12, 4))
        # 加上 zorder=4 确保折线画在背景色带的上方
        plt.plot(series, label=f'{name} Anomaly Score', color='#1f77b4', linewidth=1.5, zorder=4)
        
        if var_thresholds is not None:
            var_thresh = var_thresholds[idx]
            plt.axhline(var_thresh, color='red', linestyle='--', linewidth=2, 
                        label=f'Threshold ({var_thresh:.4f})', zorder=5)
            
            # ======== 新增：为每个变量单独计算并绘制背景颜色分区 ========
            is_anomaly = np.nan_to_num(series) > var_thresh
            diff = np.diff(is_anomaly.astype(int))
            change_points = np.where(diff != 0)[0] + 1
            split_indices = [0] + list(change_points) + [len(series) - 1]
            
            for j in range(len(split_indices) - 1):
                start = split_indices[j]
                end = split_indices[j+1]
                if is_anomaly[start]:
                    plt.axvspan(start, end, color='lightcoral', alpha=0.2, label='Anomaly Zone')
                else:
                    plt.axvspan(start, end, color='lightskyblue', alpha=0.2, label='Normal Zone')
            # ==============================================================
            
        plt.title(f'Rank {rank+1} Anomalous Variable Trend: {name}')
        plt.xlabel('Time Step')
        plt.ylabel('SPE Score')
        
        # 使用字典去重机制，防止图例中出现多个重复的 "Anomaly Zone" / "Normal Zone"
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), loc='upper left')
        
        # 添加网格线让图表更清晰
        plt.grid(True, linestyle=':', alpha=0.4)
        plt.tight_layout()
        
        # 保存图片
        save_path = os.path.join(var_trend_dir, f'rank_{rank+1}_{name.replace("/", "_")}.png')
        plt.savefig(save_path, dpi=150)
        plt.close()
        
    print(f">>> 单个异常变量的时间趋势图已保存至: {var_trend_dir}")
    # ========================================================================
    
    # --- 终端输出 ---
    print("\n" + "="*45)
    print(f"【诊断报告】 提取 Top {top_k} 变量")
    print(f"{'排名':<6} | {'变量索引':<10} | {'变量名称':<20} | {'贡献得分'}")
    print("-"*45)
    for i, idx in enumerate(save_indices):
        name = feature_names[idx] if idx < len(feature_names) else f"Var_{idx}"
        print(f"{i+1:<8} | {idx:<12} | {name:<20} | {dcc_scores[idx]:.4f}")
    print("="*45)

    # --- 写入日志文件 ---
    log_lines = [
        "==================================================",
        "            【诊断日志报告 / Diagnostic Log】",
        "==================================================",
        f"生成时间: {current_time}",
        f"数据文件: {file_path}",
        f"模型文件: {model_path}\n",
        "【模型与诊断参数】",
        f"- 小波降噪启用 (Wavelet Denoising): {'是' if denoiser else '否'}",
        f"- 保留主成分数 (n_components): {model.n_components}",
        f"- 稀疏度控制 (sparsity_k): {model.sparsity_k}",
        f"- 高斯核带宽 (sigma): {model.sigma}",
        f"- 显著性水平 (alpha): {model.alpha}",
        f"- 滑动窗口启用 (use_window): {model.use_window}"
    ]
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
    plot_results(dcc_scores, save_indices, feature_names, output_img_path, top_k)

def plot_global_anomaly(stat_scores, threshold, output_img):
    plt.figure(figsize=(12, 6))
    time_steps = np.arange(len(stat_scores))
    plt.plot(time_steps, stat_scores, color='#1f77b4', linewidth=1.5, label='Global Anomaly Score', zorder=4)
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
            
    plt.title('System Global Anomaly Monitoring with Background Zoning', fontsize=14)
    plt.xlabel('Time Step', fontsize=12)
    plt.ylabel('Anomaly Score', fontsize=12)
    
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='upper left')
    plt.tick_params(axis='both', direction='in', which='both', top=False, right=False)
    plt.grid(True, linestyle=':', alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_img, dpi=300)
    print(f">>> 全局异常趋势背景分区图已保存至: {output_img}")
    plt.close()

def plot_results(dcc_scores, highlight_indices, feature_names, output_img, top_k):
    n_vars = len(dcc_scores)
    plt.figure(figsize=(12, 6))
    x_idx = np.arange(n_vars)
    colors = ['crimson' if i in highlight_indices else 'lightgray' for i in range(n_vars)]
    plt.bar(x_idx, dcc_scores, color=colors, alpha=0.9)
    if n_vars <= 30:
        plt.xticks(x_idx, feature_names, rotation=45, ha='right', fontsize=9)
    plt.title(f'Root Cause Diagnosis  - Top {top_k} Variables Highlighted', fontsize=14)
    plt.xlabel('Sensors / Features', fontsize=12)
    plt.ylabel('Normalized Contribution Score', fontsize=12)
    plt.grid(axis='y', linestyle=':', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_img, dpi=300)
    print(f">>> 诊断可视化报告已保存至: {output_img}")
    plt.close()

if __name__ == "__main__":
    diagnose_from_csv(Config)