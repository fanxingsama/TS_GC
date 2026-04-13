import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import os
from datetime import datetime
from RSPCA import RobustScaler, RNSPCA
from pca_config import Config
from matplotlib import rcParams

# 字体与显示配置
rcParams['font.family'] = 'SimHei'
rcParams['axes.unicode_minus'] = False

# 全局字体大小配置（可自由调节）
rcParams['axes.titlesize'] = 24       # 图表标题字体大小
rcParams['axes.labelsize'] = 24       # 坐标轴标签字体大小（如"异常贡献度"）
rcParams['xtick.labelsize'] = 24      # X轴刻度标签字体大小（柱状图底部的变量名）
rcParams['ytick.labelsize'] = 24      # Y轴刻度标签字体大小
rcParams['legend.fontsize'] = 24      # 图例字体大小

# 系统总体异常分数图专用配置（可单独调节）
GLOBAL_ANOMALY_XLABEL_SIZE = 28       # "时间步"标签字体大小
GLOBAL_ANOMALY_YLABEL_SIZE = 28       # "异常分数"标签字体大小
GLOBAL_ANOMALY_LEGEND_SIZE = 22       # 图例字体大小
GLOBAL_ANOMALY_TICK_SIZE = 28         # 刻度数字字体大小


def diagnose_from_csv(config):
    file_path = config.TEST_DATA_PATH
    model_path = config.MODEL_SAVE_PATH
    top_k = config.DIAGNOSE_TOP_K
    root_save_dir = config.RESULT_SAVE_DIR
    contrib_mode = getattr(config, 'CONTRIB_MODE', 'spe')
    
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    save_dir = os.path.join(root_save_dir, f"{current_time}")
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f">>> 已创建结果保存文件夹: {save_dir}")
        
    output_img_path = os.path.join(save_dir, 'diagnostic_report.pdf')
    output_line_path = os.path.join(save_dir, 'global_anomaly_trend.pdf')
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
    X_raw = df_test.values.astype(float)
    
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
    print(f">>> 贡献度计算模式: {contrib_mode}")
    if contrib_mode == 'combined':
        normal_median = scaler.median
        diag_res = model.trigger_diagnose(X_test_scaled, mode='combined',
                                           X_raw_fault=X_raw,
                                           X_raw_normal_median=normal_median)
    else:
        diag_res = model.trigger_diagnose(X_test_scaled, mode='spe')
    
    dcc_scores = diag_res['dcc_norm']
    
    # 异常变量筛选
    sorted_indices = np.argsort(dcc_scores)[::-1]
    save_indices = sorted_indices[:top_k]
    
    # ============== 绘制筛选出的各个异常变量随时间的分数及阈值 ==============
    var_spe_series = model.get_variable_spe_series(X_test_scaled)
    
    var_trend_dir = os.path.join(save_dir, 'variable_trends')
    os.makedirs(var_trend_dir, exist_ok=True)
    
    for rank, idx in enumerate(save_indices):
        name = feature_names[idx] if idx < len(feature_names) else f"Var_{idx}"
        series = var_spe_series[:, idx]
        
        var_thresholds = getattr(model, 'var_SPE_thresholds', None) 
        
        plt.figure(figsize=(8, 4))
        plt.plot(series, label=f'变量异常分数', color='#1f77b4', linewidth=1.5, zorder=4)
        
        if var_thresholds is not None:
            var_thresh = var_thresholds[idx]
            plt.axhline(var_thresh, color='red', linestyle='--', linewidth=2, 
                        label=f'异常阈值 ({var_thresh:.4f})', zorder=5)
            
            is_anomaly = np.nan_to_num(series) > var_thresh
            diff = np.diff(is_anomaly.astype(int))
            change_points = np.where(diff != 0)[0] + 1
            split_indices = [0] + list(change_points) + [len(series) - 1]
            
            for j in range(len(split_indices) - 1):
                start = split_indices[j]
                end = split_indices[j+1]
                if is_anomaly[start]:
                    plt.axvspan(start, end, color='lightcoral', alpha=0.2)
                else:
                    plt.axvspan(start, end, color='lightskyblue', alpha=0.2)
        
        plt.title(f'{name}')
        plt.xlabel('时间步')
        plt.ylabel('异常分数')
        plt.tick_params(axis='both', which='both', length=0) # 去掉刻度线
        
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), loc='upper left')

        plt.grid(True, linestyle=':', alpha=0.4)
        plt.tight_layout()
        
        save_path = os.path.join(var_trend_dir, f'rank_{rank+1}_{name.replace("/", "_")}.pdf')
        plt.savefig(save_path, dpi=150)
        plt.close()
        
    print(f">>> 单个异常变量的时间趋势图已保存至: {var_trend_dir}")
    
    # --- 终端输出 ---
    print("\n" + "="*45)
    print(f"【诊断报告】 提取 Top {top_k} 变量 (模式: {contrib_mode})")
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
        f"- 滑动窗口启用 (use_window): {model.use_window}",
        f"- 贡献度模式 (contrib_mode): {contrib_mode}",
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
    fig, ax = plt.subplots(figsize=(14, 5))
    time_axis = np.arange(len(stat_scores))
    
    is_anomaly = stat_scores > threshold
    
    i = 0
    while i < len(stat_scores):
        j = i
        while j < len(stat_scores) and is_anomaly[j] == is_anomaly[i]:
            j += 1
        color = '#FDDEDE' if is_anomaly[i] else '#DDEEFF'
        ax.axvspan(i, j, alpha=0.5, color=color)
        i = j
    
    ax.plot(time_axis, stat_scores, color='steelblue', linewidth=1.2, label='系统状态分数')
    ax.axhline(y=threshold, color='red', linestyle='--', linewidth=1.5, 
               label=f'异常阈值（{threshold:.2f}）')
    
    ax.set_xlabel('时间步', fontsize=GLOBAL_ANOMALY_XLABEL_SIZE)
    ax.set_ylabel('异常分数', fontsize=GLOBAL_ANOMALY_YLABEL_SIZE)
    ax.legend(loc='upper left', fontsize=GLOBAL_ANOMALY_LEGEND_SIZE)
    ax.tick_params(axis='both', which='both', length=0, labelsize=GLOBAL_ANOMALY_TICK_SIZE)
    ax.grid(axis='y', linestyle=':', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_img, dpi=300, bbox_inches='tight')
    plt.close()
    print(f">>> 全局异常趋势图已保存至: {output_img}")


def plot_results(dcc_scores, highlight_indices, feature_names, output_img, top_k):
    n_vars = len(dcc_scores)
    plt.figure(figsize=(20, 10))  # 增加宽度以拉大变量名间隔
    x_idx = np.arange(n_vars)
    colors = ['crimson' if i in highlight_indices else 'lightgray' for i in range(n_vars)]
    plt.bar(x_idx, dcc_scores, color=colors, alpha=0.9)
    
    tick_labels = [feature_names[i] if i in highlight_indices else '' for i in range(n_vars)]
    plt.xticks(x_idx, tick_labels, rotation=60, ha='right')  # 增加旋转角度以避免重叠
    
    plt.tick_params(axis='both', which='both', length=0) # 去掉刻度线

    plt.ylabel('异常贡献度')
    plt.grid(axis='y', linestyle=':', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_img, dpi=300)
    print(f">>> 诊断可视化报告已保存至: {output_img}")
    plt.close()

if __name__ == "__main__":
    diagnose_from_csv(Config)