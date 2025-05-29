from pathlib import Path
import re
import joblib
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import pandas as pd
import os

try:
    from TS_GC import MutiTS_GC 
except ImportError:
    from model.TS_GC import MutiTS_GC

rcParams['font.family'] = 'SimHei'
rcParams['axes.unicode_minus'] = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_path, device):
    """加载模型的函数"""
    config_path = model_path / "model_config_ista.pkl"
    if not config_path.exists():
        config_path = model_path / "model_config.pkl"

    if not config_path.exists():
        raise FileNotFoundError(f"Model config not found at {model_path / 'model_config_ista.pkl'} or {model_path / 'model_config.pkl'}")

    saved_config = joblib.load(config_path)
    
    model = MutiTS_GC(
        input_window=saved_config['input_window'],
        output_window=saved_config['output_window'],
        series_num=saved_config['series_num'],
        feature_dim=saved_config['feature_dim'],
        temporal_layers=saved_config['temporal_layers'],
        kernel_size=saved_config['kernel_size'],
        dropout=saved_config['dropout'],
        device=device 
    ).to(device)
    
    model_weight_path = model_path / "best_model_ista.pth"
    if not model_weight_path.exists():
        model_weight_path = model_path / "best_model.pth"

    if not model_weight_path.exists():
        raise FileNotFoundError(f"Model weights not found at {model_path / 'best_model_ista.pth'} or {model_path / 'best_model.pth'}")

    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval() 
    return model, saved_config

def plot_gc_comparison_figure(model, csv_path, series_num, save_path):
    """绘制因果图的函数"""
    # 加载真实GC矩阵
    try:
        df = pd.read_csv(csv_path, header=None) 
    except FileNotFoundError:
        print(f"真实格兰杰因果矩阵文件未找到: {csv_path}")
        return
    except pd.errors.EmptyDataError:
        print(f"真实格兰杰因果矩阵文件为空: {csv_path}")
        return

    GC_true = np.zeros((series_num, series_num), dtype=int)
    for index, row in df.iterrows():
        try:
            cause = int(row[0]) 
            effect = int(row[1])
            if 0 <= cause < series_num and 0 <= effect < series_num:
                GC_true[effect, cause] = 1 
            else:
                print(f"警告: 文件 {csv_path} 中发现无效的序列索引: 因={cause}, 果={effect} (序列总数={series_num})")
        except (ValueError, TypeError) as e:
            print(f"警告: 解析文件 {csv_path} 行 {index+1} 时出错: {row.values}. 错误: {e}")
            continue

    # 获取模型估计的GC矩阵
    GC_est_binary_tensor = model.GC(threshold=True, ignore_kernel=True, weight_threshold=0.0)
    GC_est_binary = GC_est_binary_tensor.detach().cpu().numpy().astype(int)
    
    GC_est_norms_tensor = model.GC(threshold=False, ignore_kernel=True) 
    GC_est_norms = GC_est_norms_tensor.detach().cpu().numpy()

    # 验证矩阵形状
    if not (GC_true.shape == (series_num, series_num) and \
            GC_est_binary.shape == (series_num, series_num) and \
            GC_est_norms.shape == (series_num, series_num)):
        print(f"错误: 矩阵形状不一致或与期望的 ({series_num}, {series_num}) 不符。")
        print(f"GC_true shape: {GC_true.shape}, GC_est_binary shape: {GC_est_binary.shape}, GC_est_norms shape: {GC_est_norms.shape}")
        return

    # 绘制对比图
    fig, axarr = plt.subplots(1, 2, figsize=(18, 8)) 
    
    axarr[0].imshow(GC_true, cmap='Blues', aspect='auto')
    axarr[0].set_title('真实格兰杰因果矩阵 (GC actual)')
    axarr[0].set_ylabel('受影响的序列 (Effect series)')
    axarr[0].set_xlabel('原因序列 (Causal series)')
    axarr[0].set_xticks(np.arange(series_num))
    axarr[0].set_yticks(np.arange(series_num))
    axarr[0].set_xticklabels(np.arange(series_num))
    axarr[0].set_yticklabels(np.arange(series_num))

    img_est_norms = axarr[1].imshow(GC_est_norms, cmap='Blues', aspect='auto', 
                                    extent=(-0.5, series_num-0.5, series_num-0.5, -0.5))
    axarr[1].set_title('模型估计的GC (权重范数和差异)')
    axarr[1].set_ylabel('受影响的序列 (Effect series)')
    axarr[1].set_xlabel('原因序列 (Causal series)')
    axarr[1].set_xticks(np.arange(series_num))
    axarr[1].set_yticks(np.arange(series_num))
    axarr[1].set_xticklabels(np.arange(series_num))
    axarr[1].set_yticklabels(np.arange(series_num))
    
    fig.colorbar(img_est_norms, ax=axarr[1], orientation='vertical', fraction=0.046, pad=0.04)

    for i in range(series_num): 
        for j in range(series_num): 
            norm_val = GC_est_norms[i, j]
            text_color = "white" if norm_val > (GC_est_norms.max() / 2) else "black"
            axarr[1].text(j, i, f"{norm_val:.2f}", ha="center", va="center", color=text_color, fontsize=8)
            
            if GC_true[i, j] != GC_est_binary[i, j]:
                rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor='none', edgecolor='red', linewidth=1.5)
                axarr[1].add_patch(rect)
    
    fig.tight_layout(pad=3.0) 
    plt.savefig(save_path)
    plt.close()
    print(f"格兰杰因果矩阵对比图已保存至: {save_path}")

def plot_prediction_comparison_figure(model, data_path, model_path, series_num, input_window, output_window, 
                                    num_series_to_plot=5, points_to_plot=300, max_samples=300):
    """绘制预测曲线的函数"""
    print(f"\n开始加载数据进行预测对比，路径: {data_path}")
    if not data_path.exists():
        print(f"数据文件未找到: {data_path}。跳过预测绘图。")
        return

    prediction_input_df = pd.read_csv(data_path)
    
    if prediction_input_df.empty:
        print(f"加载的数据文件 {data_path} 为空，跳过预测绘图。")
        return

    if prediction_input_df.shape[1] < series_num:
        print(f"警告: 数据文件列数 ({prediction_input_df.shape[1]}) 少于模型期望的序列数 ({series_num})。跳过预测绘图。")
        return
    elif prediction_input_df.shape[1] > series_num:
        print(f"警告: 数据文件列数 ({prediction_input_df.shape[1]}) 多于模型期望的序列数 ({series_num})。将仅使用前 {series_num} 列。")
        prediction_input_df = prediction_input_df.iloc[:, :series_num]

    # 创建序列数据
    all_series_cols = prediction_input_df.columns.tolist()
    data_np = prediction_input_df[all_series_cols].values.astype(np.float32) 
    num_timesteps, num_series = data_np.shape

    if num_timesteps < input_window + output_window:
        print(f"数据不足以创建序列。需要: {input_window + output_window}, 可用: {num_timesteps}")
        return

    X_list, Y_list = [], []
    for i in range(num_timesteps - input_window - output_window + 1):
        X_list.append(data_np[i : i + input_window, :])
        Y_list.append(data_np[i + input_window : i + input_window + output_window, :])
    
    if not X_list:
        print("从数据创建的序列为空，无法进行预测绘图。")
        return

    X_data = torch.tensor(np.array(X_list), dtype=torch.float32)
    Y_data = torch.tensor(np.array(Y_list), dtype=torch.float32)

    # 获取预测结果
    model.eval()
    num_samples_to_process = min(X_data.shape[0], max_samples)
    if num_samples_to_process == 0:
        print("未能从数据获取预测结果。")
        return

    X_data_subset = X_data[:num_samples_to_process].to(DEVICE)
    Y_data_subset = Y_data[:num_samples_to_process] 

    with torch.no_grad():
        model_preds = model(X_data_subset) 
        predictions_np = model_preds.cpu().numpy()
        actuals_np = Y_data_subset.cpu().numpy()

    if predictions_np.size == 0 or actuals_np.size == 0:
        print("没有预测数据或实际数据可供绘图。")
        return

    # 绘制预测对比图
    num_available_points = predictions_np.shape[0]
    points_to_plot = min(points_to_plot, num_available_points)
    
    if points_to_plot == 0:
        print("没有足够的点来绘制预测图。")
        return

    output_step_idx = 0 
    plot_indices = list(range(min(num_series_to_plot, series_num)))
    
    fig, axes = plt.subplots(len(plot_indices), 1, figsize=(15, 3 * len(plot_indices)), sharex=True)
    if len(plot_indices) == 1: 
        axes = [axes]

    time_axis = np.arange(points_to_plot)

    for i, series_idx in enumerate(plot_indices):
        ax = axes[i]
        ax.plot(time_axis, actuals_np[:points_to_plot, output_step_idx, series_idx], label='实际值', color='blue')
        ax.plot(time_axis, predictions_np[:points_to_plot, output_step_idx, series_idx], label='预测值', color='red', linestyle='--')
        ax.set_title(f'序列 {series_idx} - 预测对比 (前 {points_to_plot} 点, 输出步 {output_step_idx+1})')
        ax.set_ylabel('值')
        ax.legend()
        ax.grid(True)
    
    axes[-1].set_xlabel('时间步')
    fig.tight_layout()
    save_plot_path = model_path / f"prediction_vs_actual_comparison_first_{points_to_plot}_points.png"
    plt.savefig(save_plot_path)
    plt.close(fig)
    print(f"预测对比图已保存至: {save_plot_path}")

def get_latest_run_id(base_dir_name='saved_ista_models'):
    """得到run_id的函数"""
    base_path = Path(base_dir_name)
    if not base_path.exists():
        base_path = Path('saved') 
        if not base_path.exists():
            print(f"模型保存目录 '{base_dir_name}' 和 'saved' 均未找到。")
            return None
    
    pattern_ista = re.compile(r'^\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_ista$')
    pattern_general = re.compile(r'^\d{2}-\d{2}_\d{2}-\d{2}-\d{2}(_.*)?$')

    sub_dirs = [d for d in base_path.iterdir() if d.is_dir()]
    
    ista_dirs = sorted([d.name for d in sub_dirs if pattern_ista.match(d.name)], reverse=True)
    if ista_dirs:
        return ista_dirs[0]

    general_dirs = sorted([d.name for d in sub_dirs if pattern_general.match(d.name)], reverse=True)
    return general_dirs[0] if general_dirs else None

def process_data_and_generate_plots(model_path):
    """处理数据的函数"""
    GROUND_TRUTH_GC_PATH = Path('data/simu_data/granger_causality.csv')
    DATA_FOR_PREDICTION_PATH = Path('data/simu_data/series_data.csv') 
    MAX_PREDICTION_POINTS_TO_PLOT = 300
    NUM_SERIES_TO_PLOT_PREDICTIONS = 5

    # 获取模型路径
    if specified_run_id:
        run_id = specified_run_id
        model_base_dir = 'saved_ista_models' if '_ista' in run_id or Path('saved_ista_models/' + run_id).exists() else 'saved'
        model_path = Path(model_base_dir) / run_id
    else:
        run_id = get_latest_run_id()
        if run_id is None:
            print("未找到已保存的模型。")
            return
        model_base_dir = 'saved_ista_models' if '_ista' in run_id else 'saved'
        model_path = Path(model_base_dir) / run_id

    print(f"尝试从以下路径加载模型: {model_path}")
    
    # 加载模型
    try:
        model, saved_config = load_model(model_path, DEVICE)
    except FileNotFoundError as e:
        print(f"加载模型失败: {e}")
        return
        
    series_num = saved_config['series_num']
    input_window = saved_config['input_window']
    output_window = saved_config['output_window']
    print(f"模型已加载。序列数: {series_num}, 输入窗口: {input_window}, 输出窗口: {output_window}")

    # 1. 格兰杰因果矩阵对比
    if GROUND_TRUTH_GC_PATH.exists():
        plot_gc_comparison_save_path = model_path / "gc_matrix_comparison_with_norms.png"
        plot_gc_comparison_figure(model, GROUND_TRUTH_GC_PATH, series_num, plot_gc_comparison_save_path)
    else:
        print("警告: 无法加载真实格兰杰因果矩阵，跳过GC对比。")

    # 2. 预测值与实际值对比
    plot_prediction_comparison_figure(model, DATA_FOR_PREDICTION_PATH, model_path, series_num, 
                                    input_window, output_window, 
                                    num_series_to_plot=NUM_SERIES_TO_PLOT_PREDICTIONS, 
                                    points_to_plot=MAX_PREDICTION_POINTS_TO_PLOT)

def main(specified_run_id=None):
    process_data_and_generate_plots(specified_run_id)

if __name__ == "__main__":
    run_id = get_latest_run_id()
    model_path = Path('saved') / run_id
    main(model_path)