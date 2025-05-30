from pathlib import Path
import re
import joblib
from scipy.special import softmax
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import pandas as pd
from config import *

from logger.logger import get_logger, setup_logging
from model.TS_GC import MutiTS_GC

rcParams['font.family'] = 'SimHei'
rcParams['axes.unicode_minus'] = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_path, device):
    config_path = model_path / "model_config.pkl"
    model_weight_path = model_path / "best_model.pth"

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

    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval() 
    return model

# 绘制因果图对比
def plot_gc_compare(pred_csv_path, true_csv_path, series_num, save_path, show_weights=True):
    # 读取真实的格兰杰因果关系（无表头）
    true_df = pd.read_csv(true_csv_path, header=None)
    GC_true = np.zeros((series_num, series_num), dtype=int)
    
    for index, row in true_df.iterrows():
        cause = int(row.iloc[0])  # 第一列是因
        effect = int(row.iloc[1])  # 第二列是果
        if 0 <= cause < series_num and 0 <= effect < series_num:
            GC_true[effect, cause] = 1
    
    # 读取预测的格兰杰因果关系（无表头）
    pred_df = pd.read_csv(pred_csv_path, header=None)
    GC_est_binary = np.zeros((series_num, series_num), dtype=int)
    GC_est_norms = np.zeros((series_num, series_num))
    
    # 检查预测数据的列数
    has_weights = pred_df.shape[1] >= 3
    
    for index, row in pred_df.iterrows():
        cause = int(row.iloc[0])  # 第一列是因
        effect = int(row.iloc[1])  # 第二列是果
        if 0 <= cause < series_num and 0 <= effect < series_num:
            GC_est_binary[effect, cause] = 1
            
            # 如果有权重列，读取权重
            if has_weights:
                GC_est_norms[effect, cause] = float(row.iloc[2])

    # 绘制对比图
    fig, axarr = plt.subplots(1, 2, figsize=(18, 8)) 
    
    # 绘制真实格兰杰因果矩阵
    axarr[0].imshow(GC_true, cmap='Blues', aspect='auto')
    axarr[0].set_title('真实格兰杰因果矩阵 (GC actual)')
    axarr[0].set_ylabel('受影响的序列 (Effect series)')
    axarr[0].set_xlabel('原因序列 (Causal series)')
    axarr[0].set_xticks(np.arange(series_num))
    axarr[0].set_yticks(np.arange(series_num))
    axarr[0].set_xticklabels(np.arange(series_num))
    axarr[0].set_yticklabels(np.arange(series_num))

    # 绘制预测的格兰杰因果矩阵
    if has_weights and show_weights:
        # 如果有权重且要显示权重，使用权重作为颜色深浅
        img_est = axarr[1].imshow(GC_est_norms, cmap='Blues', aspect='auto', 
                                  extent=(-0.5, series_num-0.5, series_num-0.5, -0.5))
        axarr[1].set_title('模型估计的GC ')
        fig.colorbar(img_est, ax=axarr[1], orientation='vertical', fraction=0.046, pad=0.04)
    else:
        # 否则只显示二进制关系
        axarr[1].imshow(GC_est_binary, cmap='Blues', aspect='auto')
        axarr[1].set_title('模型估计的GC')
    
    axarr[1].set_ylabel('受影响的序列 (Effect series)')
    axarr[1].set_xlabel('原因序列 (Causal series)')
    axarr[1].set_xticks(np.arange(series_num))
    axarr[1].set_yticks(np.arange(series_num))
    axarr[1].set_xticklabels(np.arange(series_num))
    axarr[1].set_yticklabels(np.arange(series_num))

    # 在预测矩阵上添加文本信息
    for i in range(series_num): 
        for j in range(series_num):
            # 如果有权重且要显示权重且存在因果关系
            if has_weights and show_weights and GC_est_binary[i, j] == 1:
                weight_val = GC_est_norms[i, j]
                text = f"{weight_val:.2f}"
                
                axarr[1].text(j, i, text, ha="center", va="center", 
                             color='black', fontsize=8)
            
            # 标记预测错误的位置
            if GC_true[i, j] != GC_est_binary[i, j]:
                rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor='none', 
                                   edgecolor='red', linewidth=2)
                axarr[1].add_patch(rect)
    
    fig.tight_layout(pad=3.0) 
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"格兰杰因果矩阵对比图已保存至: {save_path}")

# 混淆矩阵
def model_eval(pred_csv_path, true_csv_path, series_num):
    GC_true = np.zeros((series_num, series_num), dtype=int)
    true_df = pd.read_csv(true_csv_path, header=None)
    for _, row in true_df.iterrows():
        cause = int(row.iloc[0])
        effect = int(row.iloc[1])
        if 0 <= cause < series_num and 0 <= effect < series_num:
            GC_true[effect, cause] = 1

    GC_pred = np.zeros((series_num, series_num), dtype=int)
    pred_df = pd.read_csv(pred_csv_path, header=None)
    for _, row in pred_df.iterrows():
        cause = int(row.iloc[0])
        effect = int(row.iloc[1])
        if 0 <= cause < series_num and 0 <= effect < series_num:
            GC_pred[effect, cause] = 1

    TP = 0
    TN = 0
    FP = 0
    FN = 0

    for i in range(series_num):
        for j in range(series_num):
            true_val = GC_true[i, j]
            pred_val = GC_pred[i, j]

            if true_val == 1 and pred_val == 1:
                TP += 1
            elif true_val == 0 and pred_val == 0:
                TN += 1
            elif true_val == 0 and pred_val == 1:
                FP += 1
            elif true_val == 1 and pred_val == 0:
                FN += 1
    accuracy = 0.0
    precision = 0.0
    recall = 0.0
    f1_score = 0.0

    total_samples = TP + TN + FP + FN
    accuracy = (TP + TN) / total_samples
    if TP + FP > 0:
        precision = TP / (TP + FP)
    if TP + FN > 0:
        recall = TP / (TP + FN)
    if precision + recall > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
    
    return accuracy, precision, recall, f1_score

# 绘制预测结果与实际值的对比图
def prediction_compare(model, X_data, Y_data, series_num, num_series_to_plot=5, points_to_plot=300, max_samples=300):
    num_samples_to_process = min(X_data.shape[0], max_samples)

    X_data_subset = X_data[:num_samples_to_process].to(DEVICE)
    Y_data_subset = Y_data[:num_samples_to_process] 

    with torch.no_grad():
        model_preds = model(X_data_subset) 
        predictions_np = model_preds.cpu().numpy()
        actuals_np = Y_data_subset.cpu().numpy()

    # 计算MAE和MSE
    mae = np.mean(np.abs(predictions_np - actuals_np))
    mse = np.mean((predictions_np - actuals_np) ** 2)
    
    # 计算每个序列的MAE和MSE
    series_mae = np.mean(np.abs(predictions_np - actuals_np), axis=(0, 1))  # 对样本和时间步求均值
    series_mse = np.mean((predictions_np - actuals_np) ** 2, axis=(0, 1))

    num_available_points = predictions_np.shape[0]
    points_to_plot = min(points_to_plot, num_available_points)

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
        ax.set_title(f'序列 {series_idx} - 预测对比 - MAE: {series_mae[series_idx]:.4f}, MSE: {series_mse[series_idx]:.4f}')
        ax.set_ylabel('值')
        ax.legend()
        ax.grid(True)
    
    axes[-1].set_xlabel('时间步')
    fig.tight_layout()
    save_plot_path = model_path / f"prediction_compare.png"
    plt.savefig(save_plot_path)
    plt.close(fig)
    print(f"预测对比图已保存至: {save_plot_path}")
    
    return mae, mse

# 计算每个目标序列的主导因果滞后
def plot_first_layer_weights_heatmap(model, target_series_idx, save_path=None, use_abs_weights=True):
    """
    为指定目标序列的第一个卷积层权重绘制热力图。
    热力图展示了每个输入序列在不同滞后上的聚合影响强度。

    参数:
        model (MutiTS_GC): 训练好的 MutiTS_GC 模型。
        target_series_idx (int): 要可视化其权重的目标序列的索引。
        save_path (str, optional): 图像保存路径。如果为 None，则显示图像。
        use_abs_weights (bool): 是否使用权重的绝对值进行可视化。默认为 True。
    """
    if not (0 <= target_series_idx < model.series_num):
        print(f"错误: target_series_idx ({target_series_idx}) 超出范围 [0, {model.series_num-1}]")
        return

    # 1. 获取权重
    # W_target 的形状: (feature_dim, series_num_input, kernel_size)
    W_target = model.networks[target_series_idx].first_conv.weight.detach().cpu()

    # 提取维度信息
    num_input_series = W_target.shape[1] # 这应该等于 model.series_num
    kernel_size = W_target.shape[2]

    # 2. 处理权重 (取绝对值或不取)
    if use_abs_weights:
        processed_weights = torch.abs(W_target)
        plot_title_suffix = " (绝对值聚合)"
        cmap = 'Reds' # 使用红色系，值越大颜色越深
    else:
        processed_weights = W_target
        plot_title_suffix = " (原始值聚合)"
        cmap = 'coolwarm' # 使用冷暖色系，可以区分正负值

    # 3. 聚合特征维度
    # aggregated_weights 的形状: (num_input_series, kernel_size)
    aggregated_weights = torch.sum(processed_weights, dim=0).numpy()

    # 4. 绘制热力图
    plt.figure(figsize=(max(8, kernel_size * 0.6), max(6, num_input_series * 0.4)))
    plt.imshow(aggregated_weights, aspect='auto', cmap=cmap, interpolation='nearest')

    plt.colorbar(label='聚合权重强度')
    plt.title(f'目标序列 {target_series_idx} 的第一个卷积层权重热力图{plot_title_suffix}')
    plt.xlabel('滞后 (Lag)')
    plt.ylabel('输入序列索引 (Causative Series Index)')

    # 设置刻度
    plt.xticks(np.arange(kernel_size), labels=np.arange(kernel_size))
    plt.yticks(np.arange(num_input_series), labels=np.arange(num_input_series))

    # 添加网格线，使单元格更清晰
    ax = plt.gca()
    ax.set_xticks(np.arange(kernel_size + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(num_input_series + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="grey", linestyle='-', linewidth=0.5)
    ax.tick_params(which="minor", size=0)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"权重热力图已保存至: {save_path}")

def get_latest_run_id():
    base_path = Path('saved')
    if not base_path.exists():
        return None
    
    # 获取所有符合格式的目录名
    pattern = re.compile(r'^\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$')
    timestamps = [d.name for d in base_path.iterdir() 
                 if d.is_dir() and pattern.match(d.name)]
    
    return max(timestamps) if timestamps else None

# 将模型的格兰杰因果矩阵保存为CSV格式
def save_gc_matrix_to_csv(model, save_path, threshold=0.0, ignore_self_causality=True, is_softmax = True):
    # 获取模型估计的GC权重范数矩阵
    GC_est_norms_tensor = model.GC(threshold=False, ignore_kernel=True) 
    GC_est_norms = GC_est_norms_tensor.detach().cpu().numpy()
    
    series_num = GC_est_norms.shape[0]
    
    if is_softmax: # 是否应用softmax
        if ignore_self_causality:
            # 将对角线元素设为极小值，避免影响softmax计算
            GC_est_norms_masked = GC_est_norms.copy()
            np.fill_diagonal(GC_est_norms_masked, -np.inf)
            
            # 按行应用softmax
            GC_est_norms_softmax = softmax(GC_est_norms_masked, axis=1)
            
            # 将对角线元素重新设为0
            np.fill_diagonal(GC_est_norms_softmax, 0)
        else:
            GC_est_norms_softmax = softmax(GC_est_norms, axis=1)
        
        GC_est_norms = GC_est_norms_softmax
    
    # 创建结果列表
    results = []
    
    for i in range(series_num):  # i是受影响的序列（果）
        for j in range(series_num):  # j是原因序列（因）
            # 如果屏蔽自因果关系且i==j，则跳过
            if ignore_self_causality and i == j:
                continue
                
            weight = round(GC_est_norms[i, j], 3) # 保留三位小数
            # 只保存权重大于阈值的因果关系
            if weight > threshold:
                results.append({
                    'source': j,
                    'target': i, 
                    'strength': weight
                })
    
    # 转换为DataFrame并保存
    df = pd.DataFrame(results)
    df = df.sort_values(['source', 'target'], ascending=[True, True])  # 先按因升序，再按果降序
    df.to_csv(save_path, index=False, encoding='utf-8', header=False)
    
    print(f"格兰杰因果矩阵已保存至: {save_path}")

def main(model_path):
    gc_predict_path = model_path / "GC_matrix.csv"
    
    model = load_model(model_path, DEVICE)
    model.eval()
    
    save_gc_matrix_to_csv(model, model_path / "GC_matrix.csv", threshold=0.0, ignore_self_causality=True, is_softmax=False)
    plot_gc_compare(gc_predict_path, GC_PATH, SERIES_NUM, model_path / "GC_predict.png")
    mae, mse = prediction_compare(model, X_DATA, Y_DATA, SERIES_NUM)
    
    accuracy, precision, recall, f1_score = model_eval(gc_predict_path, GC_PATH, SERIES_NUM)
    log_file_path = model_path / "model_test_info.log"
    with open(log_file_path, 'w', encoding='utf-8') as log_file:
        log_file.write(f"模型评估结果:  准确率: {accuracy:.2f}, 精确率: {precision:.2f}, 召回率: {recall:.2f}, F1分数: {f1_score:.2f}\n, MAE: {mae:.2f}, MSE: {mse:.2f}\n")
    plot_first_layer_weights_heatmap(model, 1, save_path=model_path / "first_layer_weights_heatmap.png", use_abs_weights=True)

if __name__ == "__main__":
    run_id = get_latest_run_id()
    model_path = Path('saved') / run_id
    main(model_path)