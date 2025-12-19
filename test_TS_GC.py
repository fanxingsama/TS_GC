from pathlib import Path
import joblib
from scipy.special import softmax
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import pandas as pd
from config import *
from model.TS_GC import MutiTS_GC
import os
from visual.plot_causal_link import save_causal_links
from util import get_latest_run_id

rcParams['font.family'] = 'SimHei'
rcParams['axes.unicode_minus'] = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载模型
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

    model.load_state_dict(torch.load(model_weight_path, map_location=device, weights_only=True))
    model.eval() 
    return model

# 绘制时序序列预测结果与实际值的对比图
def prediction_compare(model, X_data, Y_data, series_num, save_path=None, series_name=None, points_to_plot=300, max_samples=300):
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
    
    series_prediction_dir = os.path.join(save_path, "series_prediction")
    os.makedirs(series_prediction_dir, exist_ok=True)
    
    time_axis = np.arange(points_to_plot)
    
    for series_idx in range(series_num):
        # 创建单个序列的图
        fig, ax = plt.subplots(1, 1, figsize=(15, 6))
        
        ax.plot(time_axis, actuals_np[:points_to_plot, output_step_idx, series_idx], 
                label='实际值', color='blue', linewidth=2)
        ax.plot(time_axis, predictions_np[:points_to_plot, output_step_idx, series_idx], 
                label='预测值', color='red', linestyle='--', linewidth=2)
        
        ax.set_title(f'序列 {series_idx} 预测对比\nMAE: {series_mae[series_idx]:.4f} | MSE: {series_mse[series_idx]:.4f}', 
                    fontsize=14)
        ax.set_xlabel('时间步', fontsize=12)
        ax.set_ylabel('值', fontsize=12)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # 确定文件名
        if series_name is not None and len(series_name) > series_idx:
            filename = f"{series_name[series_idx]}.png"
        else:
            filename = f"series_{series_idx}.png"
        
        # 保存图片
        save_plot_path = os.path.join(series_prediction_dir, filename)
        plt.savefig(save_plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    print(f"所有预测对比图已保存至文件夹: {series_prediction_dir}")
    print(f"总体 MAE: {mae:.4f}, MSE: {mse:.4f}")
    
    return mae, mse

# 计算每个目标序列的主导因果滞后热力图
def plot_first_layer_weights_heatmap(model, target_series_idx, save_path=None, use_abs_weights=True):
    if not (0 <= target_series_idx < model.series_num):
        print(f"错误: target_series_idx ({target_series_idx}) 超出范围 [0, {model.series_num-1}]")
        return

    W_target = model.networks[target_series_idx].first_conv.weight.detach().cpu()
    
    P_total = model.series_num # 数据集中的总序列数
    # P_input_to_submodel 是当前子模型第一层卷积实际接收的输入通道数
    P_input_to_submodel = W_target.shape[1] 
    kernel_size = W_target.shape[2]

    if use_abs_weights:
        processed_weights = torch.abs(W_target)
        plot_title_suffix = " (绝对值聚合)"
        cmap = 'Reds'
    else:
        processed_weights = W_target
        plot_title_suffix = " (原始值聚合)"
        cmap = 'coolwarm'

    aggregated_weights = torch.sum(processed_weights, dim=0).numpy() # Shape: (P_input_to_submodel, kernel_size)

    plt.figure(figsize=(max(8, kernel_size * 0.6), max(6, P_input_to_submodel * 0.4)))
    plt.imshow(aggregated_weights, aspect='auto', cmap=cmap, interpolation='nearest')
    plt.colorbar(label='聚合权重强度')
    plt.title(f'目标序列 {target_series_idx} 的第一个卷积层权重热力图{plot_title_suffix}')
    plt.xlabel('滞后 (Lag)')
    
    y_tick_actual_indices = []
    y_axis_label_description = ""

    if P_input_to_submodel == P_total: # 未使用mask来屏蔽目标序列
        y_tick_actual_indices = list(range(P_total))
        y_axis_label_description = '输入序列索引 (原始数据集中的索引)'
    elif P_input_to_submodel == P_total - 1: # 使用mask来屏蔽目标序列
        current_original_idx = 0
        for _ in range(P_input_to_submodel):
            while current_original_idx == target_series_idx:
                current_original_idx += 1
            y_tick_actual_indices.append(current_original_idx)
            current_original_idx += 1
        y_axis_label_description = f'输入序列索引 (原始索引, 目标 {target_series_idx} 已屏蔽)'
    else:
        # Fallback or error for unexpected P_input_to_submodel
        y_tick_actual_indices = list(range(P_input_to_submodel))
        y_axis_label_description = '输入序列索引 (相对)'


    plt.yticks(np.arange(P_input_to_submodel), labels=y_tick_actual_indices)
    plt.ylabel(y_axis_label_description)
    plt.xticks(np.arange(kernel_size), labels=np.arange(kernel_size))

    ax = plt.gca()
    ax.set_xticks(np.arange(kernel_size + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(P_input_to_submodel + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="grey", linestyle='-', linewidth=0.5)
    ax.tick_params(which="minor", size=0)
    plt.tight_layout()

    if save_path: # 检查save_path是否提供
        plt.savefig(save_path, dpi=300)
        print(f"权重热力图已保存至: {save_path}")
    else:
        plt.show() # 如果没有提供保存路径，则显示图像
    plt.close()

# 绘制三个格兰杰因果图的对比：真实、预测、约束后
def plot_gc_triple_compare(pred_csv_path, constrained_csv_path, series_num, save_path, true_csv_path=None, show_weights=True):
    """
    Args:
        pred_csv_path: 原始预测的GC矩阵CSV路径
        constrained_csv_path: 约束后的GC矩阵CSV路径
        series_num: 序列数量
        save_path: 保存路径
        true_csv_path: 真实GC矩阵CSV路径 (可选)
        show_weights: 是否显示权重
    """
    
    def load_gc_matrix(csv_path, series_num, has_weights=True):
        """加载GC矩阵"""
        df = pd.read_csv(csv_path, header=None)
        binary_matrix = np.zeros((series_num, series_num), dtype=int)
        weight_matrix = np.zeros((series_num, series_num))
        
        for _, row in df.iterrows():
            cause = int(row.iloc[0])
            effect = int(row.iloc[1])
            if 0 <= cause < series_num and 0 <= effect < series_num:
                binary_matrix[effect, cause] = 1
                if has_weights and row.shape[0] >= 3:
                    weight_matrix[effect, cause] = float(row.iloc[2])
        
        return binary_matrix, weight_matrix
    
    # 加载矩阵
    GC_pred_binary, GC_pred_weights = load_gc_matrix(pred_csv_path, series_num, has_weights=True)
    GC_constrained_binary, GC_constrained_weights = load_gc_matrix(constrained_csv_path, series_num, has_weights=True)
    
    # 根据是否有真实值决定子图数量和布局
    if true_csv_path is not None:
        GC_true, _ = load_gc_matrix(true_csv_path, series_num, has_weights=False)
        fig, axarr = plt.subplots(1, 3, figsize=(24, 8))
        has_ground_truth = True
    else:
        GC_true = None
        fig, axarr = plt.subplots(1, 2, figsize=(16, 8))
        has_ground_truth = False
    
    plot_idx = 0
    
    # 绘制真实格兰杰因果矩阵（如果提供）
    if has_ground_truth:
        axarr[plot_idx].imshow(GC_true, cmap='Blues', aspect='auto')
        axarr[plot_idx].set_title('真实格兰杰因果矩阵\n(GC Ground Truth)', fontsize=12)
        axarr[plot_idx].set_ylabel('受影响的序列 (Effect series)')
        axarr[plot_idx].set_xlabel('原因序列 (Causal series)')
        axarr[plot_idx].set_xticks(np.arange(series_num))
        axarr[plot_idx].set_yticks(np.arange(series_num))
        axarr[plot_idx].set_xticklabels(np.arange(series_num))
        axarr[plot_idx].set_yticklabels(np.arange(series_num))
        plot_idx += 1
    
    # 绘制原始预测的格兰杰因果矩阵
    if show_weights and np.any(GC_pred_weights > 0):
        img_pred = axarr[plot_idx].imshow(GC_pred_weights, cmap='Blues', aspect='auto',
                                          extent=(-0.5, series_num-0.5, series_num-0.5, -0.5))
        axarr[plot_idx].set_title('模型预测的GC矩阵\n(Model Prediction)', fontsize=12)
        fig.colorbar(img_pred, ax=axarr[plot_idx], orientation='vertical', fraction=0.046, pad=0.04)
    else:
        axarr[plot_idx].imshow(GC_pred_binary, cmap='Blues', aspect='auto')
        axarr[plot_idx].set_title('模型预测的GC矩阵\n(Model Prediction)', fontsize=12)
    
    axarr[plot_idx].set_ylabel('受影响的序列 (Effect series)')
    axarr[plot_idx].set_xlabel('原因序列 (Causal series)')
    axarr[plot_idx].set_xticks(np.arange(series_num))
    axarr[plot_idx].set_yticks(np.arange(series_num))
    axarr[plot_idx].set_xticklabels(np.arange(series_num))
    axarr[plot_idx].set_yticklabels(np.arange(series_num))
    pred_plot_idx = plot_idx
    plot_idx += 1
    
    # 绘制约束后的格兰杰因果矩阵
    if show_weights and np.any(GC_constrained_weights > 0):
        img_constrained = axarr[plot_idx].imshow(GC_constrained_weights, cmap='Blues', aspect='auto',
                                                 extent=(-0.5, series_num-0.5, series_num-0.5, -0.5))
        axarr[plot_idx].set_title('MAD约束后的GC矩阵\n(MAD Constrained)', fontsize=12)
        fig.colorbar(img_constrained, ax=axarr[plot_idx], orientation='vertical', fraction=0.046, pad=0.04)
    else:
        axarr[plot_idx].imshow(GC_constrained_binary, cmap='Blues', aspect='auto')
        axarr[plot_idx].set_title('MAD约束后的GC矩阵\n(MAD Constrained)', fontsize=12)
    
    axarr[plot_idx].set_ylabel('受影响的序列 (Effect series)')
    axarr[plot_idx].set_xlabel('原因序列 (Causal series)')
    axarr[plot_idx].set_xticks(np.arange(series_num))
    axarr[plot_idx].set_yticks(np.arange(series_num))
    axarr[plot_idx].set_xticklabels(np.arange(series_num))
    axarr[plot_idx].set_yticklabels(np.arange(series_num))
    constrained_plot_idx = plot_idx
    
    # 在预测矩阵上添加权重文本和错误标记（只有提供真实值时才标记错误）
    matrix_configs = [(GC_pred_binary, GC_pred_weights, pred_plot_idx), 
                      (GC_constrained_binary, GC_constrained_weights, constrained_plot_idx)]
    
    for binary_mat, weight_mat, ax_idx in matrix_configs:
        ax = axarr[ax_idx]
        
        for i in range(series_num):
            for j in range(series_num):
                # 显示权重文本
                if show_weights and binary_mat[i, j] == 1 and weight_mat[i, j] > 0:
                    weight_val = weight_mat[i, j]
                    text = f"{weight_val:.3f}"
                    ax.text(j, i, text, ha="center", va="center",
                           color='black', fontsize=8, weight='bold')
                
                # 标记与真实值不同的位置（仅在有真实值时）
                if has_ground_truth and GC_true[i, j] != binary_mat[i, j]:
                    rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor='none',
                                       edgecolor='red', linewidth=2)
                    ax.add_patch(rect)
    
    # 计算并显示评估指标（仅在有真实值时）
    if has_ground_truth:
        def calculate_metrics(true_mat, pred_mat):
            TP = np.sum((true_mat == 1) & (pred_mat == 1))
            TN = np.sum((true_mat == 0) & (pred_mat == 0))
            FP = np.sum((true_mat == 0) & (pred_mat == 1))
            FN = np.sum((true_mat == 1) & (pred_mat == 0))
            
            accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
            precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            return accuracy, precision, recall, f1_score
        
        # 计算原始预测和约束后的指标
        acc_pred, prec_pred, rec_pred, f1_pred = calculate_metrics(GC_true, GC_pred_binary)
        acc_const, prec_const, rec_const, f1_const = calculate_metrics(GC_true, GC_constrained_binary)
        
        # 在图上添加指标信息
        metrics_text_pred = f'Acc: {acc_pred:.3f}, Prec: {prec_pred:.3f}\nRec: {rec_pred:.3f}, F1: {f1_pred:.3f}'
        metrics_text_const = f'Acc: {acc_const:.3f}, Prec: {prec_const:.3f}\nRec: {rec_const:.3f}, F1: {f1_const:.3f}'
        
        axarr[pred_plot_idx].text(0.02, 0.98, metrics_text_pred, transform=axarr[pred_plot_idx].transAxes,
                                  verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                                  fontsize=9)
        axarr[constrained_plot_idx].text(0.02, 0.98, metrics_text_const, transform=axarr[constrained_plot_idx].transAxes,
                                         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8),
                                         fontsize=9)

    
    fig.tight_layout(pad=3.0)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"格兰杰因果矩阵对比图已保存至: {save_path}")

# 将模型的格兰杰因果矩阵保存为CSV格式
def save_gc_matrix_to_csv(model, save_path, threshold=0.0, ignore_self_causality=True, is_softmax = True):
    # 获取模型估计的GC权重范数矩阵
    GC_est_norms_tensor = model.GC(threshold=False, ignore_kernel=True) 
    GC_est_norms = GC_est_norms_tensor.detach().cpu().numpy()
    if GC_est_norms.size == 0 or np.all(GC_est_norms == 0):
        print("模型未估计出格兰杰因果关系，GC_est_norms为空或全为0。")
        return False
    
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
    return True

# 使用中位数与绝对中位差(MAD)方法约束格兰杰因果矩阵
def constrain_gc_matrix_with_mad(gc_csv_path, output_csv_path, series_num, mad_multiplier=1.0):

    gc_df = pd.read_csv(gc_csv_path, header=None)
    
    # 重建GC权重矩阵
    GC_weights = np.zeros((series_num, series_num))
    for _, row in gc_df.iterrows():
        cause = int(row.iloc[0])  # source
        effect = int(row.iloc[1])  # target
        weight = float(row.iloc[2])  # strength
        if 0 <= cause < series_num and 0 <= effect < series_num:
            GC_weights[effect, cause] = weight
    
    # 对每个目标序列分别应用MAD约束
    constrained_results = []
    
    for target_idx in range(series_num):
        # 获取当前目标序列的所有输入权重（排除自身）
        target_weights = []
        for source_idx in range(series_num):
            if source_idx != target_idx:  # 排除自因果
                weight = GC_weights[target_idx, source_idx]
                if weight > 0:  # 只考虑非零权重
                    target_weights.append(weight)
        
        if len(target_weights) == 0:
            continue
            
        # 计算中位数和绝对中位差
        median_weight = np.median(target_weights)
        mad = np.median(np.abs(np.array(target_weights) - median_weight))
        
        # 计算约束阈值
        threshold = median_weight + mad_multiplier * mad
        
        print(f"目标序列 {target_idx}: 中位数={median_weight:.4f}, MAD={mad:.4f}, 阈值={threshold:.4f}")
        
        # 筛选满足约束条件的因果关系
        for source_idx in range(series_num):
            if source_idx != target_idx:  # 排除自因果
                weight = GC_weights[target_idx, source_idx]
                if weight >= threshold:
                    constrained_results.append({
                        'source': source_idx,
                        'target': target_idx,
                        'strength': weight
                    })
    
    # 保存约束后的结果
    if constrained_results:
        constrained_df = pd.DataFrame(constrained_results)
        constrained_df = constrained_df.sort_values(['source', 'target'], ascending=[True, True])
        constrained_df.to_csv(output_csv_path, index=False, encoding='utf-8', header=False)
        print(f"约束后的格兰杰因果矩阵已保存至: {output_csv_path}")
    else:
        # 如果没有满足条件的因果关系，创建空文件
        pd.DataFrame(columns=['source', 'target', 'strength']).to_csv(
            output_csv_path, index=False, encoding='utf-8', header=False)
        print(f"约束后无满足条件的因果关系，已创建空文件: {output_csv_path}")
    
    return len(constrained_results)

def constrain_with_std_dev(gc_csv_path, output_csv_path, series_num, std_multiplier=1.5):
    gc_df = pd.read_csv(gc_csv_path, header=None)
    GC_weights = np.zeros((series_num, series_num))
    for _, row in gc_df.iterrows():
        GC_weights[int(row.iloc[1]), int(row.iloc[0])] = float(row.iloc[2])

    constrained_results = []
    for target_idx in range(series_num):
        # 获取当前目标序列的所有非零输入权重（排除自身）
        target_weights = []
        source_indices = []
        for source_idx in range(series_num):
            if source_idx != target_idx and GC_weights[target_idx, source_idx] > 0:
                target_weights.append(GC_weights[target_idx, source_idx])
                source_indices.append(source_idx)
        
        if not target_weights:
            continue

        weights_arr = np.array(target_weights)
        mean_weight = np.mean(weights_arr)
        std_weight = np.std(weights_arr)
        
        # 计算阈值
        threshold = mean_weight + std_multiplier * std_weight
        print(f"目标序列 {target_idx}: 均值={mean_weight:.4f}, 标准差={std_weight:.4f}, 阈值={threshold:.4f}")

        # 筛选
        for i, weight in enumerate(target_weights):
            if weight >= threshold:
                constrained_results.append({
                    'source': source_indices[i],
                    'target': target_idx,
                    'strength': weight
                })

    if constrained_results:
        constrained_df = pd.DataFrame(constrained_results)
        constrained_df = constrained_df.sort_values(['source', 'target'])
        constrained_df.to_csv(output_csv_path, index=False, header=False)
        print(f"均值标准差 (multiplier={std_multiplier}) 约束后的GC矩阵已保存至: {output_csv_path}")
    else:
        pd.DataFrame(columns=['source', 'target', 'strength']).to_csv(
            output_csv_path, index=False, header=False)
        print("均值标准差约束后无满足条件的因果关系，已创建空文件。")

def main(model_path):
    gc_predict_path = model_path / "GC_matrix.csv"
    gc_constrain_path = model_path / "GC_matrix_constrained.csv"
    GC_predict = model_path / "GC_predict.png"
    causal_links_path = model_path / "causal_links.png"
    
    model = load_model(model_path, DEVICE)
    model.eval()
    
    res = save_gc_matrix_to_csv(model, gc_predict_path, threshold=0.0, ignore_self_causality=True, is_softmax=False)
    
    if res:
        # constrain_gc_matrix_with_mad(gc_predict_path, gc_constrain_path, model.series_num, mad_multiplier=1.0)
        constrain_with_std_dev(gc_predict_path, gc_constrain_path, model.series_num, std_multiplier=0.3)
        plot_gc_triple_compare(gc_predict_path,gc_constrain_path,  model.series_num, GC_predict, true_csv_path=GC_PATH, show_weights=True)
        mae, mse = prediction_compare(model, X_DATA, Y_DATA, model.series_num, save_path=model_path, series_name=SERIES_NAME)
        plot_first_layer_weights_heatmap(model, 1, save_path=model_path / "first_layer_weights_heatmap.png", use_abs_weights=True)
        save_causal_links(csv_path = gc_constrain_path, img_save_path = causal_links_path)
        

if __name__ == "__main__":
    run_id = get_latest_run_id()
    # run_id = '12-18_22-14-04'
    model_path = Path('saved') / run_id
    main(model_path)