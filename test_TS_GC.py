from pathlib import Path
import joblib
from scipy.special import softmax
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import pandas as pd
from config import * # 确保这里面包含了 SERIES_NAME
from model.TS_GC import TS_GC
import os
from util.plot_causal_link import save_causal_links
from util.util import get_latest_run_id, normalize_name

rcParams['font.family'] = 'SimHei'
rcParams['axes.unicode_minus'] = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载模型
def load_model(model_path, device):
    config_path = model_path / "model_config.pkl"
    model_weight_path = model_path / "best_model.pth"

    saved_config = joblib.load(config_path)
    
    model = TS_GC(
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
def prediction_compare(model, X_data, Y_data, series_names, save_path=None, points_to_plot=300, max_samples=300):
    series_num = len(series_names)
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
    series_mae = np.mean(np.abs(predictions_np - actuals_np), axis=(0, 1))
    series_mse = np.mean((predictions_np - actuals_np) ** 2, axis=(0, 1))

    num_available_points = predictions_np.shape[0]
    points_to_plot = min(points_to_plot, num_available_points)

    output_step_idx = 0 
    
    series_prediction_dir = os.path.join(save_path, "series_prediction")
    os.makedirs(series_prediction_dir, exist_ok=True)
    
    time_axis = np.arange(points_to_plot)
    
    for series_idx in range(series_num):
        current_name = series_names[series_idx]
        
        # 创建单个序列的图
        fig, ax = plt.subplots(1, 1, figsize=(15, 6))
        
        ax.plot(time_axis, actuals_np[:points_to_plot, output_step_idx, series_idx], 
                label='实际值', color='blue', linewidth=2)
        ax.plot(time_axis, predictions_np[:points_to_plot, output_step_idx, series_idx], 
                label='预测值', color='red', linestyle='--', linewidth=2)
        
        # 标题直接使用名称
        ax.set_title(f'{current_name} 预测对比\nMAE: {series_mae[series_idx]:.4f} | MSE: {series_mse[series_idx]:.4f}', 
                    fontsize=14)
        ax.set_xlabel('时间步', fontsize=12)
        ax.set_ylabel('值', fontsize=12)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # 文件名直接使用名称 (清理非法字符以防万一)
        safe_filename = "".join([c for c in current_name if c.isalnum() or c in (' ', '_', '-')]).strip()
        filename = f"{safe_filename}.png"
        
        # 保存图片
        save_plot_path = os.path.join(series_prediction_dir, filename)
        plt.savefig(save_plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    print(f"所有预测对比图已保存至文件夹: {series_prediction_dir}")
    print(f"总体 MAE: {mae:.4f}, MSE: {mse:.4f}")
    
    return mae, mse

# 绘制三个格兰杰因果图的对比：真实、预测、约束后
# 绘制三个格兰杰因果图的对比：真实、预测、约束后
def plot_gc_triple_compare(pred_csv_path, constrained_csv_path, series_names, save_path, true_csv_path=None, show_weights=True):
    """
    基于 test_test.py 的绘图逻辑修改，增加了对序列名称的支持。
    Args:
        pred_csv_path: 预测矩阵CSV路径
        constrained_csv_path: 约束后矩阵CSV路径
        series_names: 序列名称列表 (list of strings)
        save_path: 图片保存路径
        true_csv_path: 真实矩阵CSV路径
        show_weights: 是否显示权重
    """
    series_num = len(series_names)
    
    # 建立 名称 -> 索引 的映射字典 (用于解析CSV中的名称)
    # 使用 normalize_name 确保匹配准确，假设 normalize_name 已定义在 util 中
    name_to_idx = {normalize_name(name): idx for idx, name in enumerate(series_names)}
    
    def load_gc_matrix(csv_path, series_num, has_weights=True):
        """加载GC矩阵，支持名称解析"""
        # 强制读取前两列为字符串，防止数字名称被当做数字处理
        try:
            df = pd.read_csv(csv_path, header=None, dtype={0: str, 1: str})
        except pd.errors.EmptyDataError:
            print(f"警告: 文件 {csv_path} 为空，返回零矩阵。")
            return np.zeros((series_num, series_num), dtype=int), np.zeros((series_num, series_num))

        binary_matrix = np.zeros((series_num, series_num), dtype=int)
        weight_matrix = np.zeros((series_num, series_num))
        
        for _, row in df.iterrows():
            cause_str = normalize_name(str(row.iloc[0]))
            effect_str = normalize_name(str(row.iloc[1]))
            
            try:
                # 1. 尝试通过名称查找索引
                if cause_str in name_to_idx:
                    cause = name_to_idx[cause_str]
                else:
                    # 2. 如果名称匹配失败，尝试直接作为索引数字解析 (兼容旧版CSV)
                    cause = int(float(cause_str))
                
                if effect_str in name_to_idx:
                    effect = name_to_idx[effect_str]
                else:
                    effect = int(float(effect_str))
            except (ValueError, KeyError):
                # 如果既不是已知名称也不是数字，跳过
                continue

            # 填充矩阵 (行=Effect, 列=Cause)
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
    
    # --- 1. 绘制真实格兰杰因果矩阵（如果提供）---
    if has_ground_truth:
        axarr[plot_idx].imshow(GC_true, cmap='Blues', aspect='auto')
        axarr[plot_idx].set_title('真实格兰杰因果矩阵\n(GC Ground Truth)', fontsize=12)
        axarr[plot_idx].set_ylabel('受影响的序列 (Effect series)')
        axarr[plot_idx].set_xlabel('原因序列 (Causal series)')
        axarr[plot_idx].set_xticks(np.arange(series_num))
        axarr[plot_idx].set_yticks(np.arange(series_num))
        # 使用名称作为标签
        axarr[plot_idx].set_xticklabels(series_names, rotation=45, ha='right')
        axarr[plot_idx].set_yticklabels(series_names)
        plot_idx += 1
    
    # --- 2. 绘制原始预测的格兰杰因果矩阵 ---
    if show_weights and np.any(GC_pred_weights > 0):
        # 使用 extent 确保坐标对齐
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
    # 使用名称作为标签
    axarr[plot_idx].set_xticklabels(series_names, rotation=45, ha='right')
    axarr[plot_idx].set_yticklabels(series_names)
    pred_plot_idx = plot_idx
    plot_idx += 1
    
    # --- 3. 绘制约束后的格兰杰因果矩阵 ---
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
    # 使用名称作为标签
    axarr[plot_idx].set_xticklabels(series_names, rotation=45, ha='right')
    axarr[plot_idx].set_yticklabels(series_names)
    constrained_plot_idx = plot_idx
    
    # --- 在预测矩阵上添加权重文本和错误标记 ---
    matrix_configs = [(GC_pred_binary, GC_pred_weights, pred_plot_idx), 
                      (GC_constrained_binary, GC_constrained_weights, constrained_plot_idx)]
    
    for binary_mat, weight_mat, ax_idx in matrix_configs:
        ax = axarr[ax_idx]
        
        for i in range(series_num): # Row (Effect)
            for j in range(series_num): # Col (Cause)
                # 显示权重文本
                # 只有当 binary_mat 为 1 (存在因果) 且 weight > 0 时才显示
                if show_weights and binary_mat[i, j] == 1 and weight_mat[i, j] > 0:
                    weight_val = weight_mat[i, j]
                    # 使用与 test_test.py 相同的格式化
                    text = f"{weight_val:.3f}"
                    
                    # 智能颜色调整：如果背景很深（权重很大），用白色字体
                    max_val = weight_mat.max() if weight_mat.max() > 0 else 1.0
                    text_color = 'white' if weight_val > (max_val * 0.5) else 'black'
                    
                    ax.text(j, i, text, ha="center", va="center",
                           color=text_color, fontsize=8, weight='bold')
                
                # 标记与真实值不同的位置（仅在有真实值时）
                if has_ground_truth and GC_true[i, j] != binary_mat[i, j]:
                    rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor='none',
                                       edgecolor='red', linewidth=2)
                    ax.add_patch(rect)
    
    # --- 计算并显示评估指标 ---
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
        
        acc_pred, prec_pred, rec_pred, f1_pred = calculate_metrics(GC_true, GC_pred_binary)
        acc_const, prec_const, rec_const, f1_const = calculate_metrics(GC_true, GC_constrained_binary)
        
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
def save_gc_matrix_to_csv(model, save_path, series_names, threshold=0.0, ignore_self_causality=True, is_softmax=True):
    # 获取模型估计的GC权重范数矩阵
    GC_est_norms_tensor = model.GC(threshold=False, ignore_kernel=True) 
    GC_est_norms = GC_est_norms_tensor.detach().cpu().numpy()
    if GC_est_norms.size == 0 or np.all(GC_est_norms == 0):
        print("模型未估计出格兰杰因果关系，GC_est_norms为空或全为0。")
        return False
    
    series_num = len(series_names)
    assert GC_est_norms.shape[0] == series_num, f"series_names 长度 ({series_num}) 与模型序列数 ({GC_est_norms.shape[0]}) 不匹配"

    if is_softmax:
        if ignore_self_causality:
            GC_est_norms_masked = GC_est_norms.copy()
            np.fill_diagonal(GC_est_norms_masked, -np.inf)
            GC_est_norms_softmax = softmax(GC_est_norms_masked, axis=1)
            np.fill_diagonal(GC_est_norms_softmax, 0)
        else:
            GC_est_norms_softmax = softmax(GC_est_norms, axis=1)
        GC_est_norms = GC_est_norms_softmax
    
    # 创建结果列表
    results = []
    
    for i in range(series_num):  # i是受影响的序列（果）Target
        for j in range(series_num):  # j是原因序列（因）Source
            if ignore_self_causality and i == j:
                continue
                
            weight = round(GC_est_norms[i, j], 3)
            if weight > threshold:
                # 强制使用名称
                source_name = series_names[j]
                target_name = series_names[i]
                
                results.append({
                    'source': source_name,
                    'target': target_name, 
                    'strength': weight
                })
    
    df = pd.DataFrame(results)
    # 按照名称排序可能比较乱，但为了一致性，我们还是按 Source, Target 排序
    if not df.empty:
        df = df.sort_values(['source', 'target'], ascending=[True, True])
        df.to_csv(save_path, index=False, encoding='utf-8', header=False)
    else:
        # 如果是空的，创建一个空文件
        Path(save_path).touch()
    
    print(f"格兰杰因果矩阵(带名称)已保存至: {save_path}")
    return True

# 基于均值+标准差约束GC矩阵 (处理带名称的CSV)
def constrain_with_hard_threshold(gc_csv_path, output_csv_path, series_names, threshold=0.004):
    series_num = len(series_names)
    # 1. 创建映射字典
    name_to_idx = {normalize_name(name): idx for idx, name in enumerate(series_names)}
    
    # 2. 读取 CSV
    try:
        gc_df = pd.read_csv(gc_csv_path, header=None, dtype={0: str, 1: str})
    except pd.errors.EmptyDataError:
        print("CSV 文件为空，无法进行约束。")
        return

    GC_weights = np.zeros((series_num, series_num))
    
    # 3. 遍历解析并填入矩阵 (保持原有的名称解析逻辑)
    for _, row in gc_df.iterrows():
        source_name = normalize_name(row.iloc[0])
        target_name = normalize_name(row.iloc[1])
        weight = float(row.iloc[2])
        
        if source_name in name_to_idx and target_name in name_to_idx:
            s_idx = name_to_idx[source_name]
            t_idx = name_to_idx[target_name]
            GC_weights[t_idx, s_idx] = weight

    constrained_results = []
    
    # 4. 直接根据阈值进行过滤
    for target_idx in range(series_num):
        for source_idx in range(series_num):
            # 跳过自回归，且权重必须大于等于阈值
            if source_idx != target_idx:
                weight = GC_weights[target_idx, source_idx]
                
                # === 核心修改逻辑在这里 ===
                if weight >= threshold: 
                    constrained_results.append({
                        'source': series_names[source_idx],
                        'target': series_names[target_idx],
                        'strength': weight
                    })

    # 5. 保存结果
    if constrained_results:
        constrained_df = pd.DataFrame(constrained_results)
        constrained_df = constrained_df.sort_values(['source', 'target'])
        constrained_df.to_csv(output_csv_path, index=False, header=False)
        print(f"硬约束后的GC矩阵(阈值>={threshold})已保存至: {output_csv_path}")
    else:
        print(f"无满足条件(>={threshold})的因果关系")

def main(model_path):
    # 检查 SERIES_NAME 是否存在
    if 'SERIES_NAME' not in globals() or not SERIES_NAME:
        raise ValueError("请在 config.py 中定义 SERIES_NAME，且不能为空。")
        
    gc_predict_path = model_path / "GC_matrix.csv"
    gc_constrain_path = model_path / "GC_matrix_constrained.csv"
    GC_predict_img_path = model_path / "GC_predict.png"
    causal_links_path = model_path / "causal_links.png"
    
    model = load_model(model_path, DEVICE)
    res = save_gc_matrix_to_csv(model, gc_predict_path, SERIES_NAME, threshold=0.0, ignore_self_causality=True, is_softmax=False)
    
    if res:
        constrain_with_hard_threshold(gc_predict_path, gc_constrain_path, SERIES_NAME, threshold=0.004)
        plot_gc_triple_compare(gc_predict_path, gc_constrain_path, SERIES_NAME, GC_predict_img_path, true_csv_path=GC_PATH, show_weights=True)
        prediction_compare(model, X_DATA, Y_DATA, SERIES_NAME, save_path=model_path)
        save_causal_links(csv_path=gc_constrain_path, img_save_path=causal_links_path, series_names=SERIES_NAME)
        
if __name__ == "__main__":
    run_id = get_latest_run_id()
    # run_id = '01-05_11-23-56'
    model_path = Path('saved') / run_id
    main(model_path)