from pathlib import Path
import joblib
from scipy.special import softmax
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import pandas as pd
from config import * # 确保这里面包含了 SERIES_NAME
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
def plot_gc_triple_compare(pred_csv_path, constrained_csv_path, series_names, save_path, true_csv_path=None, show_weights=True):
    """
    统一使用 series_names，CSV 读取时自动将名称映射回索引进行矩阵绘制
    """
    series_num = len(series_names)
    
    # 建立 名称 -> 索引 的映射字典
    name_to_idx = {name: idx for idx, name in enumerate(series_names)}
    
    def load_gc_matrix(csv_path, series_names, has_weights=True):
        df = pd.read_csv(csv_path, header=None)
        binary_matrix = np.zeros((series_num, series_num), dtype=int)
        weight_matrix = np.zeros((series_num, series_num))
        
        for _, row in df.iterrows():
            cause_val = row.iloc[0]
            effect_val = row.iloc[1]
            
            # 解析 Cause 和 Effect (此时 CSV 里必须是名字，或者是能转成名字的字符串)
            # 如果 CSV 还是存的旧版数字，这里会报错，强制要求 CSV 是名字版
            try:
                # 尝试直接作为名字查找
                if cause_val in name_to_idx:
                    cause = name_to_idx[cause_val]
                else:
                    # 兼容：如果 CSV 里存的是数字索引字符串
                    cause = int(float(cause_val))
                
                if effect_val in name_to_idx:
                    effect = name_to_idx[effect_val]
                else:
                    effect = int(float(effect_val))
                    
            except (KeyError, ValueError):
                print(f"警告：无法解析行 {row.values} 中的序列名称，请检查 CSV 内容是否与 config.SERIES_NAME 匹配。")
                continue
            
            # 填充矩阵
            if 0 <= cause < series_num and 0 <= effect < series_num:
                binary_matrix[effect, cause] = 1
                if has_weights and row.shape[0] >= 3:
                    weight_matrix[effect, cause] = float(row.iloc[2])
        
        return binary_matrix, weight_matrix
    
    # 加载矩阵
    GC_pred_binary, GC_pred_weights = load_gc_matrix(pred_csv_path, series_names, has_weights=True)
    GC_constrained_binary, GC_constrained_weights = load_gc_matrix(constrained_csv_path, series_names, has_weights=True)
    
    # 布局设置
    if true_csv_path is not None:
        GC_true, _ = load_gc_matrix(true_csv_path, series_names, has_weights=False)
        fig, axarr = plt.subplots(1, 3, figsize=(24, 8))
        has_ground_truth = True
    else:
        GC_true = None
        fig, axarr = plt.subplots(1, 2, figsize=(16, 8))
        has_ground_truth = False
    
    plot_idx = 0
    
    # --- 1. 真实格兰杰因果矩阵 ---
    if has_ground_truth:
        axarr[plot_idx].imshow(GC_true, cmap='Blues', aspect='auto')
        axarr[plot_idx].set_title('真实格兰杰因果矩阵\n(GC Ground Truth)', fontsize=12)
        # 设置轴标签
        axarr[plot_idx].set_xticks(np.arange(series_num))
        axarr[plot_idx].set_yticks(np.arange(series_num))
        axarr[plot_idx].set_xticklabels(series_names, rotation=45, ha='right')
        axarr[plot_idx].set_yticklabels(series_names)
        plot_idx += 1
    
    # --- 2. 模型预测矩阵 ---
    if show_weights and np.any(GC_pred_weights > 0):
        axarr[plot_idx].imshow(GC_pred_weights, cmap='Blues', aspect='auto')
    else:
        axarr[plot_idx].imshow(GC_pred_binary, cmap='Blues', aspect='auto')
    
    axarr[plot_idx].set_title('模型预测的GC矩阵\n(Model Prediction)', fontsize=12)
    axarr[plot_idx].set_xticks(np.arange(series_num))
    axarr[plot_idx].set_yticks(np.arange(series_num))
    axarr[plot_idx].set_xticklabels(series_names, rotation=45, ha='right')
    axarr[plot_idx].set_yticklabels(series_names)
    pred_plot_idx = plot_idx
    plot_idx += 1
    
    # --- 3. 约束后的矩阵 ---
    if show_weights and np.any(GC_constrained_weights > 0):
        axarr[plot_idx].imshow(GC_constrained_weights, cmap='Blues', aspect='auto')
    else:
        axarr[plot_idx].imshow(GC_constrained_binary, cmap='Blues', aspect='auto')
        
    axarr[plot_idx].set_title('MAD约束后的GC矩阵\n(MAD Constrained)', fontsize=12)
    axarr[plot_idx].set_xticks(np.arange(series_num))
    axarr[plot_idx].set_yticks(np.arange(series_num))
    axarr[plot_idx].set_xticklabels(series_names, rotation=45, ha='right')
    axarr[plot_idx].set_yticklabels(series_names)
    constrained_plot_idx = plot_idx
    
    # 标记错误 (仅在有 Ground Truth 时)
    matrix_configs = [(GC_pred_binary, pred_plot_idx), 
                      (GC_constrained_binary, constrained_plot_idx)]
    
    for binary_mat, ax_idx in matrix_configs:
        ax = axarr[ax_idx]
        if has_ground_truth:
            for i in range(series_num):
                for j in range(series_num):
                    if GC_true[i, j] != binary_mat[i, j]:
                        rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor='none',
                                           edgecolor='red', linewidth=2)
                        ax.add_patch(rect)
    
    # 计算并显示评估指标
    if has_ground_truth:
        def calculate_metrics(true_mat, pred_mat):
            TP = np.sum((true_mat == 1) & (pred_mat == 1))
            TN = np.sum((true_mat == 0) & (pred_mat == 0))
            FP = np.sum((true_mat == 0) & (pred_mat == 1))
            FN = np.sum((true_mat == 1) & (pred_mat == 0))
            
            precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            accuracy = (TP + TN) / (TP + TN + FP + FN)
            return accuracy, precision, recall, f1_score
        
        acc_pred, prec_pred, rec_pred, f1_pred = calculate_metrics(GC_true, GC_pred_binary)
        acc_const, prec_const, rec_const, f1_const = calculate_metrics(GC_true, GC_constrained_binary)
        
        metrics_text_pred = f'Acc: {acc_pred:.3f}\nPrec: {prec_pred:.3f}\nRec: {rec_pred:.3f}\nF1: {f1_pred:.3f}'
        metrics_text_const = f'Acc: {acc_const:.3f}\nPrec: {prec_const:.3f}\nRec: {rec_const:.3f}\nF1: {f1_const:.3f}'
        
        axarr[pred_plot_idx].text(0.02, 0.98, metrics_text_pred, transform=axarr[pred_plot_idx].transAxes,
                                  verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8), fontsize=9)
        axarr[constrained_plot_idx].text(0.02, 0.98, metrics_text_const, transform=axarr[constrained_plot_idx].transAxes,
                                         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8), fontsize=9)

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
def constrain_with_std_dev(gc_csv_path, output_csv_path, series_names, std_multiplier=1.5):
    series_num = len(series_names)
    name_to_idx = {name: idx for idx, name in enumerate(series_names)}
    
    gc_df = pd.read_csv(gc_csv_path, header=None)
    GC_weights = np.zeros((series_num, series_num))
    
    # 解析 CSV 到矩阵
    for _, row in gc_df.iterrows():
        source_name = row.iloc[0]
        target_name = row.iloc[1]
        weight = float(row.iloc[2])
        
        if source_name in name_to_idx and target_name in name_to_idx:
            s_idx = name_to_idx[source_name]
            t_idx = name_to_idx[target_name]
            GC_weights[t_idx, s_idx] = weight
        else:
            print(f"跳过未知序列名称: {source_name} -> {target_name}")

    constrained_results = []
    
    # 按列（Target）进行阈值过滤
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
        
        threshold = mean_weight + std_multiplier * std_weight
        # print(f"目标 {series_names[target_idx]}: 阈值={threshold:.4f}")

        # 筛选并保存结果（存回名字）
        for k, weight in enumerate(target_weights):
            if weight >= threshold:
                src_idx = source_indices[k]
                constrained_results.append({
                    'source': series_names[src_idx], # 存名字
                    'target': series_names[target_idx], # 存名字
                    'strength': weight
                })

    if constrained_results:
        constrained_df = pd.DataFrame(constrained_results)
        constrained_df = constrained_df.sort_values(['source', 'target'])
        constrained_df.to_csv(output_csv_path, index=False, header=False)
        print(f"约束后的GC矩阵(带名称)已保存至: {output_csv_path}")
    else:
        pd.DataFrame(columns=['source', 'target', 'strength']).to_csv(
            output_csv_path, index=False, header=False)
        print("无满足条件的因果关系，已创建空文件。")

def main(model_path):
    # 检查 SERIES_NAME 是否存在
    if 'SERIES_NAME' not in globals() or not SERIES_NAME:
        raise ValueError("请在 config.py 中定义 SERIES_NAME，且不能为空。")
        
    gc_predict_path = model_path / "GC_matrix.csv"
    gc_constrain_path = model_path / "GC_matrix_constrained.csv"
    GC_predict_img_path = model_path / "GC_predict.png"
    causal_links_path = model_path / "causal_links.png"
    
    model = load_model(model_path, DEVICE)
    
    # 简单的长度校验
    if len(SERIES_NAME) != model.series_num:
        print(f"警告: SERIES_NAME 长度 ({len(SERIES_NAME)}) 与模型定义 ({model.series_num}) 不一致！")
    
    # 1. 保存预测矩阵 (CSV中存名字)
    res = save_gc_matrix_to_csv(model, gc_predict_path, SERIES_NAME, threshold=0.0, ignore_self_causality=True, is_softmax=False)
    
    if res:
        # 2. 约束矩阵 (读取名字CSV -> 处理 -> 存名字CSV)
        constrain_with_std_dev(gc_predict_path, gc_constrain_path, SERIES_NAME, std_multiplier=0.3)
        
        # 3. 绘图对比 (读取名字CSV -> 映射回索引绘图 -> 轴标签用名字)
        plot_gc_triple_compare(gc_predict_path, gc_constrain_path, SERIES_NAME, GC_predict_img_path, true_csv_path=GC_PATH, show_weights=True)
        
        # 4. 预测曲线对比 (文件名和标题用名字)
        prediction_compare(model, X_DATA, Y_DATA, SERIES_NAME, save_path=model_path)
        
        # 5. 绘制因果连接图
        save_causal_links(csv_path=gc_constrain_path, img_save_path=causal_links_path, series_names=SERIES_NAME)
        
if __name__ == "__main__":
    run_id = get_latest_run_id()
    # run_id = '12-19_22-01-44'
    model_path = Path('saved') / run_id
    main(model_path)