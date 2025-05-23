from pathlib import Path
import re
import joblib
from sklearn import preprocessing
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from MutiTCN.only_tcn import MultiTCNModel
from visual.causalMatrix import visualize_single_causality_csv
from model.Granger_causalFormer import PredictModel
import pandas as pd
import os
from config import INPUT_WINDOW, OUTPUT_WINDOW, FEATURE_DIM, OUTPUT_DIM, DEVICE, timeseriesDataLoader, SERIES_NUM
# 设置 Matplotlib 中文显示
rcParams['font.family'] = 'SimHei'
rcParams['axes.unicode_minus'] = False

# 加载模型
def load_model(model_path, device):
    # 加载参数
    saved_config = joblib.load(model_path / "model_config.pkl")
    
    input_window = INPUT_WINDOW
    output_window = OUTPUT_WINDOW
    feature_dim = FEATURE_DIM
    output_dim = OUTPUT_DIM
    series_num = SERIES_NUM
    dropout = saved_config['dropout'] 
    
    # # TCN 参数
    tcn_channels = saved_config['tcn_channels'] 
    kernel_size = saved_config['kernel_size']

    # # transformer参数
    # d_model = saved_config['d_model']
    # n_head = saved_config['n_head']
    # n_layers = saved_config['n_layers']
    # ffn_hidden = saved_config['ffn_hidden']
    # tau = saved_config['tau']

    # model = PredictModel(
    #     input_window=input_window,
    #     output_window=output_window,
    #     series_num=series_num,
    #     feature_dim=feature_dim,
    #     output_dim=output_dim,
    #     device=device,  
    #     d_model=d_model,
    #     n_head=n_head,
    #     n_layers=n_layers,
    #     tcn_channels=tcn_channels,
    #     tcn_kernel_size=kernel_size,
    #     tcn_dropout=tcn_dropout,
    #     ffn_hidden=ffn_hidden,
    #     dropout=dropout, 
    #     tau=tau
    # ).to(device)

    
    model = MultiTCNModel(
        input_window=input_window,
        output_window=output_window,
        series_num=series_num,
        feature_dim=feature_dim,
        output_dim=output_dim,
        device=DEVICE,
        tcn_channels=tcn_channels,
        kernel_size=kernel_size,
        dropout=dropout,
    ).to(DEVICE)
    
    
    model_weight_path = model_path / "best_model.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    return model

# 采用真阳和假阳评估指标
def evaluate(logger, gtfile, validatedcauses, columns):
    extendedgtdelays, readgt, extendedreadgt = getextendeddelays(gtfile, columns)
    FP=0
    FPdirect=0
    TPdirect=0
    TP=0
    FN=0
    FPs = []
    FPsdirect = []
    TPsdirect = []
    TPs = []
    FNs = []
    for key in readgt:
        for v in validatedcauses[key]:
            if v not in extendedreadgt[key]:
                FP+=1
                FPs.append((key,v))
            else:
                TP+=1
                TPs.append((key,v))
            if v not in readgt[key]:
                FPdirect+=1
                FPsdirect.append((key,v))
            else:
                TPdirect+=1
                TPsdirect.append((key,v))
        for v in readgt[key]:
            if v not in validatedcauses[key]:
                FN+=1
                FNs.append((key, v))
    
    def serialization(data):
        return [f"{e[1]}->{e[0]}" for e in data]
    logger.info(f"假阳性': {FP}")
    logger.info(f"真阳性': {TP}")
    logger.info(f"假阴性: {FN}")
    logger.info(f"直接误报总数: {FPdirect}")
    logger.info(f"直接真阳性总数: {TPdirect}")
    logger.info(f"真阳性序列': {serialization(TPs)}")
    logger.info(f"假阳性序列': {serialization(FPs)}")
    logger.info(f"直接真阳性序列: {serialization(TPsdirect)}")
    logger.info(f"直接假阳性序列: {serialization(FPsdirect)}")
    logger.info(f"FNs: {serialization(FNs)}")
    precision = recall = 0.

    logger.info('(包括直接和间接的因果关系)')
    if float(TP+FP)>0:
        precision = TP / float(TP+FP)
    logger.info(f"Precision': {precision}")
    if float(TP + FN)>0:
        recall = TP / float(TP + FN)
    logger.info(f"Recall': {recall}")
    if (precision + recall) > 0:
        F1 = 2 * (precision * recall) / (precision + recall)
    else:
        F1 = 0.
    logger.info(f"F1' score: {F1}")

    logger.info('(只包括直接的因果关系)')
    precision = recall = 0.
    if float(TPdirect+FPdirect)>0:
        precision = TPdirect / float(TPdirect+FPdirect)
    logger.info(f"Precision: {precision}")
    if float(TPdirect + FN)>0:
        recall = TPdirect / float(TPdirect + FN)
    logger.info(f"Recall: {recall}")
    if (precision + recall) > 0:
        F1direct = 2 * (precision * recall) / (precision + recall)
    else:
        F1direct = 0.
    logger.info(f"F1 score: {F1direct}")
    return FP, TP, FPdirect, TPdirect, FN, FPs, FPsdirect, TPs, TPsdirect, FNs, F1, F1direct

# 评估模型
def evaluate_model(model, test_loader, device, max_samples=200):
    model.eval()
    
    # 用于存储结果的字典
    results = {
        'predictions': [], # 模型预测值
        'targets': [], # 数据集真实值
    }
    
    # 获取测试数据上的预测结果
    with torch.no_grad():
        sample_count = 0
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            # 获取预测结果
            predictions = model(batch_x) # predictions:[batch_size, output_window, series_num, output_dim]

            results['predictions'].append(predictions.cpu().numpy()) # [max_samples, output_window, series_num, output_dim]
            results['targets'].append(batch_y.cpu().numpy())
            
            # 就只要100个样本数
            sample_count += batch_x.size(0)
            if sample_count >= max_samples:
                break
    
    # 将结果拼接为连续的numpy数组
    results['predictions'] = np.concatenate(results['predictions'], axis=0)[:max_samples]
    results['targets'] = np.concatenate(results['targets'], axis=0)[:max_samples]
    
    return results

# 保存模型的格兰杰因果关系矩阵到 CSV 文件
def save_gc_to_csv(model, series_num, path, threshold=False, ignore_kernel=True):
    """
    Args:
        path (str): CSV 文件的保存路径
        threshold (bool): 是否使用阈值化的结果
        ignore_kernel (bool): 是否忽略核大小维度
    """
    csv_path = os.path.join(path, "GC_matrix.csv")

    # 获取 GC 矩阵
    gc_matrix = model.GC(threshold=threshold, ignore_kernel=ignore_kernel)
    
    if ignore_kernel: # 忽略滞后
        cause_effect_pairs = [] 
        for effect_idx in range(series_num): # 果
            for cause_idx in range(series_num): # 因
                if effect_idx != cause_idx:  # 可选：排除自因果
                    strength = gc_matrix[effect_idx, cause_idx].item() # 因果关系强度
                    # 如果需要阈值化结果，只保存非零项
                    if not threshold or (threshold and strength > 0):
                        cause_effect_pairs.append({
                            'source': f'{cause_idx}',
                            'target': f'{effect_idx}',
                            'Strength': round(strength, 2)
                        })
        df = pd.DataFrame(cause_effect_pairs)
        # 按照 'Cause' 排序，如果 'Cause' 相同，则按照 'Effect' 排序
        if not df.empty:
            df.sort_values(by=['source', 'target'], inplace=True)
        
    else:
        cause_effect_pairs = []
        for effect_idx in range(series_num):
            for cause_idx in range(series_num):
                if effect_idx != cause_idx:  # 可选：排除自因果
                    for lag in range(gc_matrix.shape[2]):
                        strength = gc_matrix[effect_idx, cause_idx, lag].item()
                        # 如果需要阈值化结果，只保存非零项
                        if not threshold or (threshold and strength > 0):
                            cause_effect_pairs.append({
                                'Cause': f'Series_{cause_idx}',
                                'Effect': f'Series_{effect_idx}',
                                'lag': lag,
                                'Strength': round(strength, 2)
                            })
        df = pd.DataFrame(cause_effect_pairs)
        # 按照 'Cause' 排序，然后 'Effect'，最后 'lag'
        if not df.empty:
            df.sort_values(by=['source', 'target', 'lag'], inplace=True)
    df.to_csv(csv_path, index=False)
    matrix_png_save_path = os.path.join(path, "GC_matrix.png")
    visualize_single_causality_csv(csv_path, matrix_png_save_path, show=False)

# 绘制预测结果和真实值的时序序列对比图
def plot_predictions(results, series_num, plot_indices, save_path):
    """
    Args:
        results: 包含预测结果和真实值的字典
        series_num: 时间序列的数量
        plot_indices: 要绘制的时间序列索引列表，默认为前5个序列
        save_path: 保存图表的路径
    """
    if plot_indices is None:
        # 默认绘制前5个时间序列或全部（如果少于5个）
        plot_indices = list(range(min(5, series_num)))
    
    # 获取样本数
    n_samples = results['predictions'].shape[0]
    time_steps = np.arange(n_samples)
    
    # 创建足够大的图表
    plt.figure(figsize=(10, 2 * len(plot_indices)))
    
    # 为每个选定的时间序列创建子图
    for i, idx in enumerate(plot_indices):
        plt.subplot(len(plot_indices), 1, i + 1)
        
        # 提取当前序列的预测和真实值（使用原始尺度的数据）
        pred_series = results['predictions'][:, 0, idx, 0]  # [samples, output_window=1, series_idx, output_dim=1]
        target_series = results['targets'][:, 0, idx, 0]
        
        # 绘制曲线
        plt.plot(time_steps, target_series, 'b-', label='真实值', linewidth=2)
        plt.plot(time_steps, pred_series, 'r--', label='预测值', linewidth=2)
        
        # 添加指标信息
        # mse = results['mse_per_series'][idx]
        # mae = results['mae_per_series'][idx]
        # r2 = results['r2_per_series'][idx]
        # plt.title(f'时间序列 {idx+1}: MSE={mse:.4f}, MAE={mae:.4f}, R²={r2:.4f}')
        plt.title(f'时间序列 {idx+1}')
        
        plt.xlabel('时间步')
        plt.ylabel('值')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)

# 获取最新的run_id
def get_latest_run_id_simple():
    base_path = Path('saved')
    if not base_path.exists():
        return None
    
    # 获取所有符合格式的目录名
    pattern = re.compile(r'^\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$')
    timestamps = [d.name for d in base_path.iterdir() 
                 if d.is_dir() and pattern.match(d.name)]
    
    return max(timestamps) if timestamps else None

def main(model_path):
    train_loader, val_loader, test_loader = timeseriesDataLoader.split_sampler()
    print(f"时间序列数量: {SERIES_NUM}")
    print(f"测试集数据大小: {len(test_loader.dataset)}")

    model = load_model(model_path, DEVICE) # 构建模型
    
    # 评估模型
    results = evaluate_model(model, test_loader, DEVICE, max_samples=200)
    
    # 得到格兰杰因果关系
    save_gc_to_csv(model, SERIES_NUM, model_path, threshold=False, ignore_kernel=True)
    
    # 选择前5个序列进行绘制
    plot_indices = list(range(min(5, SERIES_NUM)))
    plot_predictions(results, SERIES_NUM, plot_indices, save_path= model_path / "model_predict.png")



if __name__ == "__main__":
    run_id = get_latest_run_id_simple()
    # run_id = '05-20_22-10-15'
    model_path = Path('saved') / run_id
    main(model_path)