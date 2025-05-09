#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
import joblib
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import pandas as pd
from joblib import load
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from model.Granger_causalFormer import PredictModel
from data_loader import TimeSeriesDataloader

# 设置 Matplotlib 中文显示
rcParams['font.family'] = 'SimHei'
rcParams['axes.unicode_minus'] = False

# 加载模型
def load_model(config, best_params, device):
    """
    根据最佳参数加载模型
    """
    # 从最佳参数中提取模型参数
    d_model = best_params['d_model']
    n_head = best_params['n_head']
    n_layers = best_params['n_layers']
    ffn_hidden = best_params['ffn_hidden']
    dropout = best_params['dropout']
    tau = best_params['tau']
    
    # GrangerTCN 参数
    tcn_layers = best_params['tcn_layers']
    tcn_channels = best_params['tcn_channels']
    tcn_kernel_size = best_params['tcn_kernel_size']
    tcn_dropout = best_params['tcn_dropout']
    tcn_channel_list = [tcn_channels] * tcn_layers
    
    # 创建并返回模型
    model = PredictModel(
        config=config,
        d_model=d_model,
        n_head=n_head,
        n_layers=n_layers,
        tcn_channels=tcn_channel_list,
        tcn_kernel_size=tcn_kernel_size,
        tcn_dropout=tcn_dropout,
        ffn_hidden=ffn_hidden,
        drop_prob=dropout,
        tau=tau
    ).to(device)
    
    return model

# 评估模型
def evaluate_model(model, test_loader, device, series_num, max_samples=100):
    model.eval()
    
    # 用于存储结果的字典
    results = {
        'predictions': [], # 模型预测值
        'targets': [], # 数据集真实值
        'mse_per_series': [],
        'mae_per_series': [],
        'r2_per_series': []
    }
    
    # 获取测试数据上的预测结果
    with torch.no_grad():
        sample_count = 0
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            # 获取预测结果
            predictions = model(batch_x)
            
            # 将预测结果和真实值存储到 CPU 上
            results['predictions'].append(predictions.cpu().numpy())
            results['targets'].append(batch_y.cpu().numpy())
            
            # 就只要100个样本数
            sample_count += batch_x.size(0)
            if sample_count >= max_samples:
                break
    
    # 将结果拼接为完整的数组
    results['predictions'] = np.concatenate(results['predictions'], axis=0)[:max_samples]
    results['targets'] = np.concatenate(results['targets'], axis=0)[:max_samples]
    
    # 计算每个时间序列的评估指标
    for i in range(series_num):
        # 得到当前序列的预测值和真实值
        pred_series = results['predictions'][:, 0, i, 0]  # [samples, output_window=1, series_num, output_dim=1]
        target_series = results['targets'][:, 0, i, 0]
        
        # 计算指标
        mse = mean_squared_error(target_series, pred_series)
        mae = mean_absolute_error(target_series, pred_series)
        r2 = r2_score(target_series, pred_series)
        
        results['mse_per_series'].append(mse) # 计算均方误差
        results['mae_per_series'].append(mae) # 计算平均绝对误差
        results['r2_per_series'].append(r2) # 计算R2分数
    
    return results

# 获得 模型的Granger 因果关系
# def get_GC_and_save(model, threshold=True, ignore_lag=True):
#     GC = model.get_GC(threshold=True, ignore_lag=True)
    

# 绘制预测结果和真实值的时序序列对比图
def plot_predictions(results, series_num, plot_indices, save_path=None):
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
    plt.figure(figsize=(15, 3 * len(plot_indices)))
    
    # 为每个选定的时间序列创建子图
    for i, idx in enumerate(plot_indices):
        plt.subplot(len(plot_indices), 1, i + 1)
        
        # 提取当前序列的预测和真实值
        pred_series = results['predictions'][:, 0, idx, 0]  # [samples, output_window=1, series_idx, output_dim=1]
        target_series = results['targets'][:, 0, idx, 0]
        
        # 绘制曲线
        plt.plot(time_steps, target_series, 'b-', label='真实值', linewidth=2)
        plt.plot(time_steps, pred_series, 'r--', label='预测值', linewidth=2)
        
        # 添加指标信息
        mse = results['mse_per_series'][idx]
        mae = results['mae_per_series'][idx]
        r2 = results['r2_per_series'][idx]
        plt.title(f'时间序列 {idx+1}: MSE={mse:.4f}, MAE={mae:.4f}, R²={r2:.4f}')
        
        plt.xlabel('时间步')
        plt.ylabel('值')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
    
    plt.tight_layout()
    
    # 保存图表
    if save_path:
        plt.savefig(save_path)
        print(f"预测结果已保存到: {save_path}")
    
    plt.show()

def main(png_save_path):
    data_path = 'data/fMRI/timeseries9.csv'
    gc_dir = 'data/fMRI/sim9_gt_processed.csv'
    BATCH_SIZE = 64
    DATA_SEED = 42
    INPUT_WINDOW = 20
    OUTPUT_WINDOW = 1
    FEATURE_DIM = 1
    OUTPUT_DIM = 1
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载数据
    timeseriesDataLoader = TimeSeriesDataloader(data_dir=data_path, gc_dir=gc_dir, batch_size=BATCH_SIZE, 
                                            DATA_SEED=DATA_SEED, input_window=INPUT_WINDOW, output_window=OUTPUT_WINDOW)
    
    # 获取数据加载器和序列数量
    _, _, test_loader = timeseriesDataLoader.split_sampler()
    series_num = timeseriesDataLoader.series_num
    GC_true_np = timeseriesDataLoader.get_true_granger()
    
    print(f"时间序列数量: {series_num}")
    print(f"测试集数据大小: {len(test_loader.dataset)}")
    
    # 加载模型最佳参数
    best_params_file = png_save_path / "best_params.pkl"
    best_params = joblib.load(best_params_file)
    config = {
        'data_loader': {
            'args': {
                'input_window': INPUT_WINDOW,
                'output_window': OUTPUT_WINDOW,
                'feature_dim': FEATURE_DIM,
                'output_dim': OUTPUT_DIM,
                'series_num': series_num
            }
        },
        'device': device.type
    }
    model = load_model(config, best_params, device) # 构建模型
    
    # 评估模型
    results = evaluate_model(model, test_loader, device, series_num, max_samples=100)
    
    # 计算总体指标
    overall_mse = np.mean(results['mse_per_series'])
    overall_mae = np.mean(results['mae_per_series'])
    overall_r2 = np.mean(results['r2_per_series'])
    
    print("\n总体模型性能:")
    print(f"平均 MSE: {overall_mse:.6f}")
    print(f"平均 MAE: {overall_mae:.6f}")
    print(f"平均 R²: {overall_r2:.6f}")
    
    # 绘制预测结果
    print("\n绘制预测结果...")
    # 选择前5个序列进行绘制
    plot_indices = list(range(min(5, series_num)))
    plot_predictions(results, series_num, plot_indices, save_path= png_save_path / "model_predict.png")

if __name__ == "__main__":
    run_id = "05-09_15-06-15"  # 
    png_save_path = Path('saved') / run_id
    main(png_save_path)
    