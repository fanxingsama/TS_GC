from datetime import datetime
import os
from pathlib import Path
import joblib
import re
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import rcParams
import functools
# from logger.logger import get_logger, setup_logging
import optuna
from MutiTCN.only_tcn import MultiTCNModel
from logger.logger import get_logger, setup_logging
from model.Granger_causalFormer import PredictModel
from data_loader import TimeSeriesDataloader
from train_TCN import MultiTCNTrainer
from train_with_line import CausalFormerTrainer2
from train_no_line import CausalFormerTrainer3
from config import DATA_PATH, gc_dir, BATCH_SIZE, DATA_SEED, INPUT_WINDOW, OUTPUT_WINDOW, FEATURE_DIM, OUTPUT_DIM, EPOCHS, DEVICE, timeseriesDataLoader, SERIES_NUM


# 设置 Matplotlib 中文显示
rcParams['font.family'] = 'SimHei'
rcParams['axes.unicode_minus'] = False

N_TRIALS = 50 # Optuna 的试验次数
STUDY_NAME = "optuna"
STORAGE_PATH = f"sqlite:///{STUDY_NAME}.db"

train_loader, val_loader, test_loader = timeseriesDataLoader.split_sampler() # 得到训练集、验证集和测试集的数据加载器
# --- 2. 定义 Optuna 目标函数 ---
def objective(trial, logger, save_dir):
    
    # CausalFormer 参数
    # d_model = trial.suggest_categorical('d_model', [32, 64, 128, 256])       # QK 嵌入维度
    # n_head = trial.suggest_categorical('n_head', [2, 4, 8, 16, 32])             # 注意力头数
    # n_layers = trial.suggest_int('n_layers', 1, 3)                      # Encoder 层数
    # ffn_hidden = trial.suggest_categorical('ffn_hidden', [64, 128, 256, 512])# FFN 隐藏层维度
    # tau = round(trial.suggest_float('tau', 0.5, 100.0, log=True), 5)               # Softmax 温度
    dropout = round(trial.suggest_float('dropout', 0.1, 0.5), 5)                 # Dropout

    # GrangerTCN 参数
    tcn_channels = trial.suggest_categorical('tcn_channels', [16, 32, 64, 128, 256]) # TCN 通道数
    kernel_size = trial.suggest_categorical('kernel_size', [3, 4, 5]) # TCN 核大小
    tcn_dropout = round(trial.suggest_float('tcn_dropout', 0.0, 0.3), 5)         # TCN Dropout
    lasso_param = trial.suggest_float('lasso_param', 1e-3, 1, log=True)  # 正则化系数
    ridge_param = trial.suggest_float('ridge_param', 1e-4, 0.01, log=True)

    # 近端梯度下降和稀疏性参数
    loss_functions_list = {
        'MSELoss': nn.MSELoss(), # 均方误差损失
        'L1Loss': nn.L1Loss(), # MAELoss，平均绝对误差损失
        'SmoothL1Loss': nn.SmoothL1Loss() # MSE 和 MAE 的结合，平滑 L1 损失
    }
    loss_function_name = trial.suggest_categorical('criterion', list(loss_functions_list.keys()))
    criterion = loss_functions_list[loss_function_name]  # 从字典中获取实际的损失函数对象
    lr = round(trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True), 5)     # 学习率
    penalty_type = trial.suggest_categorical('penalty_type', ['GL', 'GSGL', 'H']) # 惩罚类型

    logger.info(f"\n--- Trial {trial.number} ---")
    # logger.info(f"  CausalFormer 参数:d_model={d_model}, n_head={n_head}, n_layers={n_layers}, ffn_hidden={ffn_hidden}, dropout={dropout:.3f}, tau={tau:.3f}")
    logger.info(f"  GrangerTCN 参数: channels={tcn_channels}, kernel_size={kernel_size}, dropout={tcn_dropout:.3f}")
    logger.info(f"  训练参数: loss_function={loss_function_name}, lr={lr:.6f}, penalty={penalty_type}, lasso_param={lasso_param}, ridge_param={ridge_param:.3f}")

    # --- 模型和损失函数 ---
    # 创建配置字典传递给 PredictModel
    # model = PredictModel(input_window=INPUT_WINDOW,
    #                      output_window=OUTPUT_WINDOW,
    #                      series_num=SERIES_NUM,
    #                      feature_dim=FEATURE_DIM,
    #                      output_dim=OUTPUT_DIM,
    #                      device=DEVICE,
    #                      d_model=d_model,
    #                      n_head=n_head,
    #                      n_layers=n_layers,
    #                      tcn_channels=tcn_channels,
    #                      tcn_kernel_size=kernel_size,
    #                      tcn_dropout=tcn_dropout,
    #                      ffn_hidden=ffn_hidden,
    #                      dropout=dropout, 
    #                      tau=tau).to(DEVICE)
    
    model = MultiTCNModel(
        input_window=INPUT_WINDOW,
        output_window=OUTPUT_WINDOW,
        series_num=SERIES_NUM,
        feature_dim=FEATURE_DIM,
        output_dim=OUTPUT_DIM,
        device=DEVICE,
        tcn_channels=tcn_channels,
        kernel_size=kernel_size,
        dropout=dropout,
    ).to(DEVICE)

    # ---开始训练---
    # causalFormerTrainer = CausalFormerTrainer3(model=model, epoch=EPOCHS, save_dir= save_dir, criterion=criterion,lr=lr, device=DEVICE,
    #                                            train_loader=train_loader, valid_loader=val_loader, series_num=SERIES_NUM, lasso_param = lasso_param,
    #                                            penalty_type=penalty_type)
    
    causalFormerTrainer = MultiTCNTrainer(
        model=model, 
        epochs=EPOCHS, 
        save_dir=save_dir, 
        criterion=criterion,
        lr=lr, 
        device=DEVICE,
        train_loader=train_loader, 
        valid_loader=val_loader, 
        series_num=SERIES_NUM, 
        logger=logger,
        penalty_type=penalty_type, 
        lasso_param=lasso_param,
        ridge_param=ridge_param
    )
    Vailed_mse= causalFormerTrainer.train()
    
    # 保存训练器的状态用于后续恢复
    trainer_losses = {
        'train_losses': [float(x) for x in causalFormerTrainer.train_losses],
        'train_mses': [float(x) for x in causalFormerTrainer.train_mses],
        'train_ridges': [float(x) for x in causalFormerTrainer.train_ridges],
        'train_penalties': [float(x) for x in causalFormerTrainer.train_penalties],
        'val_losses': [float(x) for x in causalFormerTrainer.val_losses],
        'val_mses': [float(x) for x in causalFormerTrainer.val_mses],
        'val_ridges': [float(x) for x in causalFormerTrainer.val_ridges],
        'val_penalties': [float(x) for x in causalFormerTrainer.val_penalties]
    }
    best_model_state = causalFormerTrainer.best_model_state
    trial.set_user_attr('best_model_state', best_model_state)
    
    causalFormerTrainer.cleanup()
    del model
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 保存训练器的状态用于后续恢复
    trial.set_user_attr('trainer_losses', trainer_losses)
    
    logger.info(f"Trial {trial.number} 完成。验证集 loss: {Vailed_mse:.6f}")

    return Vailed_mse

def begin_optuna(save_dir, logger):
    study = optuna.create_study(
        study_name=STUDY_NAME,
        # storage=STORAGE_PATH,
        # load_if_exists=True,
        direction='minimize' # 修改优化目标为最小化
    )
    optuna.logging.set_verbosity(optuna.logging.WARNING) # 设置 Optuna 的日志级别为警告级别

    print(f"\n开始 Optuna 超参数优化") # 更新打印信息
    objective_with_logger = functools.partial(objective, logger=logger, save_dir = save_dir)
    study.optimize(objective_with_logger, n_trials=N_TRIALS, timeout=None)

    # --- 4. 输出结果 ---
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE] # 从Optuna中获取所有完成的Trial
    logger.info(f"\nOptuna 优化完成,总尝试次数: {len(study.trials)}，成功完成的 Trial 数量: {len(completed_trials)}")

    best_params = None
    best_trial = None 

    best_trial = study.best_trial
    logger.info(f"\n最佳 Trial:")
    logger.info(f"  编号: {best_trial.number}")
    logger.info(f"  目标值 (Val MSE): {best_trial.value:.6f}") 
    logger.info(f"  最佳超参数:")
    best_params = best_trial.params
    
    for key, value in best_params.items():
        logger.info(f"    {key}: {value}")
         # 保存最佳参数和模型
        joblib.dump(best_params, save_dir / "model_config.pkl")
        torch.save(best_trial.user_attrs['best_model_state'], save_dir / "best_model.pth")
    return completed_trials, best_trial, best_params

# ---  可视化optuna优化过程 ---
def matplot_optuna(completed_trials, best_trial, save_dir):
    mse_values = [t.value for t in completed_trials if t.value is not None and np.isfinite(t.value)] # 过滤掉 None 和非有限值
    valid_trial_numbers = [t.number for t in completed_trials if t.value is not None and np.isfinite(t.value)]

    plt.figure(figsize=(12, 7))
    plt.scatter(valid_trial_numbers, mse_values, alpha=0.6, label='完成的 Trials', s=50)

    # 标记最佳 trial (如果存在且值有效)
    if best_trial is not None and best_trial.value is not None and np.isfinite(best_trial.value):
        plt.scatter([best_trial.number], [best_trial.value], color='red', s=150, label=f'最佳 Trial ({best_trial.number}) MSE: {best_trial.value:.4f}', zorder=5, edgecolors='black')

    plt.xlabel("Trial 编号")
    plt.ylabel("验证集 MSE") # 修改 Y 轴标签
    plt.title(f"Optuna 优化历史") # 修改标题
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / 'optuna_history.png') # 保存图像

# 可视化最优的一组的训练过程
def plot_best_trial_curves(best_trial, save_dir):
    # 获取最佳trial的损失数据
    losses = best_trial.user_attrs['trainer_losses']
    
    # 创建保存图表的目录
    best_plots_dir = save_dir / "best_trial_train_plots"
    os.makedirs(best_plots_dir, exist_ok=True)
    
    # 绘制训练和验证的总损失
    plt.figure(figsize=(10, 6))
    plt.plot(losses['train_losses'], label='Train Loss')
    plt.plot(losses['val_losses'], label='Validation Loss')
    plt.title(f'Best Trial ({best_trial.number}) - Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(best_plots_dir / 'total_loss.png')
    plt.close()
    
    # 绘制训练和验证的MSE损失
    plt.figure(figsize=(10, 6))
    plt.plot(losses['train_mses'], label='Train MSE')
    plt.plot(losses['val_mses'], label='Validation MSE')
    plt.title(f'Best Trial ({best_trial.number}) - MSE Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()
    plt.grid(True)
    plt.savefig(best_plots_dir / 'mse_loss.png')
    plt.close()
    
    # 绘制训练和验证的Ridge正则化损失
    plt.figure(figsize=(10, 6))
    plt.plot(losses['train_ridges'], label='Train Ridge')
    plt.plot(losses['val_ridges'], label='Validation Ridge')
    plt.title(f'Best Trial ({best_trial.number}) - Ridge Regularization')
    plt.xlabel('Epoch')
    plt.ylabel('Ridge Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(best_plots_dir / 'ridge_loss.png')
    plt.close()
    
    # 绘制训练和验证的非平滑正则化损失
    plt.figure(figsize=(10, 6))
    plt.plot(losses['train_penalties'], label='Train Penalty')
    plt.plot(losses['val_penalties'], label='Validation Penalty')
    plt.title(f'Best Trial ({best_trial.number}) - Non-smooth Regularization')
    plt.xlabel('Epoch')
    plt.ylabel('Penalty Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(best_plots_dir / 'penalty_loss.png')
    plt.close()
    
    print(f"Best trial ({best_trial.number}) training plots saved to {best_plots_dir}")
def main(run_id):
    # 设置记录器
    log_save_path = Path('saved') / run_id
    setup_logging(log_save_path)
    train_logger = get_logger() # 日志记录器
    
    completed_trials, best_trial, best_params = begin_optuna(log_save_path, train_logger) # 开始Optuna超参数优化
    matplot_optuna(completed_trials, best_trial, log_save_path) # 可视化优化过程
    plot_best_trial_curves(best_trial, log_save_path) # 绘制最佳 trial 的训练曲线

    
if __name__ == '__main__':
    run_id = datetime.now().strftime(r'%m-%d_%H-%M-%S') # 获得当前时间
    main(run_id)
