from datetime import datetime
from pathlib import Path
import joblib
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import rcParams
import functools
# from logger.logger import get_logger, setup_logging
import optuna
from logger.logger import get_logger, setup_logging
from model.Granger_causalFormer import PredictModel
from data_loader import TimeSeriesDataloader
from train_new import CausalFormerTrainer2

# from util import read_json
from train_causalformer import CausalFormerTrainer

# 设置 Matplotlib 中文显示
rcParams['font.family'] = 'SimHei'
rcParams['axes.unicode_minus'] = False

# 参数设置
# P = 5           # 时间序列的数量
# T = 1000        # 总时间点
# LAG = 2         # 真实的 VAR 滞后
# SPARSITY = 0.4  # 格兰杰因果矩阵的稀疏度
# BETA_VALUE = 0.8# 系数值
# SD = 0.1        # 噪声的标准差

DATA_SEED = 42  # 用于可重复性的随机种子
INPUT_WINDOW = 10 # 输入序列长度 (输入窗口)
OUTPUT_WINDOW = 1 # 预测下一个时间步
FEATURE_DIM = 1 # 每个时间序列在每个时间点上的特征数量
OUTPUT_DIM = 1  # 每个时间序列在每个预测时间步上输出的目标数量


# --- 训练和 Optuna 参数 ---
EPOCHS = 30
BATCH_SIZE = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_TRIALS = 50 # Optuna 的试验次数
STUDY_NAME = "optuna"
STORAGE_PATH = f"sqlite:///{STUDY_NAME}.db"


# --- 1. 生成并预处理数据 ---
# X_np, _, GC_true_np = simulate_var(p=P, T=T, lag=LAG, sparsity=SPARSITY,
#                                    beta_value=BETA_VALUE, sd=SD, seed=DATA_SEED)

# --- 1. 得到数据并预处理 ---
data_path = 'data/fMRI/timeseries9.csv'
true_gc_path = 'data/fMRI/sim9_gt_processed.csv'

timeseriesDataLoader = TimeSeriesDataloader(data_dir=data_path, gc_dir=true_gc_path, batch_size=BATCH_SIZE, 
                                            DATA_SEED=DATA_SEED, input_window=INPUT_WINDOW, output_window=OUTPUT_WINDOW)
train_loader, val_loader, test_loader = timeseriesDataLoader.split_sampler() # 得到训练集、验证集和测试集的数据加载器
series_num = timeseriesDataLoader.series_num # 获取序列数量
# --- 2. 定义 Optuna 目标函数 ---
def objective(trial, logger, save_dir):
    global FEATURE_DIM, OUTPUT_DIM
    
    # CausalFormer 参数
    d_model = trial.suggest_categorical('d_model', [32, 64, 128, 256])       # QK 嵌入维度
    n_head = trial.suggest_categorical('n_head', [2, 4, 8])             # 注意力头数
    n_layers = trial.suggest_int('n_layers', 1, 3)                      # Encoder 层数
    ffn_hidden = trial.suggest_categorical('ffn_hidden', [64, 128, 256, 512])# FFN 隐藏层维度
    dropout = trial.suggest_float('dropout', 0.0, 0.3)                  # Dropout
    tau = trial.suggest_float('tau', 0.5, 100.0, log=True)               # Softmax 温度

    # GrangerTCN 参数
    tcn_channels = trial.suggest_categorical('tcn_channels', [32, 64, 128, 256]) # TCN 通道数
    tcn_kernel_size = trial.suggest_categorical('tcn_kernel_size', [2, 3, 4]) # TCN 核大小
    tcn_dropout = trial.suggest_float('tcn_dropout', 0.0, 0.3)          # TCN Dropout

    # 近端梯度下降和稀疏性参数
    loss_functions_list = {
        'MSELoss': nn.MSELoss(), # 均方误差损失
        'L1Loss': nn.L1Loss(), # MAELoss，平均绝对误差损失
        'SmoothL1Loss': nn.SmoothL1Loss() # MSE 和 MAE 的结合，平滑 L1 损失
    }
    loss_function_name = trial.suggest_categorical('criterion', list(loss_functions_list.keys()))
    criterion = loss_functions_list[loss_function_name]  # 从字典中获取实际的损失函数对象
    lr = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)     # 学习率
    lambda_reg = trial.suggest_float('lambda_reg', 1e-5, 1e-1, log=True) # 正则化惩罚项在总损失函数中的整体权重或强度
    penalty_type = trial.suggest_categorical('penalty_type', ['GL', 'GSGL', 'H']) # 惩罚类型
    # penalty_type = 'GL' # 惩罚类型

    print(f"\n--- Trial {trial.number} ---")
    print(f"  CausalFormer 参数:d_model={d_model}, n_head={n_head}, n_layers={n_layers}, ffn_hidden={ffn_hidden}, dropout={dropout:.3f}, tau={tau:.3f}")
    print(f"  GrangerTCN 参数: channels={tcn_channels}, kernel={tcn_kernel_size}, dropout={tcn_dropout:.3f}")
    print(f"  训练参数: loss_function={loss_function_name}, lr={lr:.6f}, lambda_reg={lambda_reg:.6f}, penalty={penalty_type}")

    # --- 模型和损失函数 ---
    # 创建配置字典传递给 PredictModel
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
        'device': DEVICE.type # 传递设备类型
    }

    model = PredictModel(config=config,
                         d_model=d_model,
                         n_head=n_head,
                         n_layers=n_layers,
                         tcn_channels=tcn_channels,
                         tcn_kernel_size=tcn_kernel_size,
                         tcn_dropout=tcn_dropout,
                         ffn_hidden=ffn_hidden,
                         drop_prob=dropout,
                         tau=tau).to(DEVICE)

    # --- 近端优化训练循环 ---
    final_avg_val_loss = float('inf')

    # ---开始训练---
    causalFormerTrainer = CausalFormerTrainer2(model=model, epoch=EPOCHS, save_dir= save_dir, criterion=criterion,lr=lr, device=DEVICE,
                                               train_loader=train_loader, valid_loader=val_loader, series_num=series_num,
                                               penalty_type=penalty_type, lambda_reg=lambda_reg)
    # causalFormerTrainer = CausalFormerTrainer(model=model, epoch=EPOCHS, save_dir= save_dir, criterion=criterion,lr=lr, device=DEVICE,
    #                                            train_loader=train_loader, valid_loader=val_loader, series_num=series_num,
    #                                            penalty_type=penalty_type, lambda_reg=lambda_reg)
    final_avg_val_loss = causalFormerTrainer.train()
    logger.info(f"Trial {trial.number} 完成。最终验证集 loss: {final_avg_val_loss:.6f}")

    return final_avg_val_loss

def begin_optuna(save_dir, logger):
    # --- 3. 创建或加载 Optuna Study 并运行优化 ---
    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=STORAGE_PATH,
        load_if_exists=True,
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
        joblib.dump(best_params, save_dir / "best_params.pkl") # 保存最佳参数
    return completed_trials, best_trial, best_params

# --- 5. Matplotlib可视化优化过程 ---
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

def main(run_id):
    # 设置记录器
    log_save_path = Path('saved') / run_id
    setup_logging(log_save_path)
    train_logger = get_logger() # 日志记录器
    
    completed_trials, best_trial, best_params = begin_optuna(log_save_path, train_logger) # 开始Optuna超参数优化
    matplot_optuna(completed_trials, best_trial, log_save_path) # 可视化优化过程

    
if __name__ == '__main__':
    run_id = datetime.now().strftime(r'%m-%d_%H-%M-%S') # 获得当前时间
    main(run_id)
