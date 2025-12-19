from datetime import datetime
from pathlib import Path
import joblib
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import functools
from logger.logger import get_logger, setup_logging
import optuna
from train_TS_GC import TS_GC_Trainer
from model.TS_GC import MutiTS_GC
from config import *


# 设置 Matplotlib 中文显示
rcParams['font.family'] = 'SimHei'
rcParams['axes.unicode_minus'] = False

N_TRIALS = 20 # Optuna 的试验次数
STUDY_NAME = "optuna"
STORAGE_PATH = f"sqlite:///{STUDY_NAME}.db"

overall_best_loss_optuna = float('inf')
overall_best_model_state_optuna = None

def objective(trial, logger, save_dir):
    global overall_best_loss_optuna, overall_best_model_state_optuna
    # 模型参数参数
    kernel_size = trial.suggest_categorical('kernel_size', [3, 5]) 
    temporal_layers = trial.suggest_categorical('temporal_layers', [2, 3, 4])
    feature_dim = trial.suggest_categorical('feature_dim', [16, 32, 64, 128, 256]) # 特征维度
    
    # 训练参数
    dropout = trial.suggest_float('dropout', 0, 0.3)
    lr = trial.suggest_float('lr', 0.006, 0.05, log=True) # 学习率
    lasso_param = trial.suggest_float('lasso_param', 0.001, 0.01, log=True) 
    ridge_param = trial.suggest_float('ridge_param', 0.001, 0.1, log=True)
    loss_functions_list = {
        'MSELoss': nn.MSELoss(), # 均方误差损失
        'L1Loss': nn.L1Loss(), # MAELoss，平均绝对误差损失
        'SmoothL1Loss': nn.SmoothL1Loss() # MSE 和 MAE 的结合，平滑 L1 损失
    }
    loss_function_name = trial.suggest_categorical('criterion', list(loss_functions_list.keys()))
    criterion = loss_functions_list[loss_function_name]  # 从字典中获取实际的损失函数对象
    penalty_type = trial.suggest_categorical('penalty_type', ['GL', 'GSGL', 'H']) # 惩罚类型

    logger.info(f"\n--- Trial {trial.number} ---")
    logger.info(f"  MutiTS_GC 参数: kernel_size={kernel_size}, temporal_layers={temporal_layers}, feature_dim={feature_dim}")
    logger.info(f"  训练参数: dropout = {dropout} loss_function={criterion}, lr={lr:.6f}, penalty={penalty_type}, lasso_param={lasso_param}, ridge_param={ridge_param:.3f}")

    model = MutiTS_GC(
        input_window = INPUT_WINDOW,
        output_window = OUTPUT_WINDOW,
        series_num = SERIES_NUM,
        feature_dim = feature_dim,
        temporal_layers = temporal_layers,
        kernel_size = kernel_size,
        dropout = dropout,
        device = DEVICE 
    ).to(DEVICE)

    
    trainer = TS_GC_Trainer(
        model=model, 
        epochs=20000, 
        save_dir=save_dir, 
        criterion=criterion,
        lr=lr, 
        device=DEVICE,
        series_num=SERIES_NUM,
        X_full=X_DATA,
        Y_full=Y_DATA,
        logger=logger,
        penalty_type=penalty_type, 
        lasso_param=lasso_param,
        ridge_param=ridge_param,
        verbose=1
    )
    
    best_loss= trainer.train()
    best_model_state = trainer.best_model_state
    
    if best_loss < overall_best_loss_optuna:
        overall_best_loss_optuna = best_loss
        overall_best_model_state_optuna = best_model_state
    
    trainer.cleanup()
    del model
    
    logger.info(f"Trial {trial.number} 完成。验证集 loss: {best_loss:.6f}")

    return best_loss

def begin_optuna(save_dir, logger):
    global overall_best_loss_optuna, overall_best_model_state_optuna
    
    # 为新的 Optuna study 初始化/重置全局最佳追踪变量
    overall_best_loss_optuna = float('inf')
    overall_best_model_state_optuna = None
    
    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=STORAGE_PATH,
        load_if_exists=True,
        direction='minimize' # 修改优化目标为最小化
    )
    optuna.logging.set_verbosity(optuna.logging.WARNING) # 设置 Optuna 的日志级别为警告级别

    print(f"\n开始 Optuna 超参数优化") 
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
    logger.info(f"  best_loss: {best_trial.value:.6f}") 
    logger.info(f"  最佳超参数:")
    best_params = best_trial.params
    
    for key, value in best_params.items():
        logger.info(f"    {key}: {value}")

    joblib.dump(best_params, save_dir / "model_config.pkl")
    if overall_best_model_state_optuna is not None:
        torch.save(overall_best_model_state_optuna, save_dir / "best_model.pth")
    return completed_trials, best_trial

# ---  可视化optuna优化过程 ---
def matplot_optuna(completed_trials, best_trial, save_dir):
    loss_values = [t.value for t in completed_trials if t.value is not None and np.isfinite(t.value)] # 过滤掉 None 和非有限值
    trial_numbers = [t.number for t in completed_trials if t.value is not None and np.isfinite(t.value)]

    plt.figure(figsize=(12, 7))
    plt.scatter(trial_numbers, loss_values, alpha=0.6, label='完成的 Trials', s=50)

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
    log_save_path = Path('saved') / run_id
    setup_logging(log_save_path)
    train_logger = get_logger() # 日志记录器
    
    completed_trials, best_trial = begin_optuna(log_save_path, train_logger) # 开始Optuna超参数优化
    matplot_optuna(completed_trials, best_trial, log_save_path) # 可视化优化过程
    
    
if __name__ == '__main__':
    run_id = datetime.now().strftime(r'%m-%d_%H-%M-%S') # 获得当前时间
    main(run_id)
