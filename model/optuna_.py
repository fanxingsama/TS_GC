import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import optuna
import traceback
import matplotlib.pyplot as plt
from matplotlib import rcParams
import os

# 根据你的项目结构调整导入路径
# 假设此文件位于 model/optuna_tuning/
from train_causalformer import train_and_evaluate, create_sequences

# 设置 Matplotlib 中文显示 
rcParams['font.family'] = 'SimHei' # 或其他可用的中文字体
rcParams['axes.unicode_minus'] = False


# --- 常量 (考虑移到配置文件或作为参数传递) ---
OUTPUT_WINDOW = 1         # 预测范围
FEATURE_DIM = 1           # 特征维度
OUTPUT_DIM = 1            # 输出维度
BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 50               # 每个 Optuna 试验的轮数
N_TRIALS = 50             # Optuna 试验次数
STUDY_NAME = "causalformer-grangertcn-pgd-study-refactored"
STORAGE_PATH = f"sqlite:///{STUDY_NAME}.db"

# --- 全局数据占位符 (由调用脚本填充) ---
X_train_np_global = None
X_val_np_global = None
GC_true_np_global = None
P_global = None

# --- Optuna 目标函数 ---
def objective(trial):
    global X_train_np_global, X_val_np_global, GC_true_np_global, P_global

    if X_train_np_global is None or X_val_np_global is None or GC_true_np_global is None or P_global is None:
        raise ValueError("调用目标函数前未设置全局数据变量。")

    # --- 超参数建议 ---
    params = {
        'input_window': trial.suggest_categorical('input_window', [10, 20, 30]),
        'd_model': trial.suggest_categorical('d_model', [32, 64, 128]),
        'n_head': trial.suggest_categorical('n_head', [2, 4, 8]),
        'n_layers': trial.suggest_int('n_layers', 1, 3),
        'ffn_hidden': trial.suggest_categorical('ffn_hidden', [64, 128, 256]),
        'dropout': trial.suggest_float('dropout', 0.0, 0.3),
        'tau': trial.suggest_float('tau', 0.5, 10.0, log=True),
        'tcn_layers': trial.suggest_int('tcn_layers', 2, 5),
        'tcn_channels': trial.suggest_categorical('tcn_channels', [16, 32, 48]),
        'tcn_kernel_size': trial.suggest_categorical('tcn_kernel_size', [2, 3, 4]),
        'tcn_dropout': trial.suggest_float('tcn_dropout', 0.0, 0.3),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
        'lambda_reg': trial.suggest_float('lambda_reg', 1e-5, 1e-1, log=True),
        'penalty_type': trial.suggest_categorical('penalty_type', ['GL', 'GSGL']),
    }
    if params['penalty_type'] == 'GSGL':
        params['alpha_gsgl'] = trial.suggest_float('alpha_gsgl', 0.1, 0.9)

    print(f"\n--- 试验 {trial.number} ---")

    # --- 数据准备 (使用建议的 input_window) ---
    input_window = params['input_window']
    output_window = OUTPUT_WINDOW

    X_train_seq, y_train_seq = create_sequences(X_train_np_global, input_window, output_window)
    X_val_seq, y_val_seq = create_sequences(X_val_np_global, input_window, output_window)

    # 处理无法创建序列的情况 (数据太短)
    if X_train_seq.shape[0] == 0 or X_val_seq.shape[0] == 0:
        print(f"  跳过试验 {trial.number}: 数据不足以创建 input_window={input_window} 的序列")
        return -1.0 # 对于最大化问题返回一个较差的值

    X_train_tensor = torch.tensor(X_train_seq, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_seq, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val_seq, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val_seq, dtype=torch.float32)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    # 使用 drop_last=True 可能使 PGD 训练更稳定
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # --- 模型配置 ---
    config = {
        'data_loader': {
            'args': {
                'time_step': input_window,
                'output_window': output_window,
                'series_num': P_global,
                'feature_dim': FEATURE_DIM,
                'output_dim': OUTPUT_DIM
            }
        },
        'device': DEVICE.type
    }

    # --- 运行训练和评估 ---
    # 注意：这里传递了 GC_true_np_global
    val_auroc, val_mse = train_and_evaluate(
        params, config, train_loader, val_loader, GC_true_np_global, DEVICE, EPOCHS
    )

    # --- Optuna 报告和剪枝 ---
    trial.report(val_auroc, EPOCHS - 1) # 报告最终的 AUROC
    if trial.should_prune():
        print(f"  试验 {trial.number} 被剪枝。")
        raise optuna.exceptions.TrialPruned()

    return val_auroc # 返回要最大化的值

# --- Optuna 执行函数 ---
def run_tuning(x_train, x_val, gc_true, p_val):
    global X_train_np_global, X_val_np_global, GC_true_np_global, P_global
    X_train_np_global = x_train
    X_val_np_global = x_val
    GC_true_np_global = gc_true
    P_global = p_val

    study = optuna.create_study(
        study_name=STUDY_NAME, storage=STORAGE_PATH, direction='maximize',
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=15, interval_steps=1)
    )
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    print(f"\n开始 Optuna 超参数优化")
    print(f"Study 名称: {STUDY_NAME}")
    print(f"试验次数: {N_TRIALS}")
    print(f"设备: {DEVICE}")

    try:
        study.optimize(objective, n_trials=N_TRIALS, timeout=None)
    except KeyboardInterrupt:
        print("用户中断优化。")
    except Exception as e:
        print(f"Optuna 优化过程中发生错误: {e}")
        traceback.print_exc()

    # --- 结果分析 ---
    print(f"\nOptuna 优化完成。总试验次数: {len(study.trials)}")
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    print(f"完成的试验次数: {len(completed_trials)}")

    best_params = None
    if completed_trials:
        try:
            best_trial = study.best_trial
            print(f"\n最佳试验:")
            print(f"  编号: {best_trial.number}")
            print(f"  值 (验证集 AUROC): {best_trial.value:.6f}")
            print(f"  最佳参数:")
            best_params = best_trial.params
            for key, value in best_params.items():
                print(f"    {key}: {value}")
            # 保存最佳参数
            import joblib
            joblib.dump(best_params, f"{STUDY_NAME}_best_params.pkl")
        except ValueError:
            print("\n无法确定最佳试验 (可能全部失败或被剪枝)。")
    else:
        print("\n没有成功完成的试验。")

     # --- 可视化 (仅使用 Matplotlib) ---
    if completed_trials:
        try:
            trial_numbers = [t.number for t in completed_trials]
            auroc_values = [t.value for t in completed_trials if t.value is not None] # 过滤 None 值
            valid_trial_numbers = [t.number for t in completed_trials if t.value is not None]

            if valid_trial_numbers:
                plt.figure(figsize=(12, 7))
                plt.scatter(valid_trial_numbers, auroc_values, alpha=0.6, label='完成的试验', s=50)
                # 检查 best_trial 是否已定义且有有效值
                if best_params is not None and 'best_trial' in locals() and best_trial.value is not None:
                    plt.scatter([best_trial.number], [best_trial.value], color='red', s=150, label=f'最佳试验 ({best_trial.number})', zorder=5, edgecolors='black')
                plt.xlabel("试验编号")
                plt.ylabel("验证集 AUROC")
                plt.title(f"Optuna 优化历史 ({STUDY_NAME}) - Matplotlib")
                plt.grid(True, linestyle='--', alpha=0.6)
                plt.legend()
                plt.tight_layout()
                plt.savefig(f"{STUDY_NAME}_matplotlib_history.png")
                print(f"\nMatplotlib 优化历史图已保存至 {STUDY_NAME}_matplotlib_history.png")
                # plt.show() # 取消注释以显示绘图
            else:
                 print("未找到有效的 AUROC 值用于绘制历史记录。")

        except Exception as e:
            print(f"生成 Matplotlib 可视化时出错: {e}")
            traceback.print_exc()
    else:
        print("没有完成的试验可供可视化。")

    return best_params
