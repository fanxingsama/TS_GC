# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, mean_squared_error
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import rcParams
import optuna
import os
import traceback
import math # 用于初始化

# --- 模型和工具函数导入 ---
from Granger_causalFormer import PredictModel
# 假设 TCN 相关文件在 model/TCN_granger/
from TCN_granger.data_create import simulate_var
from TCN_granger.granger_utils import (
    prox_group_lasso,
    prox_group_sparse_group_lasso,
    calculate_group_lasso_penalty,
    calculate_group_sparse_group_lasso_penalty
)

# 设置 Matplotlib 中文显示
rcParams['font.family'] = 'SimHei'
rcParams['axes.unicode_minus'] = False

# --- 数据模拟参数 ---
P = 5           # Number of time series (series_num)
T = 1000        # Total time points
LAG = 2         # True VAR lag
SPARSITY = 0.4  # Sparsity level for GC matrix
BETA_VALUE = 0.8# Coefficient value
SD = 0.1        # Standard deviation of noise
DATA_SEED = 42  # Random seed for reproducibility
FEATURE_DIM = 1 # Assuming univariate time series for simplicity with current model structure
OUTPUT_DIM = 1  # Assuming predicting the series value itself

# --- 训练和 Optuna 参数 ---
EPOCHS = 50 # PGD might need more epochs
BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_TRIALS = 50 # Number of Optuna trials
STUDY_NAME = "causalformer-grangertcn-pgd-study"
STORAGE_PATH = f"sqlite:///{STUDY_NAME}.db"
DEFAULT_OUTPUT_WINDOW = 1 # Predict next step

print(f"使用设备: {DEVICE}")

# --- 1. 生成并预处理数据 ---
print("正在生成和预处理数据...")
# X_np shape: [T, P]
X_np, _, GC_true_np = simulate_var(p=P, T=T, lag=LAG, sparsity=SPARSITY,
                                   beta_value=BETA_VALUE, sd=SD, seed=DATA_SEED)
print(f"原始数据 X: {X_np.shape}")
print(f"真实格兰杰因果矩阵 GC (前{P}x{P}):\n{GC_true_np}")

# 添加特征维度
if FEATURE_DIM == 1:
    X_np = X_np[:, :, np.newaxis] # Shape: [T, P, 1]
else:
    # 如果 FEATURE_DIM > 1, 需要相应地生成或处理数据
    raise NotImplementedError("Data generation for FEATURE_DIM > 1 not implemented here.")

# 分割数据
X_train_val_np, X_test_np = train_test_split(X_np, test_size=0.2, random_state=DATA_SEED, shuffle=False)
X_train_np, X_val_np = train_test_split(X_train_val_np, test_size=0.25, random_state=DATA_SEED, shuffle=False) # 60% train, 20% val, 20% test
print(f"训练集 X: {X_train_np.shape}")
print(f"验证集 X: {X_val_np.shape}")
print(f"测试集 X: {X_test_np.shape}")

# --- 辅助函数：创建序列 (适配 CausalFormer 输入输出) ---
def create_sequences(data, input_seq_len, output_seq_len):
    """
    创建适用于 CausalFormer 的序列数据。
    Args:
        data (np.array): 输入数据，形状 [Time, NumSeries, NumFeatures]。
        input_seq_len (int): 输入序列长度 (input_window)。
        output_seq_len (int): 输出序列长度 (output_window)。
    Returns:
        Tuple[np.array, np.array]: X (输入序列), Y (目标序列)
            X shape: [num_samples, input_seq_len, NumSeries, NumFeatures]
            Y shape: [num_samples, output_seq_len, NumSeries, NumFeatures]
    """
    xs, ys = [], []
    total_len = len(data)
    if total_len <= input_seq_len + output_seq_len -1: # 需要足够的数据点
        print(f"警告: 数据长度 {total_len} 不足以创建长度为 input={input_seq_len}, output={output_seq_len} 的序列。")
        return np.array(xs), np.array(ys)

    for i in range(total_len - input_seq_len - output_seq_len + 1):
        x = data[i:(i + input_seq_len)]
        y = data[(i + input_seq_len):(i + input_seq_len + output_seq_len)]
        xs.append(x)
        ys.append(y)

    if not xs:
        return np.array(xs), np.array(ys)
    return np.array(xs), np.array(ys)

# --- 2. 定义 Optuna 目标函数 (使用 PGD) ---
def objective(trial):
    global X_train_np, X_val_np, GC_true_np, P, FEATURE_DIM, OUTPUT_DIM

    # --- 超参数建议 ---
    # CausalFormer 参数
    input_window = trial.suggest_categorical('input_window', [10, 20, 30]) # 输入序列长度
    output_window = DEFAULT_OUTPUT_WINDOW # 固定预测下一步
    d_model = trial.suggest_categorical('d_model', [32, 64, 128])       # QK 嵌入维度
    n_head = trial.suggest_categorical('n_head', [2, 4, 8])             # 注意力头数
    n_layers = trial.suggest_int('n_layers', 1, 3)                      # Encoder 层数
    ffn_hidden = trial.suggest_categorical('ffn_hidden', [64, 128, 256])# FFN 隐藏层维度
    dropout = trial.suggest_float('dropout', 0.0, 0.3)                  # Dropout
    tau = trial.suggest_float('tau', 0.5, 10.0, log=True)               # Softmax 温度

    # GrangerTCN 参数 (集成在 CausalFormer 内)
    tcn_layers = trial.suggest_int('tcn_layers', 2, 5)                  # TCN 块数
    tcn_channels = trial.suggest_categorical('tcn_channels', [16, 32, 48]) # TCN 通道数
    tcn_kernel_size = trial.suggest_categorical('tcn_kernel_size', [2, 3, 4]) # TCN 核大小
    tcn_dropout = trial.suggest_float('tcn_dropout', 0.0, 0.3)          # TCN Dropout
    tcn_channel_list = [tcn_channels] * tcn_layers

    # PGD 和稀疏性参数
    lr = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)     # 学习率
    lambda_reg = trial.suggest_float('lambda_reg', 1e-5, 1e-1, log=True)# 正则化强度
    penalty_type = trial.suggest_categorical('penalty_type', ['GL', 'GSGL']) # 惩罚类型
    alpha_gsgl = 0.5 # Default for GSGL
    if penalty_type == 'GSGL':
        alpha_gsgl = trial.suggest_float('alpha_gsgl', 0.1, 0.9)

    print(f"\n--- Trial {trial.number} ---")
    print(f"  CausalFormer Params: input_window={input_window}, d_model={d_model}, n_head={n_head}, n_layers={n_layers}, ffn_hidden={ffn_hidden}, dropout={dropout:.3f}, tau={tau:.3f}")
    print(f"  GrangerTCN Params: layers={tcn_layers}, channels={tcn_channels}, kernel={tcn_kernel_size}, dropout={tcn_dropout:.3f}")
    print(f"  Optimization Params: lr={lr:.6f}, lambda_reg={lambda_reg:.6f}, penalty={penalty_type}"
          f"{f', alpha={alpha_gsgl:.2f}' if penalty_type == 'GSGL' else ''}")

    # --- 数据准备 ---
    X_train_seq, y_train_seq = create_sequences(X_train_np, input_window, output_window)
    X_val_seq, y_val_seq = create_sequences(X_val_np, input_window, output_window)

    if X_train_seq.size == 0 or X_val_seq.size == 0:
        print(f"Trial {trial.number} skipped: Not enough data for sequence length {input_window}.")
        return -1.0 # Return a bad value for maximization problem

    X_train_tensor = torch.tensor(X_train_seq, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_seq, dtype=torch.float32) # Shape: [N, T_out, P, F_out]
    X_val_tensor = torch.tensor(X_val_seq, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val_seq, dtype=torch.float32)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    # Drop last to ensure consistent batch sizes for PGD state, could be removed if handled carefully
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # --- 模型和损失函数 ---
    # 创建配置字典传递给 PredictModel
    config = {
        'data_loader': {'args': {
            'time_step': input_window,
            'output_window': output_window,
            'series_num': P,
            'feature_dim': FEATURE_DIM,
            'output_dim': OUTPUT_DIM
        }},
        'model': {'args': {}}, # 可以添加其他模型相关配置
        'device': DEVICE.type # 传递设备类型
    }

    model = PredictModel(config=config,
                         d_model=d_model,
                         n_head=n_head,
                         tcn_channels=tcn_channel_list,
                         tcn_kernel_size=tcn_kernel_size,
                         tcn_dropout=tcn_dropout,
                         n_layers=n_layers,
                         ffn_hidden=ffn_hidden,
                         drop_prob=dropout,
                         tau=tau).to(DEVICE)

    criterion = nn.MSELoss()

    # --- PGD 训练循环 ---
    final_val_auroc = 0.0
    final_avg_val_mse = float('inf')

    # 查找需要正则化的参数对象 (只正则化第一个 EncoderLayer 的 TCN 的第一个 conv1)
    granger_weights_param = None
    try:
        # 这个路径取决于 PredictModel -> Encoder -> EncoderLayer -> MultiHeadAttention -> GrangerTCN -> TemporalBlock -> conv1
        granger_weights_param = model.encoder.layers[0].attention.tcn_processor.network_layers[0].conv1.weight
        print("成功定位到 Granger TCN 权重进行正则化。")
    except AttributeError:
        print("警告：无法自动定位 Granger TCN 权重，将不应用稀疏惩罚。请检查模型结构和 get_granger_weights() 方法。")
        # 如果找不到，PGD 将退化为标准梯度下降

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        epoch_penalty = 0.0
        num_batches = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)

            # 1. 计算主损失和梯度
            model.zero_grad()
            predictions = model(batch_x) # Shape: [B, T_out, P, F_out]
            # 确保 target 形状匹配 prediction
            main_loss = criterion(predictions, batch_y)
            main_loss.backward() # 计算所有参数的梯度

            epoch_loss += main_loss.item()
            num_batches += 1

            # --- 手动执行 PGD 更新 ---
            with torch.no_grad():
                # 计算当前批次的惩罚值 (用于记录)
                current_penalty = torch.tensor(0.0, device=DEVICE)
                if granger_weights_param is not None:
                    current_weights = granger_weights_param.data # 获取当前权重数据
                    if penalty_type == 'GL':
                        current_penalty = calculate_group_lasso_penalty(current_weights, lambda_reg)
                    elif penalty_type == 'GSGL':
                        current_penalty = calculate_group_sparse_group_lasso_penalty(current_weights, lambda_reg, alpha_gsgl)
                epoch_penalty += current_penalty.item()


                # 更新参数
                for name, param in model.named_parameters():
                    if param.grad is None: continue # 跳过没有梯度的参数

                    # 检查是否是需要正则化的权重
                    is_regularized_weight = (param is granger_weights_param)

                    if is_regularized_weight:
                        # 应用 PGD 更新
                        w_tilde = param.data - lr * param.grad # W - step * grad(L_mse)
                        lambda_gamma = lr * lambda_reg         # step * lambda
                        w_new = torch.zeros_like(w_tilde)      # 初始化新权重

                        if penalty_type == 'GL':
                            # Prox for Group Lasso (Frobenius norm per input feature group)
                            # w_tilde shape: [out_ch, in_ch=P, kernel_size]
                            for j in range(w_tilde.shape[1]): # Iterate over input features (P)
                                w_new[:, j, :] = prox_group_lasso(w_tilde[:, j, :], lambda_gamma)
                        elif penalty_type == 'GSGL':
                            # Prox for Group Sparse Group Lasso
                            for j in range(w_tilde.shape[1]):
                                w_new[:, j, :] = prox_group_sparse_group_lasso(w_tilde[:, j, :], lambda_gamma, alpha_gsgl)
                        else: # Should not happen with categorical suggestion
                            w_new = w_tilde # Fallback to gradient descent if penalty unknown

                        param.copy_(w_new) # 更新参数
                    else:
                        # 对其他参数执行标准梯度下降
                        param.copy_(param.data - lr * param.grad)

        avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0
        avg_epoch_penalty = epoch_penalty / num_batches if num_batches > 0 else 0
        # print(f"  Epoch {epoch+1}/{EPOCHS}, Avg Train Loss: {avg_epoch_loss:.6f}, Avg Penalty: {avg_epoch_penalty:.6f}")

        # --- 验证 ---
        model.eval()
        val_mse = 0.0
        learned_norms = np.zeros(P) # 每个输入特征的范数
        current_val_auroc = 0.0
        current_val_aupr = 0.0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
                predictions = model(batch_x)
                loss = criterion(predictions, batch_y)
                val_mse += loss.item() * batch_x.size(0)

            if len(val_loader.dataset) > 0:
                avg_val_mse = val_mse / len(val_loader.dataset)
            else:
                avg_val_mse = float('inf')

            # 获取训练后的 Granger TCN 权重
            final_weights = model.get_granger_weights(layer_index=0) # 获取第一个 encoder layer 的 TCN 权重
            if final_weights is not None:
                for j in range(P): # P is num_input_features
                    try:
                        norm_val = torch.linalg.norm(final_weights[:, j, :].float(), ord='fro').cpu().item()
                        learned_norms[j] = norm_val if np.isfinite(norm_val) else 0.0
                    except Exception as e_norm:
                        learned_norms[j] = 0.0
            else:
                learned_norms = np.zeros(P)

            # --- 计算 AUROC 和 AUPR ---
            # 真实标签: feature j 是否导致了 *任何* 其他 feature i (GC_true_np[i, j] == 1 for any i != j)
            true_causes = np.array([np.any(GC_true_np[np.arange(P) != j, j] == 1) for j in range(P)], dtype=int)
            scores = learned_norms # 分数是每个输入特征的权重范数

            if len(np.unique(true_causes)) > 1 and not np.all(np.isnan(scores)) and len(np.unique(scores)) > 1:
                try:
                    valid_indices = ~np.isnan(scores)
                    if np.any(valid_indices):
                        current_val_auroc = roc_auc_score(true_causes[valid_indices], scores[valid_indices])
                        current_val_aupr = average_precision_score(true_causes[valid_indices], scores[valid_indices])
                    else:
                        current_val_auroc, current_val_aupr = 0.0, 0.0
                except ValueError as e_auc:
                    # print(f"计算 AUROC/AUPR 时出错: {e_auc}")
                    current_val_auroc, current_val_aupr = 0.0, 0.0
            else:
                current_val_auroc, current_val_aupr = 0.0, 0.0

        final_val_auroc = current_val_auroc
        final_avg_val_mse = avg_val_mse

        # --- Optuna 剪枝 (使用 AUROC) ---
        trial.report(final_val_auroc, epoch)
        if trial.should_prune():
            print(f"Trial {trial.number} pruned at epoch {epoch+1}.")
            return -1.0 # Return bad value for maximization

    print(f"Trial {trial.number} finished. Final Val AUROC: {final_val_auroc:.4f}, Final Val MSE: {final_avg_val_mse:.6f}")
    # 返回最终验证集 AUROC 给 Optuna
    return final_val_auroc if np.isfinite(final_val_auroc) else -1.0

# --- 3. 创建或加载 Optuna Study 并运行优化 ---
study = optuna.create_study(
    study_name=STUDY_NAME, storage=STORAGE_PATH, direction='maximize',
    load_if_exists=True,
    pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=15, interval_steps=1) # 调整剪枝参数
)
optuna.logging.set_verbosity(optuna.logging.WARNING)

print(f"\n开始 Optuna 超参数优化 (目标: 最大化验证集 AUROC 使用 PGD)...")
try:
    study.optimize(objective, n_trials=N_TRIALS, timeout=None)
except KeyboardInterrupt:
    print("优化被手动中断。")
except Exception as e:
    print(f"Optuna 优化过程中发生错误: {e}")
    traceback.print_exc()

# --- 4. 输出结果 ---
print("\nOptuna 优化完成。")
print(f"总尝试次数: {len(study.trials)}")

completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
print(f"成功完成的 Trial 数量: {len(completed_trials)}")

best_params = None
if completed_trials:
    try:
        best_trial = study.best_trial
        print(f"\n最佳 Trial:")
        print(f"  编号: {best_trial.number}")
        print(f"  目标值 (Val AUROC): {best_trial.value:.6f}")
        print(f"  最佳超参数:")
        best_params = best_trial.params
        for key, value in best_params.items():
            print(f"    {key}: {value}")
        # joblib.dump(best_params, f"{STUDY_NAME}_best_params.pkl") # 保存最佳参数
    except ValueError:
        print("\n无法确定最佳 Trial (可能所有 Trial 都失败或被剪枝)。")
else:
    print("没有找到完成的 Trial。")

# --- 5. 可视化优化过程 ---
if completed_trials:
    try:
        # --- 使用 Matplotlib 绘制优化历史 ---
        trial_numbers = [t.number for t in completed_trials]
        auroc_values = [t.value for t in completed_trials if t.value is not None] # 过滤掉 None 值
        # 确保 trial_numbers 和 auroc_values 长度匹配
        valid_trial_numbers = [t.number for t in completed_trials if t.value is not None]

        if not valid_trial_numbers:
             print("没有找到有效的 AUROC 值来绘制优化历史。")
        else:
            plt.figure(figsize=(12, 7))
            plt.scatter(valid_trial_numbers, auroc_values, alpha=0.6, label='完成的 Trials', s=50)

            # 标记最佳 trial (如果存在且值有效)
            if best_params is not None and 'best_trial' in locals() and best_trial.value is not None:
                plt.scatter([best_trial.number], [best_trial.value], color='red', s=150, label=f'最佳 Trial ({best_trial.number})', zorder=5, edgecolors='black')

            plt.xlabel("Trial 编号")
            plt.ylabel("验证集 AUROC")
            plt.title(f"Optuna 优化历史 ({STUDY_NAME}) - Matplotlib")
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"{STUDY_NAME}_matplotlib_history.png")
            print(f"\n使用 Matplotlib 绘制的优化历史图已保存到 {STUDY_NAME}_matplotlib_history.png")
            # plt.show() # 如果需要显示图像，取消注释

    except Exception as e:
        print(f"生成可视化图像时出错: {e}")
        traceback.print_exc() # 打印详细错误
else:
     print("没有完成的 Trial 可供可视化。")


# # --- 6. (可选) 使用最佳参数在完整训练集上训练并评估测试集 ---
# if best_params is not None:
#     print("\n使用最佳参数重新训练模型并在测试集上评估...")
#     # --- 数据准备 ---
#     X_train_full_np = np.concatenate((X_train_np, X_val_np), axis=0) # 合并训练集和验证集
#     input_window = best_params['input_window']
#     output_window = DEFAULT_OUTPUT_WINDOW

#     X_train_full_seq, y_train_full_seq = create_sequences(X_train_full_np, input_window, output_window)
#     X_test_seq, y_test_seq = create_sequences(X_test_np, input_window, output_window)

#     if X_train_full_seq.size > 0 and X_test_seq.size > 0:
#         X_train_full_tensor = torch.tensor(X_train_full_seq, dtype=torch.float32)
#         y_train_full_tensor = torch.tensor(y_train_full_seq, dtype=torch.float32)
#         X_test_tensor = torch.tensor(X_test_seq, dtype=torch.float32)
#         y_test_tensor = torch.tensor(y_test_seq, dtype=torch.float32)

#         train_full_dataset = TensorDataset(X_train_full_tensor, y_train_full_tensor)
#         test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
#         train_full_loader = DataLoader(train_full_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
#         test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

#         # --- 创建最终模型 ---
#         final_config = {
#             'data_loader': {'args': {
#                 'time_step': input_window, 'output_window': output_window,
#                 'series_num': P, 'feature_dim': FEATURE_DIM, 'output_dim': OUTPUT_DIM
#             }},
#             'model': {'args': {}},
#             'device': DEVICE.type
#         }
#         final_tcn_channel_list = [best_params['tcn_channels']] * best_params['tcn_layers']
#         final_model = PredictModel(config=final_config,
#                                    d_model=best_params['d_model'], n_head=best_params['n_head'],
#                                    tcn_channels=final_tcn_channel_list, tcn_kernel_size=best_params['tcn_kernel_size'],
#                                    tcn_dropout=best_params['tcn_dropout'], n_layers=best_params['n_layers'],
#                                    ffn_hidden=best_params['ffn_hidden'], drop_prob=best_params['dropout'],
#                                    tau=best_params['tau']).to(DEVICE)

#         final_criterion = nn.MSELoss()
#         final_lr = best_params['learning_rate']
#         final_lambda_reg = best_params['lambda_reg']
#         final_penalty_type = best_params['penalty_type']
#         final_alpha_gsgl = best_params.get('alpha_gsgl', 0.5)

#         # 定位最终模型的正则化权重
#         final_granger_weights_param = None
#         try:
#             final_granger_weights_param = final_model.encoder.layers[0].attention.tcn_processor.network_layers[0].conv1.weight
#         except AttributeError:
#              print("警告：无法定位最终模型的 Granger TCN 权重，训练将不应用稀疏惩罚。")


#         # --- 训练最终模型 ---
#         print(f"在完整训练集上训练 {EPOCHS * 2} 个 epochs...") # 训练更长时间
#         for epoch in tqdm(range(EPOCHS * 2), desc="Final Training"):
#             final_model.train()
#             for batch_x, batch_y in train_full_loader:
#                 batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
#                 final_model.zero_grad()
#                 predictions = final_model(batch_x)
#                 main_loss = final_criterion(predictions, batch_y)
#                 main_loss.backward()

#                 # 手动 PGD 更新
#                 with torch.no_grad():
#                     for name, param in final_model.named_parameters():
#                         if param.grad is None: continue
#                         is_regularized = (param is final_granger_weights_param)
#                         if is_regularized:
#                             w_tilde = param.data - final_lr * param.grad
#                             lambda_gamma = final_lr * final_lambda_reg
#                             w_new = torch.zeros_like(w_tilde)
#                             if final_penalty_type == 'GL':
#                                 for j in range(w_tilde.shape[1]):
#                                     w_new[:, j, :] = prox_group_lasso(w_tilde[:, j, :], lambda_gamma)
#                             elif final_penalty_type == 'GSGL':
#                                 for j in range(w_tilde.shape[1]):
#                                     w_new[:, j, :] = prox_group_sparse_group_lasso(w_tilde[:, j, :], lambda_gamma, final_alpha_gsgl)
#                             else:
#                                 w_new = w_tilde
#                             param.copy_(w_new)
#                         else:
#                             param.copy_(param.data - final_lr * param.grad)

#         # --- 在测试集上评估 ---
#         final_model.eval()
#         test_mse = 0.0
#         test_norms = np.zeros(P)
#         test_predictions_list = []
#         test_targets_list = []
#         with torch.no_grad():
#             for batch_x, batch_y in test_loader:
#                 batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
#                 predictions = final_model(batch_x)
#                 loss = final_criterion(predictions, batch_y)
#                 test_mse += loss.item() * batch_x.size(0)
#                 test_predictions_list.append(predictions.cpu().numpy())
#                 test_targets_list.append(batch_y.cpu().numpy())

#             if len(test_loader.dataset) > 0: avg_test_mse = test_mse / len(test_loader.dataset)
#             else: avg_test_mse = float('inf')

#             final_test_weights = final_model.get_granger_weights(layer_index=0)
#             if final_test_weights is not None:
#                 for j in range(P):
#                     try:
#                         norm_val = torch.linalg.norm(final_test_weights[:, j, :].float(), ord='fro').cpu().item()
#                         test_norms[j] = norm_val if np.isfinite(norm_val) else 0.0
#                     except Exception: test_norms[j] = 0.0
#             else: test_norms = np.zeros(P)

#             # 计算测试集 AUROC 和 AUPR
#             test_true_causes = np.array([np.any(GC_true_np[np.arange(P) != j, j] == 1) for j in range(P)], dtype=int)
#             test_scores = test_norms

#             test_auroc = 0.0
#             test_aupr = 0.0
#             if len(np.unique(test_true_causes)) > 1 and not np.all(np.isnan(test_scores)) and len(np.unique(test_scores)) > 1:
#                 try:
#                     valid_indices = ~np.isnan(test_scores)
#                     if np.any(valid_indices):
#                         test_auroc = roc_auc_score(test_true_causes[valid_indices], test_scores[valid_indices])
#                         test_aupr = average_precision_score(test_true_causes[valid_indices], test_scores[valid_indices])
#                     else: test_auroc, test_aupr = 0.0, 0.0
#                 except ValueError: test_auroc, test_aupr = 0.0, 0.0
#             else: test_auroc, test_aupr = 0.0, 0.0

#         print("\n--- 测试集评估结果 ---")
#         print(f"  测试集 MSE: {avg_test_mse:.6f}")
#         print(f"  测试集 AUROC (基于权重范数): {test_auroc:.4f}")
#         print(f"  测试集 AUPR (基于权重范数): {test_aupr:.4f}")
#         print(f"  测试集最终权重范数:")
#         for idx, norm_val in enumerate(test_norms):
#             print(f"    特征 {idx}: {norm_val:.4f}")

#         # (可选) 保存最终模型
#         # torch.save(final_model.state_dict(), f"{STUDY_NAME}_final_model.pth")
#         # print(f"最终模型已保存到 {STUDY_NAME}_final_model.pth")
#     else:
#         print("数据不足，无法在测试集上评估。")
# else:
#     print("未找到最佳参数，跳过最终训练和测试。")

print("\n脚本执行完毕。")
