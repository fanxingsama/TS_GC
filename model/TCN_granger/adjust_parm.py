# -*- coding: utf-8 -*-
from matplotlib import rcParams
import torch
import torch.nn as nn
# 不再需要 optim，因为我们手动实现 PGD
# import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, mean_squared_error
from tqdm import tqdm
import matplotlib.pyplot as plt # 导入 Matplotlib
import optuna # 导入 Optuna
import joblib # 用于保存/加载 study
import os
import traceback # 用于打印详细错误


from data_create import simulate_var
from granger_tcn_model import GrangerTCN
# 导入近端算子和惩罚计算函数
from granger_utils import prox_group_lasso, prox_group_sparse_group_lasso, calculate_group_lasso_penalty, calculate_group_sparse_group_lasso_penalty

# 设置字体为SimHei，这是Windows系统常用的中文字体
rcParams['font.family'] = 'SimHei'
rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# --- simulate_var参数 ---
P = 5
T = 1000
LAG = 2
SPARSITY = 0.4
BETA_VALUE = 0.8
SD = 0.1
DATA_SEED = 42

# --- 训练参数 ---
EPOCHS = 30 # PGD 可能需要更多 epochs 来收敛，可以适当增加
BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_TRIALS = 50 # 保持 Optuna 尝试次数
STUDY_NAME = "granger-tcn-pgd-study-matplotlib" # 使用新名字区分 PGD 实验
STORAGE_PATH = f"sqlite:///{STUDY_NAME}.db"

print(f"使用设备: {DEVICE}")

# --- 1. 生成并预处理数据 ---
print("正在生成和预处理数据...")
X_np, _, GC_true_np = simulate_var(p=P, T=T, lag=LAG, sparsity=SPARSITY,
                                   beta_value=BETA_VALUE, sd=SD, seed=DATA_SEED)
print(f"原始数据 X: {X_np.shape}")
print(f"真实格兰杰因果矩阵 GC (前5x5):\n{GC_true_np[:5, :5]}")
# 保持与之前一致的分割比例
X_train_val_np, X_test_np = train_test_split(X_np, test_size=0.2, random_state=DATA_SEED, shuffle=False)
X_train_np, X_val_np = train_test_split(X_train_val_np, test_size=0.25, random_state=DATA_SEED, shuffle=False) # 0.8 * 0.25 = 0.2
print(f"训练集 X: {X_train_np.shape}") # 60%
print(f"验证集 X: {X_val_np.shape}") # 20%
print(f"测试集 X: {X_test_np.shape}") # 20%

# --- 辅助函数：创建序列 ---
def create_sequences(data, seq_length):
    xs, ys = [], []
    if len(data) <= seq_length:
        print(f"警告: 数据长度 {len(data)} 不足以创建长度为 {seq_length} 的序列。")
        return np.array(xs), np.array(ys)
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length] # 预测下一个时间点的值
        xs.append(x)
        ys.append(y)
    if not xs:
        return np.array(xs), np.array(ys)
    return np.array(xs), np.array(ys)

# --- 2. 定义 Optuna 目标函数 (使用 PGD) ---
def objective(trial):
    global X_train_np, X_val_np, GC_true_np
    # --- 超参数建议 ---
    # 序列长度
    seq_length = trial.suggest_categorical('seq_length', [10, 20, 30, 40]) # 可以尝试更多选项
    # 学习率 (PGD 可能需要不同的范围)
    lr = trial.suggest_float('learning_rate', 5e-5, 5e-3, log=True) # 调整范围以适应 PGD
    # 正则化强度
    lambda_reg = trial.suggest_float('lambda_reg', 1e-4, 5e-2, log=True) # 调整范围
    # 惩罚类型
    penalty_type = trial.suggest_categorical('penalty_type', ['GL', 'GSGL'])
    alpha_gsgl = 0.5 # 默认值
    if penalty_type == 'GSGL':
        alpha_gsgl = trial.suggest_float('alpha_gsgl', 0.1, 0.9)
    # TCN 结构参数
    num_layers = trial.suggest_int('num_layers', 2, 5) # 增加层数选项
    num_channels = trial.suggest_categorical('num_channels', [16, 32, 48]) # 调整通道数选项
    channel_list = [num_channels] * num_layers
    kernel_size = trial.suggest_categorical('kernel_size', [2, 3, 4, 5]) # 增加核大小选项
    dropout = trial.suggest_float('dropout', 0.0, 0.5) # 增加 dropout 范围

    print(f"\n--- Trial {trial.number} ---")
    print(f"  Params: seq_length={seq_length}, lr={lr:.6f}, lambda_reg={lambda_reg:.6f}, penalty={penalty_type}"
          f"{f', alpha={alpha_gsgl:.2f}' if penalty_type == 'GSGL' else ''}, layers={num_layers}, channels={num_channels}, "
          f"kernel={kernel_size}, dropout={dropout:.3f}")

    # --- 数据准备 ---
    X_train_seq, y_train_seq = create_sequences(X_train_np, seq_length)
    X_val_seq, y_val_seq = create_sequences(X_val_np, seq_length)

    # 如果数据不足以创建任何序列，则返回一个差值
    if X_train_seq.size == 0 or X_val_seq.size == 0:
        print(f"Trial {trial.number} skipped: Not enough data for sequence length {seq_length}.")
        return -1.0 # 返回负值或 NaN 表示失败/跳过

    X_train_tensor = torch.tensor(X_train_seq, dtype=torch.float32)
    # y 应该是 (batch, output_size)
    y_train_tensor = torch.tensor(y_train_seq, dtype=torch.float32).view(-1, P) # 确保 y 的形状正确
    X_val_tensor = torch.tensor(X_val_seq, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val_seq, dtype=torch.float32).view(-1, P) # 确保 y 的形状正确

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True) # drop_last 可能有助于稳定训练
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # --- 模型和损失函数 ---
    model = GrangerTCN(input_size=P, output_size=P, num_channels_list=channel_list,
                       kernel_size=kernel_size, dropout=dropout).to(DEVICE)
    criterion = nn.MSELoss()

    # --- PGD 训练循环 ---
    final_val_auroc = 0.0 # 初始化最终 AUROC
    final_avg_val_mse = float('inf') # 初始化最终 MSE

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)

            # 1. 计算损失和梯度 (只计算 MSE 的梯度)
            model.zero_grad() # 清除之前的梯度
            predictions = model(batch_x)
            main_loss = criterion(predictions, batch_y)
            main_loss.backward() # 只计算 MSE 损失的梯度

            epoch_loss += main_loss.item()
            num_batches += 1

            # --- 手动执行 PGD 更新 ---
            with torch.no_grad(): # 禁用梯度计算以进行参数更新
                # 获取第一个块 conv1 的权重参数对象
                first_block_conv1_weight_param = None
                first_block_conv1_bias_param = None
                if len(model.network_layers) > 0 and hasattr(model.network_layers[0], 'conv1'):
                    first_block_conv1_weight_param = model.network_layers[0].conv1.weight
                    if model.network_layers[0].conv1.bias is not None:
                         first_block_conv1_bias_param = model.network_layers[0].conv1.bias

                for name, param in model.named_parameters():
                    if param.grad is None: # 如果某个参数没有梯度（可能因为没用到），跳过
                        continue

                    # 检查是否是需要正则化的权重
                    is_regularized_weight = (param is first_block_conv1_weight_param)

                    if is_regularized_weight:
                        # 对第一个块的 conv1 权重应用 PGD
                        # W_tilde = W - lr * grad(L_mse)
                        w_tilde = param - lr * param.grad
                        # W_new = prox_{lr * lambda * Penalty}(W_tilde)
                        lambda_gamma = lr * lambda_reg # lambda * step_size
                        if penalty_type == 'GL':
                            # Prox 需要 [out_channels, kernel_size] 形状
                            # 我们需要对每个 input_channel (j) 分别应用 prox
                            w_new = torch.zeros_like(w_tilde)
                            for j in range(w_tilde.shape[1]): # 遍历 input channels
                                w_new[:, j, :] = prox_group_lasso(w_tilde[:, j, :], lambda_gamma)
                        elif penalty_type == 'GSGL':
                            w_new = torch.zeros_like(w_tilde)
                            for j in range(w_tilde.shape[1]): # 遍历 input channels
                                w_new[:, j, :] = prox_group_sparse_group_lasso(w_tilde[:, j, :], lambda_gamma, alpha_gsgl)
                        else: # 如果没有惩罚类型或类型错误，则执行标准梯度下降
                            w_new = w_tilde
                        param.copy_(w_new) # 更新参数
                    else:
                        # 对其他参数（包括第一个块 conv1 的偏置和其他层的权重偏置）执行标准梯度下降
                        param.copy_(param - lr * param.grad)

        avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0
        # print(f"  Epoch {epoch+1}/{EPOCHS}, Avg Train Loss: {avg_epoch_loss:.6f}") # 训练时可以取消注释

        # --- 验证 ---
        model.eval()
        val_mse = 0.0
        learned_norms = np.zeros(P) # 每个输入特征的范数
        current_val_auroc = 0.0 # 当前 epoch 的 AUROC
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
                predictions = model(batch_x)
                loss = criterion(predictions, batch_y)
                val_mse += loss.item() * batch_x.size(0) # 累加总误差

            # 计算平均验证 MSE
            if len(val_loader.dataset) > 0:
                avg_val_mse = val_mse / len(val_loader.dataset)
            else:
                avg_val_mse = float('inf') # 避免除以零

            # 获取训练后的第一个块 conv1 权重
            final_weights = model.get_first_block_conv1_weights()
            if final_weights is not None:
                # 计算每个输入特征对应的权重组的 Frobenius 范数
                for j in range(P): # P是输入特征的数量
                    try:
                        # 确保使用 float() 以防权重是半精度等
                        norm_val = torch.linalg.norm(final_weights[:, j, :].float(), ord='fro').cpu().item()
                        # 处理可能的 NaN 或 Inf 值
                        learned_norms[j] = norm_val if np.isfinite(norm_val) else 0.0
                    except Exception as e_norm:
                        # print(f"计算范数时出错 (特征 {j}): {e_norm}") # 调试时可以取消注释
                        learned_norms[j] = 0.0 # 出错时设为0
            else:
                learned_norms = np.zeros(P) # 如果没有权重，范数为0

            # --- 计算 AUROC ---
            # 准备真实标签 (j 是否导致了 i, i!=j) 和分数 (learned_norms)
            # 注意: 这里我们评估的是模型是否能识别出 *哪些输入特征* 对 *任何其他输出* 有影响
            # 这与论文中 cMLP/cLSTM 直接评估 i->j 不同，但对于 TCN 是一种合理的替代评估
            true_causes = []
            scores = learned_norms # 分数是每个输入特征的权重范数

            # 确定每个输入变量 j 是否是至少一个其他变量 i (i!=j) 的原因
            for j in range(P):
                # 检查 GC_true_np 中第 j 列（除了对角线元素）是否有 1
                is_j_a_cause = np.any(GC_true_np[:, j][np.arange(P) != j] == 1)
                true_causes.append(int(is_j_a_cause))
            true_causes = np.array(true_causes)

            # 仅在标签有两种类别且分数有效时计算 AUROC
            if len(np.unique(true_causes)) > 1 and not np.all(np.isnan(scores)) and len(np.unique(scores)) > 1:
                try:
                    # 过滤掉 NaN 分数（虽然前面处理过，但以防万一）
                    valid_indices = ~np.isnan(scores)
                    if np.any(valid_indices): # 确保至少有一个有效分数
                        current_val_auroc = roc_auc_score(true_causes[valid_indices], scores[valid_indices])
                    else:
                        current_val_auroc = 0.0 # 如果没有有效分数
                except ValueError as e_auc:
                    # print(f"计算 AUROC 时出错: {e_auc}") # 调试时可以取消注释
                    current_val_auroc = 0.0 # 计算错误时设为0
            else:
                # print("无法计算 AUROC：标签只有一类或分数无效/单一。") # 调试时可以取消注释
                current_val_auroc = 0.0 # 无法计算时设为0

        # 更新最终结果 (记录最后一个 epoch 的值)
        final_val_auroc = current_val_auroc
        final_avg_val_mse = avg_val_mse

        # --- Optuna 剪枝 ---
        trial.report(final_val_auroc, epoch) # 使用 AUROC 进行报告
        if trial.should_prune():
            print(f"Trial {trial.number} pruned at epoch {epoch+1}.")
            # 对于最大化问题，返回一个非常差的值（例如负无穷或一个大负数）
            # Optuna 默认处理 NaN 或 None 作为失败
            # 返回一个明确的差值可能更清晰
            return -1.0 # 返回负值表示剪枝

    print(f"Trial {trial.number} finished. Final Val AUROC: {final_val_auroc:.4f}, Final Val MSE: {final_avg_val_mse:.6f}")
    # 返回最终验证集 AUROC 给 Optuna
    return final_val_auroc if np.isfinite(final_val_auroc) else -1.0 # 确保返回值是有限的

# --- 3. 创建或加载 Optuna Study 并运行优化 ---
study = optuna.create_study(
    study_name=STUDY_NAME, storage=STORAGE_PATH, direction='maximize', # 目标是最大化 AUROC
    load_if_exists=True, # 如果存在同名 study，则加载它
    pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10, interval_steps=1) # 调整剪枝器参数
)
# 设置日志级别，减少不必要的输出
optuna.logging.set_verbosity(optuna.logging.WARNING)

print(f"\n开始 Optuna 超参数优化 (目标: 最大化验证集 AUROC 使用 PGD)...")
try:
    # 运行优化，n_trials 是尝试的总次数
    study.optimize(objective, n_trials=N_TRIALS, timeout=None) # timeout=None 表示没有时间限制
except KeyboardInterrupt:
    print("优化被手动中断。")
except Exception as e:
    print(f"Optuna 优化过程中发生错误: {e}")
    traceback.print_exc() # 打印详细的错误堆栈

# --- 4. 输出结果 ---
print("\nOptuna 优化完成。")
print(f"总尝试次数: {len(study.trials)}")

# 筛选出成功完成的 trials
completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
print(f"成功完成的 Trial 数量: {len(completed_trials)}")

best_params = None # 初始化 best_params
if completed_trials:
    try:
        best_trial = study.best_trial
        print(f"\n最佳 Trial:")
        print(f"  编号: {best_trial.number}")
        print(f"  目标值 (Val AUROC): {best_trial.value:.6f}")
        print(f"  最佳超参数:")
        best_params = best_trial.params # 保存最佳参数
        for key, value in best_params.items():
            print(f"    {key}: {value}")
        # 保存最佳参数到文件
        # joblib.dump(best_params, f"{STUDY_NAME}_best_params.pkl")
    except ValueError:
        print("\n无法确定最佳 Trial (可能所有 Trial 都失败或被剪枝)。")
else:
    print("没有找到完成的 Trial。")

# Matplotlib可视化优化过程
if completed_trials:
    trial_numbers = [t.number for t in completed_trials]
    auroc_values = [t.value for t in completed_trials] # 获取每个 trial 的最终 AUROC

    plt.figure(figsize=(12, 7)) # 调整图像大小
    plt.scatter(trial_numbers, auroc_values, alpha=0.6, label='完成的 Trials', s=50) # 调整透明度和大小

    # 标记最佳 trial
    if best_params is not None and 'best_trial' in locals(): # 确保 best_trial 存在
        best_trial_num = best_trial.number
        best_auroc = best_trial.value
        plt.scatter([best_trial_num], [best_auroc], color='red', s=150, label=f'最佳 Trial ({best_trial_num})', zorder=5, edgecolors='black') # 突出显示最佳点

    plt.xlabel("Trial 编号")
    plt.ylabel("验证集 AUROC")
    plt.title(f"Optuna 优化历史 ({STUDY_NAME})")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout() # 调整布局防止标签重叠
    plt.savefig(f"{STUDY_NAME}_matplotlib_history.png") # 保存图像
    print(f"\n使用 Matplotlib 绘制的优化历史图已保存到 {STUDY_NAME}_matplotlib_history.png")
    # plt.show() # 如果需要在脚本运行时显示图像，取消此行注释，但可能导致脚本挂起直到关闭图像窗口

# --- 6. (可选) 使用最佳参数在完整训练集上训练并评估测试集 ---
# if best_params is not None:
#     print("\n使用最佳参数重新训练模型并在测试集上评估...")
#     # 1. 准备完整训练数据 (合并 train 和 val) 和测试数据
#     X_train_full_np = X_train_val_np
#     X_test_seq, y_test_seq = create_sequences(X_test_np, best_params['seq_length'])
#     X_train_full_seq, y_train_full_seq = create_sequences(X_train_full_np, best_params['seq_length'])
#
#     if X_train_full_seq.size > 0 and X_test_seq.size > 0:
#         X_train_full_tensor = torch.tensor(X_train_full_seq, dtype=torch.float32)
#         y_train_full_tensor = torch.tensor(y_train_full_seq, dtype=torch.float32).view(-1, P)
#         X_test_tensor = torch.tensor(X_test_seq, dtype=torch.float32)
#         y_test_tensor = torch.tensor(y_test_seq, dtype=torch.float32).view(-1, P)
#
#         train_full_dataset = TensorDataset(X_train_full_tensor, y_train_full_tensor)
#         test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
#         train_full_loader = DataLoader(train_full_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
#         test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
#
#         # 2. 使用最佳参数创建模型
#         final_model = GrangerTCN(input_size=P, output_size=P,
#                                  num_channels_list=[best_params['num_channels']] * best_params['num_layers'],
#                                  kernel_size=best_params['kernel_size'],
#                                  dropout=best_params['dropout']).to(DEVICE)
#         final_criterion = nn.MSELoss()
#         final_lr = best_params['learning_rate']
#         final_lambda_reg = best_params['lambda_reg']
#         final_penalty_type = best_params['penalty_type']
#         final_alpha_gsgl = best_params.get('alpha_gsgl', 0.5) # 获取 alpha，如果不存在则用默认值
#
#         # 3. 训练更长时间 (例如 2*EPOCHS)
#         print(f"在完整训练集上训练 {EPOCHS * 2} 个 epochs...")
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
#                     first_block_conv1_weight_param = None
#                     if len(final_model.network_layers) > 0 and hasattr(final_model.network_layers[0], 'conv1'):
#                         first_block_conv1_weight_param = final_model.network_layers[0].conv1.weight
#
#                     for name, param in final_model.named_parameters():
#                         if param.grad is None: continue
#                         is_regularized = (param is first_block_conv1_weight_param)
#                         if is_regularized:
#                             w_tilde = param - final_lr * param.grad
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
#                             param.copy_(param - final_lr * param.grad)
#
#         # 4. 在测试集上评估
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
#
#             if len(test_loader.dataset) > 0: avg_test_mse = test_mse / len(test_loader.dataset)
#             else: avg_test_mse = float('inf')
#
#             final_test_weights = final_model.get_first_block_conv1_weights()
#             if final_test_weights is not None:
#                 for j in range(P):
#                     try:
#                         norm_val = torch.linalg.norm(final_test_weights[:, j, :].float(), ord='fro').cpu().item()
#                         test_norms[j] = norm_val if np.isfinite(norm_val) else 0.0
#                     except Exception: test_norms[j] = 0.0
#             else: test_norms = np.zeros(P)
#
#             # 计算测试集 AUROC 和 AUPR
#             test_true_causes = []
#             for j in range(P):
#                 is_j_a_cause = np.any(GC_true_np[:, j][np.arange(P) != j] == 1)
#                 test_true_causes.append(int(is_j_a_cause))
#             test_true_causes = np.array(test_true_causes)
#
#             test_auroc = 0.0
#             test_aupr = 0.0
#             if len(np.unique(test_true_causes)) > 1 and not np.all(np.isnan(test_norms)) and len(np.unique(test_norms)) > 1:
#                 try:
#                     valid_indices = ~np.isnan(test_norms)
#                     if np.any(valid_indices):
#                         test_auroc = roc_auc_score(test_true_causes[valid_indices], test_norms[valid_indices])
#                         test_aupr = average_precision_score(test_true_causes[valid_indices], test_norms[valid_indices])
#                     else: test_auroc, test_aupr = 0.0, 0.0
#                 except ValueError: test_auroc, test_aupr = 0.0, 0.0
#             else: test_auroc, test_aupr = 0.0, 0.0
#
#         print("\n--- 测试集评估结果 ---")
#         print(f"  测试集 MSE: {avg_test_mse:.6f}")
#         print(f"  测试集 AUROC (基于权重范数): {test_auroc:.4f}")
#         print(f"  测试集 AUPR (基于权重范数): {test_aupr:.4f}")
#         print(f"  测试集权重范数:")
#         for idx, norm_val in enumerate(test_norms):
#             print(f"    特征 {idx}: {norm_val:.4f}")
#
#         # (可选) 保存最终模型
#         # torch.save(final_model.state_dict(), f"{STUDY_NAME}_final_model.pth")
#         # print(f"最终模型已保存到 {STUDY_NAME}_final_model.pth")
#     else:
#         print("数据不足，无法在测试集上评估。")

print("\n脚本执行完毕。")