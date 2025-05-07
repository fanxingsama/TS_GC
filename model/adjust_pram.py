from pathlib import Path
import joblib
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import rcParams
import optuna
from Granger_causalFormer import PredictModel

from TCN_granger.granger_utils import (
    prox_group_lasso,
    prox_group_sparse_group_lasso,
    calculate_group_lasso_penalty,
    calculate_group_sparse_group_lasso_penalty
)

# 设置 Matplotlib 中文显示
rcParams['font.family'] = 'SimHei'
rcParams['axes.unicode_minus'] = False

# 参数设置
# P = 5           # 时间序列的数量
# T = 1000        # 总时间点
LAG = 2         # 真实的 VAR 滞后
SPARSITY = 0.4  # 格兰杰因果矩阵的稀疏度
BETA_VALUE = 0.8# 系数值
SD = 0.1        # 噪声的标准差
DATA_SEED = 42  # 用于可重复性的随机种子
FEATURE_DIM = 1 # 假设当前模型结构简单，为单变量时间序列
OUTPUT_DIM = 1  # 假设预测序列值本身

# --- 训练和 Optuna 参数 ---
EPOCHS = 50
DATA_SEED = 42  # 用于可重复性的随机种子 (主要用于数据分割)
BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_TRIALS = 50 # Optuna 的试验次数
STUDY_NAME = "causalformer-grangertcn-pgd-study-min-mse" # 修改研究名称以反映新的目标
STORAGE_PATH = f"sqlite:///{STUDY_NAME}.db"


# --- 1. 生成并预处理数据 ---
# X_np, _, GC_true_np = simulate_var(p=P, T=T, lag=LAG, sparsity=SPARSITY,
#                                    beta_value=BETA_VALUE, sd=SD, seed=DATA_SEED)

# --- 1. 得到数据并预处理 ---
data_path = '../data/fMRI/timeseries9.csv'
true_gc_path = '../data/fMRI/sim9_gt_processed.csv'
df_a = pd.read_csv(data_path)
df_b = pd.read_csv(true_gc_path, header=None) # 读取真实的格兰杰因果矩阵
series_names = df_a.columns.tolist() # 获取序列名称
P = len(series_names) # 获取序列数量
X_np = df_a.values  # 获取所有的数据点
T = X_np.shape[0]   # 获取时间点数量 T
series_to_idx = {name: i for i, name in enumerate(series_names)} # 创建从序列名称到索引的映射

GC_true_np = np.zeros((P, P), dtype=int) # 初始化真实的格兰杰因果矩阵
# 读取真实的格兰杰因果矩阵
for _, row in df_b.iterrows():
    # 修改点：使用 .iloc 按位置访问 Series 中的元素
    cause_name = row.iloc[0]  # 第一列为 "因"
    effect_name = row.iloc[1] # 第二列为 "果"
    # lag_value = row.iloc[2] if df_b.shape[1] > 2 else None # 第三列为 "延迟" (可选)
    str_cause_name = str(cause_name)
    str_effect_name = str(effect_name)

    if str_cause_name in series_to_idx and str_effect_name in series_to_idx:
        idx_cause = series_to_idx[str_cause_name]
        idx_effect = series_to_idx[str_effect_name]
        GC_true_np[idx_effect, idx_cause] = 1

print(f"真实的格兰杰因果矩阵 GC_true_np (形状 {GC_true_np.shape}):\n{GC_true_np}")

X_np = X_np[:, :, np.newaxis] # 形状: [T, P, 1]

# 分割数据
X_train_val_np, X_test_np = train_test_split(X_np, test_size=0.2, random_state=DATA_SEED, shuffle=False)
X_train_np, X_val_np = train_test_split(X_train_val_np, test_size=0.25, random_state=DATA_SEED, shuffle=False)
print(f"训练集 X_train_np: {X_train_np.shape}")
print(f"验证集 X_val_np: {X_val_np.shape}")
print(f"测试集 X_test_np: {X_test_np.shape}")

# --- 辅助函数：创建序列 (适配 CausalFormer 输入输出) ---
def create_sequences(data, input_seq_len, output_seq_len):
    """
    创建适用于 CausalFormer 的序列数据。
    Args:
        data (np.array): 输入数据，形状 [时间步, 序列数量, 特征数量]。
        input_seq_len (int): 输入序列长度 (输入窗口)。
        output_seq_len (int): 输出序列长度 (输出窗口)。
    Returns:
        Tuple[np.array, np.array]: X (输入序列), Y (目标序列)
            X 形状: [样本数量, input_seq_len, 序列数量, 特征数量]
            Y 形状: [样本数量, output_seq_len, 序列数量, 特征数量]
    """
    xs, ys = [], []
    total_len = len(data)

    for i in range(total_len - input_seq_len - output_seq_len + 1):
        x = data[i:(i + input_seq_len)]  # 提取输入序列
        y = data[(i + input_seq_len):(i + input_seq_len + output_seq_len)]  # 提取目标序列
        xs.append(x)  # 将输入序列添加到列表 xs 中
        ys.append(y)  # 将目标序列添加到列表 ys 中
    return np.array(xs), np.array(ys)

# --- 2. 定义 Optuna 目标函数 ---
def objective(trial):
    global X_train_np, X_val_np, GC_true_np, P, FEATURE_DIM, OUTPUT_DIM
    # CausalFormer 参数
    input_window = trial.suggest_categorical('input_window', [10, 20, 30]) # 输入序列长度
    output_window = 1 # 固定预测下一步
    d_model = trial.suggest_categorical('d_model', [32, 64, 128])       # QK 嵌入维度
    n_head = trial.suggest_categorical('n_head', [2, 4, 8])             # 注意力头数
    n_layers = trial.suggest_int('n_layers', 1, 3)                      # Encoder 层数
    ffn_hidden = trial.suggest_categorical('ffn_hidden', [64, 128, 256])# FFN 隐藏层维度
    dropout = trial.suggest_float('dropout', 0.0, 0.3)                  # Dropout
    tau = trial.suggest_float('tau', 0.5, 10.0, log=True)               # Softmax 温度

    # GrangerTCN 参数
    tcn_layers = trial.suggest_int('tcn_layers', 2, 5)                  # TCN 块数
    tcn_channels = trial.suggest_categorical('tcn_channels', [16, 32, 48]) # TCN 通道数
    tcn_kernel_size = trial.suggest_categorical('tcn_kernel_size', [2, 3, 4]) # TCN 核大小
    tcn_dropout = trial.suggest_float('tcn_dropout', 0.0, 0.3)          # TCN Dropout
    tcn_channel_list = [tcn_channels] * tcn_layers

    # 近端梯度下降和稀疏性参数
    lr = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)     # 学习率
    lambda_reg = trial.suggest_float('lambda_reg', 1e-5, 1e-1, log=True)# 正则化强度
    penalty_type = trial.suggest_categorical('penalty_type', ['GL', 'GSGL']) # 惩罚类型
    alpha_gsgl = 0.5 # 默认值为 GSGL
    if penalty_type == 'GSGL':
        alpha_gsgl = trial.suggest_float('alpha_gsgl', 0.1, 0.9)

    print(f"\n--- Trial {trial.number} ---")
    print(f"  CausalFormer 参数: input_window={input_window}, d_model={d_model}, n_head={n_head}, n_layers={n_layers}, ffn_hidden={ffn_hidden}, dropout={dropout:.3f}, tau={tau:.3f}")
    print(f"  GrangerTCN 参数: layers={tcn_layers}, channels={tcn_channels}, kernel={tcn_kernel_size}, dropout={tcn_dropout:.3f}")
    print(f"  优化参数: lr={lr:.6f}, lambda_reg={lambda_reg:.6f}, penalty={penalty_type}"
          f"{f', alpha={alpha_gsgl:.2f}' if penalty_type == 'GSGL' else ''}")

    # --- 数据准备 ---
    X_train_seq, y_train_seq = create_sequences(X_train_np, input_window, output_window)
    X_val_seq, y_val_seq = create_sequences(X_val_np, input_window, output_window)

    # 转换为 PyTorch 张量
    X_train_tensor = torch.tensor(X_train_seq, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_seq, dtype=torch.float32) # 形状: [N, T_out, P, F_out]
    X_val_tensor = torch.tensor(X_val_seq, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val_seq, dtype=torch.float32)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    
    # 为了确保 PGD 状态的一致性，丢弃最后一个批次（如果处理得当可以移除）
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # --- 模型和损失函数 ---
    # 创建配置字典传递给 PredictModel
    config = {
        'data_loader': {
            'args': {
                'time_step': input_window,
                'output_window': output_window,
                'series_num': P,
                'feature_dim': FEATURE_DIM,
                'output_dim': OUTPUT_DIM
            }
        },
        'device': DEVICE.type # 传递设备类型
    }

    model = PredictModel(config=config,
                         d_model=d_model,
                         n_head=n_head,
                         n_layers=n_layers,
                         tcn_channels=tcn_channel_list,
                         tcn_kernel_size=tcn_kernel_size,
                         tcn_dropout=tcn_dropout,
                         ffn_hidden=ffn_hidden,
                         drop_prob=dropout,
                         tau=tau).to(DEVICE)

    criterion = nn.MSELoss()

    # --- 近端优化训练循环 ---
    final_val_auroc = 0.0 # 仍然计算 AUROC 用于记录，但不是优化目标
    final_avg_val_mse = float('inf')

    # 查找需要正则化的参数对象，这个路径取决于 PredictModel -> Encoder -> EncoderLayer -> MultiHeadAttention -> GrangerTCN -> TemporalBlock -> conv1
    granger_weights_param = model.encoder.layers[0].attention.tcn_processor.network_layers[0].conv1.weight

    # --- 训练循环 ---
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        epoch_penalty = 0.0
        num_batches = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)

            # 1. 计算主损失和梯度
            model.zero_grad()
            predictions = model(batch_x) # 形状: [B, T_out, P, F_out]
            # 确保 target 形状匹配 prediction
            main_loss = criterion(predictions, batch_y)
            main_loss.backward() # 计算所有参数的梯度

            epoch_loss += main_loss.item()
            num_batches += 1

            # --- 手动执行 PGD 更新 ---
            with torch.no_grad():
                # 得到Lasso惩罚值
                current_penalty = torch.tensor(0.0, device=DEVICE)
                current_weights = granger_weights_param.data # 获取当前权重数据
                if penalty_type == 'GL':
                    current_penalty = calculate_group_lasso_penalty(current_weights, lambda_reg)
                elif penalty_type == 'GSGL':
                    current_penalty = calculate_group_sparse_group_lasso_penalty(current_weights, lambda_reg, alpha_gsgl)
                epoch_penalty += current_penalty.item()

                # 更新整个模型的参数
                for name, param in model.named_parameters():
                    if param.grad is None: continue # 跳过没有梯度的参数

                    # 检查是否是需要正则化的权重
                    is_regularized_weight = (param is granger_weights_param)

                    #对第一层使用近端操作符进行近端更新
                    if is_regularized_weight:
                        w_tilde = param.data - lr * param.grad  # 梯度下降公式
                        lambda_gamma = lr * lambda_reg #  是正则化参数，在近端操作中控制正则化的强度。
                        w_new = torch.zeros_like(w_tilde)  # 初始化新的权重张量

                        if penalty_type == 'GL':
                            for j in range(w_tilde.shape[1]): # 遍历输入特征 (P)
                                w_new[:, j, :] = prox_group_lasso(w_tilde[:, j, :], lambda_gamma)
                        elif penalty_type == 'GSGL':
                            for j in range(w_tilde.shape[1]):
                                w_new[:, j, :] = prox_group_sparse_group_lasso(w_tilde[:, j, :], lambda_gamma, alpha_gsgl)
                        param.copy_(w_new) # 更新参数
                    else:
                        # 对其他参数执行标准梯度下降
                        param.copy_(param.data - lr * param.grad)

        avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else float('inf')
        avg_epoch_penalty = epoch_penalty / num_batches if num_batches > 0 else float('inf')
        print(f"  Epoch {epoch+1}/{EPOCHS}, Avg Train Loss: {avg_epoch_loss:.6f}, Avg Penalty: {avg_epoch_penalty:.6f}")

        # --- 验证 ---
        model.eval()
        val_mse = 0.0 # 累计验证集的均方误差
        learned_norms = np.zeros(P) # 每个输入特征的范数
        current_val_auroc = 0.0 # 验证集的 AUROC
        current_val_aupr = 0.0 # 验证集的 AUPR

        with torch.no_grad(): # 禁用梯度计算，以提高评估效率。
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
                predictions = model(batch_x)
                loss = criterion(predictions, batch_y) # 验证集损失值
                val_mse += loss.item() * batch_x.size(0) # 乘以批次的大小以得到总的损失值
            avg_val_mse = val_mse / len(val_loader.dataset) if len(val_loader.dataset) > 0 else float('inf')


            # 获取训练后的 Granger TCN 权重
            final_weights = model.get_granger_weights(layer_index=0) # 获取第一个 encoder layer 的 TCN 权重
            if final_weights is not None:
                for j in range(P): # 计算每个输入特征的权重范数（Frobenius 范数）
                    norm_val = torch.linalg.norm(final_weights[:, j, :].float(), ord='fro').cpu().item()
                    learned_norms[j] = norm_val if np.isfinite(norm_val) else 0.0
            else:
                learned_norms = np.zeros(P)

            # --- 计算 AUROC 和 AUPR (仅供参考) ---
            if GC_true_np.shape[0] == P and P > 0 : # Ensure GC_true_np is valid
                true_causes = np.array([np.any(GC_true_np[np.arange(P) != j, j] == 1) for j in range(P)], dtype=int) # 一个布尔数组，表示每个输入特征是否是其他特征的因果因素
                scores = learned_norms # 分数是每个输入特征的权重范数
                valid_indices = ~np.isnan(scores) & ~np.isinf(scores) # 权重矩阵只要有值，就是True

                if len(np.unique(true_causes[valid_indices])) > 1 and len(scores[valid_indices]) > 1:
                    try:
                        current_val_auroc = roc_auc_score(true_causes[valid_indices], scores[valid_indices])
                        current_val_aupr = average_precision_score(true_causes[valid_indices], scores[valid_indices])
                    except ValueError: # Handle cases where AUROC/AUPR cannot be computed
                        current_val_auroc = 0.0
                        current_val_aupr = 0.0
                else:
                    current_val_auroc = 0.0
                    current_val_aupr = 0.0
            else:
                current_val_auroc = 0.0
                current_val_aupr = 0.0


        # 最终的验证集 AUROC 和 MSE
        final_val_auroc = current_val_auroc
        final_avg_val_mse = avg_val_mse

        # --- Optuna 剪枝 (使用 MSE) ---
        trial.report(final_avg_val_mse, epoch) # 在每个 epoch 结束时报告当前的 MSE
        if trial.should_prune(): # 检查是否需要剪枝
            print(f"Trial {trial.number} 在第 {epoch+1} 轮被剪枝。")
            return float('inf') # 返回一个很大的值，因为我们要最小化 MSE

    print(f"Trial {trial.number} 完成。最终验证集 AUROC: {final_val_auroc:.4f}, 最终验证集 MSE: {final_avg_val_mse:.6f}")
    # 返回最终验证集 MSE 给 Optuna
    return final_avg_val_mse if np.isfinite(final_avg_val_mse) else float('inf')

# --- 3. 创建或加载 Optuna Study 并运行优化 ---
study = optuna.create_study(
    study_name=STUDY_NAME, # 本次Optuna实验的名称
    storage=None, # 不保存db文件
    direction='minimize', # 修改优化目标为最小化
    load_if_exists=True, # 如果存在则加载现有的 Study
    pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=15, interval_steps=1) # 调整剪枝参数
)
optuna.logging.set_verbosity(optuna.logging.WARNING) # 设置 Optuna 的日志级别为警告级别

print(f"\n开始 Optuna 超参数优化 (目标: 最小化 MSE)") # 更新打印信息
study.optimize(objective, n_trials=N_TRIALS, timeout=None)

# --- 4. 输出结果 ---
completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE] # 从Optuna中获取所有完成的Trial
print(f"\nOptuna 优化完成,总尝试次数: {len(study.trials)}，成功完成的 Trial 数量: {len(completed_trials)}")

best_params = None
best_trial = None # Initialize best_trial

if completed_trials:
    try:
        best_trial = study.best_trial
        print(f"\n最佳 Trial:")
        print(f"  编号: {best_trial.number}")
        print(f"  目标值 (Val MSE): {best_trial.value:.6f}") # 修改标签为 MSE
        print(f"  最佳超参数:")
        best_params = best_trial.params
        for key, value in best_params.items():
            print(f"    {key}: {value}")
        joblib.dump(best_params, f"{STUDY_NAME}_best_params.pkl") # 保存最佳参数
    except ValueError: # Catches case where no trials complete successfully for best_trial
        print("警告: 没有找到最佳 Trial (可能所有 Trial 都被剪枝或失败)。")
else:
    print("没有 Trial 成功完成。")


# --- 5. Matplotlib可视化优化过程 ---
if completed_trials: # Check if there are completed trials to plot
    mse_values = [t.value for t in completed_trials if t.value is not None and np.isfinite(t.value)] # 过滤掉 None 和非有限值
    valid_trial_numbers = [t.number for t in completed_trials if t.value is not None and np.isfinite(t.value)]

    if mse_values: # Proceed only if there are valid mse values
        plt.figure(figsize=(12, 7))
        plt.scatter(valid_trial_numbers, mse_values, alpha=0.6, label='完成的 Trials', s=50)

        # 标记最佳 trial (如果存在且值有效)
        if best_trial is not None and best_trial.value is not None and np.isfinite(best_trial.value):
            plt.scatter([best_trial.number], [best_trial.value], color='red', s=150, label=f'最佳 Trial ({best_trial.number}) MSE: {best_trial.value:.4f}', zorder=5, edgecolors='black')

        plt.xlabel("Trial 编号")
        plt.ylabel("验证集 MSE") # 修改 Y 轴标签
        plt.title(f"Optuna 优化历史 ({STUDY_NAME}) - Matplotlib") # 修改标题
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{STUDY_NAME}_matplotlib_history.png")
        print(f"\n使用 Matplotlib 绘制的优化历史图已保存到 {STUDY_NAME}_matplotlib_history.png")
    else:
        print("\n没有有效的 MSE 值可以绘制优化历史图。")
else:
    print("\n没有完成的 Trial，无法绘制优化历史。")


# def main(run_id):
#     log_save_path = Path('saved') / run_id / 'train_result_log'
#     filename = Path('saved') / run_id / 'model'
#     filename.mkdir(parents=True, exist_ok=True)
#     setup_logging(log_save_path)
#     train_logger = get_logger() # 日志记录器
    
#     train_logger.info(model) # 模型架构保存
#     train_logger.info("==============模型训练开始==============")
#     train_logger.info(f"模型所使用数据集: {config['data_loader']['args']['data_dir']}")
    
#     print("\n脚本执行完毕。")

# # 单独运行这个文件
# if __name__ == '__main__':
#     run_id = datetime.now().strftime(r'%m%d_%H%M%S') # 获得当前时间
#     main(run_id)
#     torch.cuda.empty_cache() # 清理显存
