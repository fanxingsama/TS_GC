import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import traceback

# 根据你的项目结构调整导入路径
# 假设此文件位于 model/training/
from Granger_causalFormer import PredictModel
from TCN_granger.granger_utils import (
    prox_group_lasso,
    prox_group_sparse_group_lasso,
    calculate_group_lasso_penalty,
    calculate_group_sparse_group_lasso_penalty
)

def train_and_evaluate(params, config, train_loader, val_loader, gc_true_np, device, epochs):
    """
    Returns:
        tuple: (final_val_auroc, final_avg_val_mse)
               final_val_auroc (float): 最后一轮后验证集上的 AUROC。
               final_avg_val_mse (float): 最后一轮后验证集上的平均 MSE。
               如果发生错误，则返回 (-1.0, float('inf'))。
    """
    try:
        d_model = params['d_model']
        n_head = params['n_head']
        n_layers = params['n_layers']
        tcn_channels = params['tcn_channels']
        tcn_kernel_size = params['tcn_kernel_size']
        tcn_dropout = params['tcn_dropout']
        tcn_layers = params['tcn_layers']
        ffn_hidden = params['ffn_hidden']
        dropout = params['dropout']
        tau = params['tau']
        lr = params['learning_rate']
        lambda_reg = params['lambda_reg']
        penalty_type = params['penalty_type']
        alpha_gsgl = params.get('alpha_gsgl', 0.5) # 如果不是 GSGL，则使用默认值

        tcn_channel_list = [tcn_channels] * tcn_layers
        P = config['data_loader']['args']['series_num'] # 获取序列数量

        # --- 模型设置 ---
        model = PredictModel(config=config,
                             d_model=d_model,
                             n_head=n_head,
                             n_layers=n_layers,
                             tcn_channels=tcn_channel_list,
                             tcn_kernel_size=tcn_kernel_size,
                             tcn_dropout=tcn_dropout,
                             ffn_hidden=ffn_hidden,
                             drop_prob=dropout,
                             tau=tau).to(device)

        criterion = nn.MSELoss()

        # --- 近端优化设置 ---
        granger_weights_param = None
        try:
            # 定位用于正则化的权重
            granger_weights_param = model.encoder.layers[0].attention.tcn_processor.network_layers[0].conv1.weight
            print("  成功定位 Granger TCN 权重进行正则化。")
        except AttributeError:
            print("  警告：无法定位 Granger TCN 权重。将不应用稀疏惩罚。")

        # --- 训练循环 ---
        final_val_auroc = -1.0
        final_avg_val_mse = float('inf')

        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0
            epoch_penalty = 0.0
            num_batches = 0

            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)

                # 1. 计算主损失和梯度
                model.zero_grad()
                predictions = model(batch_x)
                main_loss = criterion(predictions, batch_y)
                main_loss.backward() # 计算所有参数的梯度

                epoch_loss += main_loss.item()
                num_batches += 1

                # --- 手动 PGD 更新 ---
                with torch.no_grad():
                    current_penalty = torch.tensor(0.0, device=device)
                    if granger_weights_param is not None:
                        current_weights = granger_weights_param.data
                        if penalty_type == 'GL':
                            current_penalty = calculate_group_lasso_penalty(current_weights, lambda_reg)
                        elif penalty_type == 'GSGL':
                            current_penalty = calculate_group_sparse_group_lasso_penalty(current_weights, lambda_reg, alpha_gsgl)
                    epoch_penalty += current_penalty.item()

                    # 更新参数
                    for name, param in model.named_parameters():
                        if param.grad is None: continue

                        is_regularized_weight = (param is granger_weights_param)

                        if is_regularized_weight:
                            # 对正则化权重应用近端算子
                            w_tilde = param.data - lr * param.grad
                            lambda_gamma = lr * lambda_reg
                            w_new = torch.zeros_like(w_tilde)

                            if penalty_type == 'GL':
                                for j in range(w_tilde.shape[1]): # 遍历输入特征 (P)
                                    w_new[:, j, :] = prox_group_lasso(w_tilde[:, j, :], lambda_gamma)
                            elif penalty_type == 'GSGL':
                                for j in range(w_tilde.shape[1]):
                                    w_new[:, j, :] = prox_group_sparse_group_lasso(w_tilde[:, j, :], lambda_gamma, alpha_gsgl)
                            param.copy_(w_new) # 更新参数
                        else:
                            # 对其他参数进行标准梯度下降
                            param.copy_(param.data - lr * param.grad)

            avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0
            avg_epoch_penalty = epoch_penalty / num_batches if num_batches > 0 else 0
            # 可选：减少打印频率或在 Optuna 运行时移除
            if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
                 print(f"    轮次 {epoch+1}/{epochs}, 平均训练损失: {avg_epoch_loss:.6f}, 平均惩罚项: {avg_epoch_penalty:.6f}")

        # --- 验证 (在最后一轮之后) ---
        model.eval()
        val_mse = 0.0
        learned_norms = np.zeros(P)
        current_val_auroc = 0.0
        current_val_aupr = 0.0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                predictions = model(batch_x)
                loss = criterion(predictions, batch_y)
                val_mse += loss.item() * batch_x.size(0)

            if len(val_loader.dataset) > 0:
                final_avg_val_mse = val_mse / len(val_loader.dataset)
            else:
                final_avg_val_mse = float('inf')

            # 获取最终权重用于 AUROC/AUPR 计算
            final_weights = model.get_granger_weights(layer_index=0)
            if final_weights is not None:
                for j in range(P):
                    try:
                        norm_val = torch.linalg.norm(final_weights[:, j, :].float(), ord='fro').cpu().item()
                        learned_norms[j] = norm_val if np.isfinite(norm_val) else 0.0
                    except Exception as e_norm:
                        learned_norms[j] = 0.0
            else:
                learned_norms = np.zeros(P)

            # --- 计算 AUROC 和 AUPR ---
            true_causes = np.array([np.any(gc_true_np[np.arange(P) != j, j] == 1) for j in range(P)], dtype=int)
            scores = learned_norms

            if len(np.unique(true_causes)) > 1 and not np.all(np.isnan(scores)) and len(np.unique(scores)) > 1:
                try:
                    valid_indices = ~np.isnan(scores)
                    if np.any(valid_indices):
                        current_val_auroc = roc_auc_score(true_causes[valid_indices], scores[valid_indices])
                        current_val_aupr = average_precision_score(true_causes[valid_indices], scores[valid_indices])
                    else:
                        current_val_auroc, current_val_aupr = 0.0, 0.0
                except ValueError as e_auc:
                    current_val_auroc, current_val_aupr = 0.0, 0.0
            else:
                current_val_auroc, current_val_aupr = 0.0, 0.0

        final_val_auroc = current_val_auroc if np.isfinite(current_val_auroc) else -1.0

        print(f"  训练完成。最终验证集 AUROC: {final_val_auroc:.4f}, 最终验证集 MSE: {final_avg_val_mse:.6f}")
        return final_val_auroc, final_avg_val_mse

    except Exception as e:
        print(f"训练/评估过程中出错: {e}")
        traceback.print_exc()
        return -1.0, float('inf') # 指示失败

# 辅助函数：创建序列 (移到此处以封装)
def create_sequences(data, input_seq_len, output_seq_len):
    """
    为时间序列预测创建序列。
    Args:
        data (np.array): 输入数据，形状 [时间步, 序列数, 特征数]。
        input_seq_len (int): 输入窗口长度。
        output_seq_len (int): 输出窗口长度。
    Returns:
        Tuple[np.array, np.array]: X (输入), Y (目标)
            X 形状: [样本数, input_seq_len, 序列数, 特征数]
            Y 形状: [样本数, output_seq_len, 序列数, 特征数]
    """
    xs, ys = [], []
    total_len = len(data)
    for i in range(total_len - input_seq_len - output_seq_len + 1):
        x = data[i:(i + input_seq_len)]
        y = data[(i + input_seq_len):(i + input_seq_len + output_seq_len)]
        xs.append(x)
        ys.append(y)
    if not xs: # 处理数据过短的情况
        return np.empty((0, input_seq_len, data.shape[1], data.shape[2])), \
               np.empty((0, output_seq_len, data.shape[1], data.shape[2]))
    return np.array(xs), np.array(ys)
