import torch
import torch.nn.functional as F # 可能会用到，虽然当前实现未使用

def _calculate_frobenius_norm(tensor):
    """计算张量的 Frobenius 范数 (L2 范数)。"""
    return torch.linalg.norm(tensor, ord='fro')

def _calculate_l2_norm(tensor):
    """计算张量的 L2 范数。"""
    return torch.linalg.norm(tensor, ord=2)

# 对一组权重应用组软阈值操作。
def group_soft_thresholding(group_weights, lambda_thresh, norm_type='fro'):
    """
    Args:
        group_weights (torch.Tensor): 需要进行阈值操作的权重组。
                                      对于 Group Lasso (TCN): 形状 [out_channels, kernel_size]
                                      对于 Group Sparse Group Lasso (内部步骤): 形状 [out_channels]
        lambda_thresh (float): 阈值参数 (通常是 lambda * step_size)。
        norm_type (str): 使用的范数类型 ('fro' for Frobenius, 'l2' for L2)。
                         默认为 'fro'。

    Returns:
        torch.Tensor: 应用阈值操作后的权重组。
    """
    if lambda_thresh < 0:
        raise ValueError("阈值 lambda_thresh 不能为负数。")
    if lambda_thresh == 0:
        return group_weights # 如果阈值为0，不进行操作

    # 计算所选类型的范数
    if norm_type == 'fro':
        norm = _calculate_frobenius_norm(group_weights)
    elif norm_type == 'l2':
        norm = _calculate_l2_norm(group_weights)
    else:
        raise ValueError("不支持的 norm_type。请使用 'fro' 或 'l2'。")

    # 应用软阈值逻辑
    # 使用 torch.clamp 保证数值稳定性，避免除以非常小的范数
    scale = 1.0 - (lambda_thresh / torch.clamp(norm, min=1e-12)) # 避免除零
    scale = torch.clamp(scale, min=0.0) # max(0, scale)

    return group_weights * scale

# --- 群Lasso的近端算子---
def prox_group_lasso(weights_for_feature, lambda_gamma):
    """
    Args:
        weights_for_feature (torch.Tensor): 对应单个输入特征 j 的权重矩阵。
                                            期望形状: [out_channels, kernel_size]
        lambda_gamma (float): 正则化强度与步长的乘积 (lambda * gamma)。

    Returns:
        torch.Tensor: 应用近端算子后的权重矩阵。
    """
    # Group Lasso 使用 Frobenius 范数进行组软阈值
    return group_soft_thresholding(weights_for_feature, lambda_gamma, norm_type='fro')

# --- 群稀疏Lasso的近端算子---
def prox_group_sparse_group_lasso(weights_for_feature, lambda_gamma, alpha):
    """
    Args:
        weights_for_feature (torch.Tensor): 对应单个输入特征 j 的权重矩阵。
                                            期望形状: [out_channels, kernel_size]
        lambda_gamma (float): 正则化强度与步长的乘积 (lambda * gamma)。
        alpha (float): 混合系数，介于 0 和 1 之间，控制组稀疏和元素稀疏的平衡。

    Returns:
        torch.Tensor: 应用近端算子后的权重矩阵。
    """
    if not (0 <= alpha <= 1):
        raise ValueError("alpha 必须在 0 和 1 之间。")

    out_channels, kernel_size = weights_for_feature.shape
    # 创建一个副本用于存储中间结果，避免原地修改输入
    thresholded_weights_l2 = weights_for_feature.clone()

    # 步骤 1: 对每个 kernel position (类似滞后) 应用 L2 组软阈值
    # 惩罚项: (1-alpha) * sum_k ||W[:, j, k]||_2
    # 对应阈值: lambda_gamma * (1 - alpha)
    lambda_gamma_l2 = lambda_gamma * (1.0 - alpha)
    if lambda_gamma_l2 > 0: # 仅当阈值大于0时执行
        for k in range(kernel_size):
            weight_column = weights_for_feature[:, k] # 从原始权重计算
            # 应用 L2 软阈值
            thresholded_weights_l2[:, k] = group_soft_thresholding(weight_column,
                                                                   lambda_gamma_l2,
                                                                   norm_type='l2')
    # 步骤 2: 对经过步骤 1 处理后的整个特征权重矩阵应用 Frobenius 组软阈值
    # 惩罚项: alpha * ||W[:, j, :]||_F
    # 对应阈值: lambda_gamma * alpha
    lambda_gamma_fro = lambda_gamma * alpha
    # 应用 Frobenius 软阈值到上一步的结果上
    final_weights = group_soft_thresholding(thresholded_weights_l2,
                                            lambda_gamma_fro,
                                            norm_type='fro')

    return final_weights

# --- 计算整个权重张量的 Group Lasso 惩罚总和。---
def calculate_group_lasso_penalty(weights, lambda_reg):
    """
    Args:
        weights (torch.Tensor): 完整的权重张量。
                                 期望形状: [out_channels, in_channels, kernel_size]
        lambda_reg (float): 正则化强度超参数。

    Returns:
        torch.Tensor: 计算得到的总 Group Lasso 惩罚（标量）。
    """
    if weights is None:
        # 如果没有权重，返回一个在 CPU 上的 0 张量
        return torch.tensor(0.0, device='cpu')
    if not isinstance(weights, torch.Tensor):
         raise TypeError("输入 'weights' 必须是 PyTorch 张量。")

    # 确保惩罚张量与权重在同一设备和数据类型
    penalty = torch.tensor(0.0, device=weights.device, dtype=weights.dtype)
    num_input_features = weights.shape[1]

    # 累加每个输入特征组的 Frobenius 范数
    for j in range(num_input_features):
        group_weights = weights[:, j, :]
        penalty += _calculate_frobenius_norm(group_weights)

    # 乘以正则化强度
    return lambda_reg * penalty

# 计算整个权重张量的 Group Sparse Group Lasso 惩罚总和。
def calculate_group_sparse_group_lasso_penalty(weights, lambda_reg, alpha):
    """
    Args:
        weights (torch.Tensor): 完整的权重张量。
                                 期望形状: [out_channels, in_channels, kernel_size]
        lambda_reg (float): 正则化强度超参数。
        alpha (float): 混合系数 (0 <= alpha <= 1)。

    Returns:
        torch.Tensor: 计算得到的总 Group Sparse Group Lasso 惩罚（标量）。
    """
    if weights is None:
        return torch.tensor(0.0, device='cpu')
    if not isinstance(weights, torch.Tensor):
         raise TypeError("输入 'weights' 必须是 PyTorch 张量。")
    if not (0 <= alpha <= 1):
        raise ValueError("alpha 必须在 0 和 1 之间。")

    # 确保惩罚张量与权重在同一设备和数据类型
    penalty_fro = torch.tensor(0.0, device=weights.device, dtype=weights.dtype)
    penalty_l2 = torch.tensor(0.0, device=weights.device, dtype=weights.dtype)
    num_input_features = weights.shape[1]
    kernel_size = weights.shape[2]

    # 累加 Frobenius 和 L2 范数部分
    for j in range(num_input_features):
        # Frobenius 部分: ||W[:, j, :]||_F
        group_weights_matrix = weights[:, j, :]
        penalty_fro += _calculate_frobenius_norm(group_weights_matrix)
        # L2 部分: sum_k ||W[:, j, k]||_2
        for k in range(kernel_size):
            group_weights_vector = weights[:, j, k]
            penalty_l2 += _calculate_l2_norm(group_weights_vector)

    # 组合两部分惩罚
    total_penalty = alpha * penalty_fro + (1.0 - alpha) * penalty_l2
    # 乘以正则化强度
    return lambda_reg * total_penalty