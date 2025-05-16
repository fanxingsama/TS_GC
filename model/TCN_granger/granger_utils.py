import torch
# --- 组Lasso的近端操作符 ---
def prox_group_lasso(group_weights, lambda_thresh, norm_type='fro'):
    """
    对一组权重应用组软阈值操作。
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

    # 直接计算所选类型的范数
    if norm_type == 'fro':
        norm = torch.linalg.norm(group_weights, 'fro')
    elif norm_type == 'l2':
        norm = torch.linalg.norm(group_weights, ord=2)
    else:
        raise ValueError("不支持的 norm_type。请使用 'fro' 或 'l2'。")

    # 应用软阈值逻辑
    # 使用 torch.clamp 保证数值稳定性，避免除以非常小的范数
    scale = 1.0 - (lambda_thresh / torch.clamp(norm, min=1e-12)) # 避免除零
    scale = torch.clamp(scale, min=0.0) # max(0, scale)
    return group_weights * scale

# --- 组稀疏Lasso的近端操作符 ---
def prox_group_sparse_group_lasso(weights_for_feature, lambda_gamma, alpha):
    """
    对单个输入特征 j 的权重矩阵应用 Group Sparse Group Lasso 或 Group Lasso 的近端操作符。
    当 alpha = 1.0 时，此函数等效于 Group Lasso 的近端操作符。
    当 0 <= alpha < 1.0 时，此函数执行 Group Sparse Group Lasso 的近端操作。

    Args:
        weights_for_feature (torch.Tensor): 对应单个输入特征 j 的权重矩阵。
                                            期望形状: [out_channels, kernel_size]
        lambda_gamma (float): 正则化强度与步长的乘积 (lambda * gamma)。
        alpha (float): 混合系数，介于 0 和 1 之间。
                       alpha = 1.0 表示纯 Group Lasso (L_F 惩罚)。
                       alpha = 0.0 表示纯 L2 组稀疏 (L_2 惩罚)。
                       0 < alpha < 1 表示两者的混合。
    Returns:
        torch.Tensor: 应用近端算子后的权重矩阵。
    """
    if not (0 <= alpha <= 1):
        raise ValueError("alpha 必须在 0 和 1 之间。")

    out_channels, kernel_size = weights_for_feature.shape
    thresholded_weights_step1 = weights_for_feature.clone()

    # 步骤 1: (仅当 alpha < 1.0 时相关) 对每个 kernel position 应用 L2 组软阈值
    if alpha < 1.0: # 仅当 L2 部分的惩罚存在时执行
        lambda_gamma_l2 = lambda_gamma * (1.0 - alpha)
        if lambda_gamma_l2 > 0: # 仅当阈值大于0时执行
            for k in range(kernel_size):
                weight_column = weights_for_feature[:, k] # 从原始权重计算
                # 应用 L2 软阈值
                thresholded_weights_step1[:, k] = prox_group_lasso(
                    weight_column,
                    lambda_gamma_l2,
                    norm_type='l2'
                )

    # 步骤 2: (仅当 alpha > 0 时相关) 对经过步骤 1 处理后的整个特征权重矩阵应用 Frobenius 组软阈值
    final_weights = thresholded_weights_step1 # 如果 alpha = 0, 则这是最终结果
    if alpha > 0.0: # 仅当 Frobenius 部分的惩罚存在时执行
        lambda_gamma_fro = lambda_gamma * alpha
        # 应用 Frobenius 软阈值到上一步的结果上
        final_weights = prox_group_lasso(
            thresholded_weights_step1, # 注意：这里是对步骤1的结果进行操作
            lambda_gamma_fro,
            norm_type='fro'
        )
    
    return final_weights

# --- 计算整个权重张量的 组Lasso。---
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
        penalty += torch.linalg.norm(group_weights, ord='fro')
    
    return lambda_reg * penalty # 惩罚乘以正则化强度

# 计算整个权重张量的 组稀疏Lasso 。
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
        group_weights_matrix = weights[:, j, :]
        penalty_fro += torch.linalg.norm(group_weights_matrix, ord='fro')
        for k in range(kernel_size):
            group_weights_vector = weights[:, j, k]
            penalty_l2 += torch.linalg.norm(group_weights_vector, ord=2)

    # 组合两部分惩罚
    total_penalty = alpha * penalty_fro + (1.0 - alpha) * penalty_l2
    
    return lambda_reg * total_penalty # 惩罚乘以正则化强度

# 对神经网络的第一层权重矩阵进行近端更新，作用于函数，直接对函数的参数进行稀疏性约束，使得某些参数被设置为零，从而实现稀疏性。
def PGD_update(network, lam, lr, penalty):
    hidden, p, lag = network.shape
    if penalty == 'GL': # 组Loss惩罚
        norm = torch.norm(network, dim=(0, 2), keepdim=True)
        network.data = ((network / torch.clamp(norm, min=(lr * lam)))
                  * torch.clamp(norm - (lr * lam), min=0.0))
    elif penalty == 'GSGL': # 组稀疏组Lasso惩罚
        norm = torch.norm(network, dim=0, keepdim=True)
        network.data = ((network / torch.clamp(norm, min=(lr * lam)))
                  * torch.clamp(norm - (lr * lam), min=0.0))
        norm = torch.norm(network, dim=(0, 2), keepdim=True)
        network.data = ((network / torch.clamp(norm, min=(lr * lam)))
                  * torch.clamp(norm - (lr * lam), min=0.0))
    elif penalty == 'H': # 层次Lasso惩罚
        for i in range(lag):
            norm = torch.norm(network[:, :, :(i + 1)], dim=(0, 2), keepdim=True)
            network.data[:, :, :(i+1)] = (
                (network.data[:, :, :(i+1)] / torch.clamp(norm, min=(lr * lam)))
                * torch.clamp(norm - (lr * lam), min=0.0))
    else:
        raise ValueError('unsupported penalty: %s' % penalty)

# 计算第一层权重矩阵的正则化项，正则化是在损失函数里加入惩罚项，限制模型复杂度，让模型参数变得更简洁
def lasso_penalty(network, lam, penalty):
    hidden, p, lag = network.shape
    if penalty == 'GL': # 组Loss惩罚
        return lam * torch.sum(torch.norm(network, dim=(0, 2)))
    elif penalty == 'GSGL': # 组稀疏组Lasso惩罚
        return lam * (torch.sum(torch.norm(network, dim=(0, 2)))
                      + torch.sum(torch.norm(network, dim=0)))
    elif penalty == 'H': # 层次Lasso惩罚
        return lam * sum([torch.sum(torch.norm(network[:, :, :(i+1)], dim=(0, 2)))
                          for i in range(lag)])
    else:
        raise ValueError('unsupported penalty: %s' % penalty)


