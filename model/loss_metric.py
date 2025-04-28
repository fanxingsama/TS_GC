import torch
import torch.nn.functional as F
import torch
import numpy as np
from sklearn.metrics import r2_score, explained_variance_score

'''
定义所需要的所有损失函数和评价指标
nll_loss:使用 PyTorch 的负对数似然损失（Negative Log Likelihood Loss）。通常用于分类任务。
mse_loss:计算均方误差（Mean Squared Error, MSE）。这是回归任务中常用的损失函数，用于衡量预测值与真实值之间的差异。
masked_mae_loss:计算掩码后的平均绝对误差（Masked Mean Absolute Error, MAE）。通过掩码忽略某些值（如零值），适用于处理稀疏数据。
masked_mae_torch:类似于 masked_mae_loss，但更灵活地处理无效值（如 NaN 或特定的 null_val）。通过掩码对数据进行加权，避免无效值对损失计算的影响。
log_cosh_loss:计算 log(cosh(preds - labels)) 的平均值。这种损失函数对异常值的敏感度较低，适用于回归任务。
huber_loss:Huber 损失函数，结合了平方误差和绝对误差的优点。对于小误差使用平方误差，对于大误差使用线性误差。
quantile_loss:分位数损失函数，用于量化回归任务。通过调整参数 delta，可以关注不同分位数的误差。
masked_mape_torch:计算掩码后的平均绝对百分比误差（Masked Mean Absolute Percentage Error, MAPE）。适用于处理零值或接近零的标签。
masked_mse_torch:计算掩码后的均方误差（Masked MSE）。通过掩码忽略无效值。
masked_rmse_torch:计算掩码后的均方根误差（Masked RMSE）。它是 masked_mse_torch 的平方根。
smooth_l1_loss:使用 PyTorch 的平滑 L1 损失函数。它结合了 L1 和 L2 损失的优点，对异常值更鲁棒。
评估指标:
r2_score_torch:使用 sklearn.metrics.r2_score 计算 R² 分数。R² 分数衡量模型对数据的拟合程度，值越接近 1 表示拟合越好。
explained_variance_score_torch:使用 sklearn.metrics.explained_variance_score 计算解释方差分数。它衡量模型解释数据方差的能力。
r2_score_np 和 explained_variance_score_np:这些函数与上述类似，但使用 NumPy 处理数据，适用于非 PyTorch 环境。
掩码和处理无效值:
masked_mae_np、masked_mse_np、masked_mape_np:这些函数使用 NumPy 实现掩码操作，适用于处理无效值（如 NaN 或特定的 null_val）。它们在处理大规模数据时可能更高效。
accuracy:计算分类任务的准确率。通过 torch.argmax 获取预测的类别，并与目标类别进行比较，计算正确预测的比例。
top_k_acc:计算 Top-k 准确率。对于每个样本，检查预测的前 k 个最高概率类别中是否包含真实类别。这在多分类任务中很有用，尤其是当类别不平衡时。
'''
def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)

def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)

# 负对数似然损失
def nll_loss(output, target):
    return F.nll_loss(output, target)

# 均方误差损失
def mse_loss(y_pred, y_true):
    return F.mse_loss(y_pred, y_true)

# 交叉熵损失
def cross_entropy_loss(output, target):
    return F.cross_entropy(output, target)

def masked_mae_loss(y_pred, y_true):
    mask = (y_true != 0).float()
    mask /= mask.mean()
    loss = torch.abs(y_pred - y_true)
    loss = loss * mask
    loss[loss != loss] = 0
    return loss.mean()

def masked_mae_torch(preds, labels, null_val=np.nan):
    labels[torch.abs(labels) < 1e-4] = 0
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = labels.ne(null_val)
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(torch.sub(preds, labels))
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def log_cosh_loss(preds, labels):
    loss = torch.log(torch.cosh(preds - labels))
    return torch.mean(loss)

def huber_loss(preds, labels, delta=1.0):
    residual = torch.abs(preds - labels)
    condition = torch.le(residual, delta)
    small_res = 0.5 * torch.square(residual)
    large_res = delta * residual - 0.5 * delta * delta
    return torch.mean(torch.where(condition, small_res, large_res))

def quantile_loss(preds, labels, delta=0.25):
    condition = torch.ge(labels, preds)
    large_res = delta * (labels - preds)
    small_res = (1 - delta) * (preds - labels)
    return torch.mean(torch.where(condition, large_res, small_res))

def masked_mape_torch(preds, labels, null_val=np.nan, eps=0):
    labels[torch.abs(labels) < 1e-4] = 0
    if np.isnan(null_val) and eps != 0:
        loss = torch.abs((preds - labels) / (labels + eps))
        return torch.mean(loss)
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = labels.ne(null_val)
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs((preds - labels) / labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def masked_mse_torch(preds, labels, null_val=np.nan):
    labels[torch.abs(labels) < 1e-4] = 0
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = labels.ne(null_val)
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.square(torch.sub(preds, labels))
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def masked_rmse_torch(preds, labels, null_val=np.nan):
    labels[torch.abs(labels) < 1e-4] = 0
    return torch.sqrt(masked_mse_torch(preds=preds, labels=labels,
                                       null_val=null_val))

def r2_score_torch(preds, labels):
    preds = preds.cpu().flatten()
    labels = labels.cpu().flatten()
    return r2_score(labels, preds)

def explained_variance_score_torch(preds, labels):
    preds = preds.cpu().flatten()
    labels = labels.cpu().flatten()
    return explained_variance_score(labels, preds)

def masked_rmse_np(preds, labels, null_val=np.nan):
    return np.sqrt(masked_mse_np(preds=preds, labels=labels,
                   null_val=null_val))

def masked_mse_np(preds, labels, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(labels)
        else:
            mask = np.not_equal(labels, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        rmse = np.square(np.subtract(preds, labels)).astype('float32')
        rmse = np.nan_to_num(rmse * mask)
        return np.mean(rmse)

def masked_mae_np(preds, labels, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(labels)
        else:
            mask = np.not_equal(labels, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mae = np.abs(np.subtract(preds, labels)).astype('float32')
        mae = np.nan_to_num(mae * mask)
        return np.mean(mae)

def masked_mape_np(preds, labels, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(labels)
        else:
            mask = np.not_equal(labels, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mape = np.abs(np.divide(np.subtract(
            preds, labels).astype('float32'), labels))
        mape = np.nan_to_num(mask * mape)
        return np.mean(mape)

def r2_score_np(preds, labels):
    preds = preds.flatten()
    labels = labels.flatten()
    return r2_score(labels, preds)

def explained_variance_score_np(preds, labels):
    preds = preds.flatten()
    labels = labels.flatten()
    return explained_variance_score(labels, preds)

def smooth_l1_loss(preds, labels):
    criterion = torch.nn.SmoothL1Loss()
    return criterion(labels, preds)
