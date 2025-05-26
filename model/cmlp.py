import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy


def activation_helper(activation):
    if activation == 'relu':
        return nn.ReLU()
    elif activation == 'prelu':
        return nn.PReLU()
    elif activation == 'leaky_relu':
        return nn.LeakyReLU()
    elif activation == 'tanh':
        return nn.Tanh()
    elif activation == 'sigmoid':
        return nn.Sigmoid()
    else:
        return nn.ReLU()

# 一个MLP模型，用于单个时间序列的预测。
class MLP(nn.Module):
    def __init__(self, num_series, lag, hidden, activation):
        super(MLP, self).__init__()
        self.activation = activation_helper(activation)
        # 使用卷积来构建模型的第一层，输入维度为num_series，输出维度为hidden[0]，卷积核大小是滞后值
        layer = nn.Conv1d(num_series, hidden[0], lag) 
        # 输入数据的形状(batch_size, num_series, T)，经过第一层后，输出的形状 (batch_size, hidden[0], T - lag + 1)。
        modules = [layer]
        # 后续隐藏层
        for d_in, d_out in zip(hidden, hidden[1:] + [1]): # 遍历隐藏层的输入和输出维度
            layer = nn.Conv1d(d_in, d_out, 1) 
            modules.append(layer)

        # 把这些层注册到模型中
        self.layers = nn.ModuleList(modules)

    def forward(self, X):
        X = X.transpose(2, 1) # 变成[batch_size, series_num, series_length]
        for i, fc in enumerate(self.layers):
            if i != 0: # 如果当前层不是第一层
                X = self.activation(X) # 使用激活函数进行激活
            X = fc(X)

        return X.transpose(2, 1)


# 一个组件化多层感知机模型，为每个时间序列分别训练一个MLP。
class cMLP(nn.Module):
    def __init__(self, num_series, lag, hidden, activation='relu'):
        super(cMLP, self).__init__()
        self.p = num_series
        self.lag = lag
        self.activation = activation_helper(activation)

        # Set up networks.
        self.networks = nn.ModuleList([
            MLP(num_series, lag, hidden, activation)
            for _ in range(num_series)])

    def forward(self, X):
        '''
        Args:
          X: torch tensor of shape (batch, T, p).
        '''
        # 在指定维度拼接张量
        return torch.cat([network(X) for network in self.networks], dim=2)

    def GC(self, threshold=True, ignore_lag=True):
        '''
        Args:
          ignore_lag：是否忽略滞后。为True，返回一个 (p x p) 的矩阵
                                   为False，返回一个 (p x p x lag) 的矩阵。
          threshold: 是否返回权重的阈值化结果。为 True，则返回权重范数是否非零的二值矩阵；
                                            为 False，则返回权重范数的实际值。
        '''
        if ignore_lag: # 如果忽略滞后
            GC = [torch.norm(net.layers[0].weight, dim=(0, 2))
                  for net in self.networks]
        else:
            GC = [torch.norm(net.layers[0].weight, dim=0)
                  for net in self.networks]
        GC = torch.stack(GC) # 堆叠成一个张量
        if threshold: # 如果需要进行阈值处理
            return (GC > 0).int() # 变成0和1
        else:
            return GC


# 一个稀疏化的组件化多层感知机模型，仅使用指定的交互关系。
class cMLPSparse(nn.Module):
    def __init__(self, num_series, sparsity, lag, hidden, activation='relu'):
        '''
        cMLP model that only uses specified interactions.

        Args:
          num_series: dimensionality of multivariate time series.
          sparsity: torch byte tensor indicating Granger causality, with size
            (num_series, num_series).
          lag: number of previous time points to use in prediction.
          hidden: list of number of hidden units per layer.
          activation: nonlinearity at each layer.
        '''
        super(cMLPSparse, self).__init__()
        self.p = num_series
        self.lag = lag
        self.activation = activation_helper(activation)
        self.sparsity = sparsity

        # Set up networks.
        self.networks = []
        for i in range(num_series):
            num_inputs = int(torch.sum(sparsity[i].int()))
            self.networks.append(MLP(num_inputs, lag, hidden, activation))

        # Register parameters.
        param_list = []
        for i in range(num_series):
            param_list += list(self.networks[i].parameters())
        self.param_list = nn.ParameterList(param_list)

    def forward(self, X):
        '''
        Perform forward pass.

        Args:
          X: torch tensor of shape (batch, T, p).
        '''
        return torch.cat([self.networks[i](X[:, :, self.sparsity[i]])
                          for i in range(self.p)], dim=2)

# 对神经网络的第一层权重矩阵进行近端更新，作用于函数，直接对函数的参数进行稀疏性约束，使得某些参数被设置为零，从而实现稀疏性。
def prox_update(network, lam, lr, penalty):
    '''
    Args:
      network: MLP network.
      lam: 正则化参数
      lr: 学习率
      penalty: one of GL (group lasso), GSGL (group sparse group lasso),
        H (hierarchical).
    '''
    W = network.layers[0].weight
    hidden, p, lag = W.shape
    if penalty == 'GL': # 组Loss惩罚
        norm = torch.norm(W, dim=(0, 2), keepdim=True)
        W.data = ((W / torch.clamp(norm, min=(lr * lam)))
                  * torch.clamp(norm - (lr * lam), min=0.0))
    elif penalty == 'GSGL': # 组稀疏组Lasso惩罚
        norm = torch.norm(W, dim=0, keepdim=True)
        W.data = ((W / torch.clamp(norm, min=(lr * lam)))
                  * torch.clamp(norm - (lr * lam), min=0.0))
        norm = torch.norm(W, dim=(0, 2), keepdim=True)
        W.data = ((W / torch.clamp(norm, min=(lr * lam)))
                  * torch.clamp(norm - (lr * lam), min=0.0))
    elif penalty == 'H': # 层次Lasso惩罚
        # Lowest indices along third axis touch most lagged values.
        for i in range(lag):
            norm = torch.norm(W[:, :, :(i + 1)], dim=(0, 2), keepdim=True)
            W.data[:, :, :(i+1)] = (
                (W.data[:, :, :(i+1)] / torch.clamp(norm, min=(lr * lam)))
                * torch.clamp(norm - (lr * lam), min=0.0))
    else:
        raise ValueError('unsupported penalty: %s' % penalty)

# 计算第一层权重矩阵的Lasso惩罚项值
def regularize(network, lam, penalty):
    '''
    Args:
      network: MLP network.
      penalty: one of GL (group lasso), GSGL (group sparse group lasso),
        H (hierarchical).
    '''
    W = network.layers[0].weight # 选择第一层
    hidden, p, lag = W.shape
    if penalty == 'GL': # 组Loss惩罚
        return lam * torch.sum(torch.norm(W, dim=(0, 2)))
    elif penalty == 'GSGL': # 组稀疏组Lasso惩罚
        return lam * (torch.sum(torch.norm(W, dim=(0, 2)))
                      + torch.sum(torch.norm(W, dim=0)))
    elif penalty == 'H': # 层次Lasso惩罚
        # Lowest indices along third axis touch most lagged values.
        return lam * sum([torch.sum(torch.norm(W[:, :, :(i+1)], dim=(0, 2)))
                          for i in range(lag)])
    else:
        raise ValueError('unsupported penalty: %s' % penalty)

# 对后续层使用L2正则化
def ridge_regularize(network, lam):
    '''Apply ridge penalty at all subsequent layers.'''
    return lam * sum([torch.sum(fc.weight ** 2) for fc in network.layers[1:]])

# 使用 ISTA算法训练模型。
def train_model_ista(cmlp, X, lr, max_iter, lam=0, lam_ridge=0, penalty='H',
                     lookback=5, check_every=100, verbose=1):
    lag = cmlp.lag
    p = X.shape[-1]
    loss_fn = nn.MSELoss(reduction='mean')
    train_loss_list = []

    # 早停机制
    best_it = None
    best_loss = np.inf
    best_model = None

    # 使用i:i+1而不使用i是为了保持三维的形状，让输出维度和目标相匹配，使用i就是二维了
    loss = sum([loss_fn(cmlp.networks[i](X[:, :-1]), X[:, lag:, i:i+1])
                for i in range(p)])
    ridge = sum([ridge_regularize(net, lam_ridge) for net in cmlp.networks])
    smooth = loss + ridge

    for it in range(max_iter): 
        smooth.backward() # 反向传播
        for param in cmlp.parameters(): # 手动更新参数
            param.data = param - lr * param.grad

        # 对第一层进行近端优化
        if lam > 0:
            for net in cmlp.networks:
                prox_update(net, lam, lr, penalty)

        cmlp.zero_grad() # 梯度归零

        # 得到优化后模型的损失
        loss = sum([loss_fn(cmlp.networks[i](X[:, :-1]), X[:, lag:, i:i+1])
                    for i in range(p)])
        ridge = sum([ridge_regularize(net, lam_ridge) for net in cmlp.networks])
        smooth = loss + ridge

        # Check progress.
        if (it + 1) % check_every == 0:
            # 增加非平滑损失（第一层的Lass值）
            nonsmooth = sum([regularize(net, lam, penalty)
                             for net in cmlp.networks])
            mean_loss = (smooth + nonsmooth) / p
            train_loss_list.append(mean_loss.detach())

            if verbose > 0:
                print(('-' * 10 + 'Iter = %d' + '-' * 10) % (it + 1))
                print('Loss = %f' % mean_loss)
                print('Variable usage = %.2f%%'
                      % (100 * torch.mean(cmlp.GC().float())))

            # 检查是否早停
            if mean_loss < best_loss:
                best_loss = mean_loss
                best_it = it
                best_model = deepcopy(cmlp)
            elif (it - best_it) == lookback * check_every:
                if verbose:
                    print('Stopping early')
                break

    # Restore best model.
    restore_parameters(cmlp, best_model)

    return train_loss_list


# 将参数值从best_model移动到model
def restore_parameters(model, best_model):
    for params, best_params in zip(model.parameters(), best_model.parameters()):
        params.data = best_params
