import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy
from models.model_helper import activation_helper

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

# 使用 GISTA算法训练模型，通过线搜索，动态调整学习率
def train_model_gista(cmlp, X, lam, lam_ridge, lr, penalty, max_iter,
                      check_every=100, r=0.8, lr_min=1e-8, sigma=0.5,
                      monotone=False, m=10, lr_decay=0.5,
                      begin_line_search=True, switch_tol=1e-3, verbose=1):
    '''
    Train cMLP model with GISTA.

    Args:
        clstm：clstm模型。
        X：模型的数据张量，形状为（batch, T, p）。
        lam：非平滑正则化的参数。
        lam_ridge：输出层的岭正则化参数。
        lr：学习率。
        penalty：非平滑正则化的类型。
        max_iter：GISTA迭代的最大次数。
        check_every：记录损失的频率。
        r：用于线搜索。
        lr_min：用于线搜索的最小学习率。
        sigma：用于线搜索的参数。
        monotone：用于线搜索的参数，是否要求单调性。
        m：用于线搜索的参数。
        lr_decay：用于调整线搜索初始学习率的参数。
        begin_line_search：是否从线搜索开始。
        switch_tol：切换到线搜索的容差。
        verbose：日志输出的详细程度（0、1、2）。
    '''
    p = cmlp.p # 时间序列的数量
    lag = cmlp.lag # 滞后值
    cmlp_copy = deepcopy(cmlp)
    loss_fn = nn.MSELoss(reduction='mean')
    lr_list = [lr for _ in range(p)] # 每个网络的学习率列表

    # Calculate full loss.
    mse_list = []
    smooth_list = []
    loss_list = []
    for i in range(p): # 对每个网络进行处理
        net = cmlp.networks[i]
        mse = loss_fn(net(X[:, :-1]), X[:, lag:, i:i+1]) # 基于滞后值计算均方误差
        ridge = ridge_regularize(net, lam_ridge) # 对除了第一层以外的层进行L2正则化
        smooth = mse + ridge # 计算平滑损失，结合了MSE和L2
        mse_list.append(mse)
        smooth_list.append(smooth)
        # 对第一层的权重矩阵进行正则化
        with torch.no_grad():
            nonsmooth = regularize(net, lam, penalty) # 正则化之后的值
            loss = smooth + nonsmooth # 计算总损失，模型的目标应该是最小化总损失
            loss_list.append(loss) 

    with torch.no_grad():
        loss_mean = sum(loss_list) / p
        mse_mean = sum(mse_list) / p
    train_loss_list = [loss_mean]
    train_mse_list = [mse_mean]

    # 线搜索
    line_search = begin_line_search

    # For line search criterion.
    done = [False for _ in range(p)] # 记录每个网络是否已经收敛
    # 声明线搜索参数
    assert 0 < sigma <= 1
    assert m > 0
    if not monotone:
        last_losses = [[loss_list[i]] for i in range(p)]

    # 对所有未收敛的网络的平滑损失求和，并计算梯度。
    for it in range(max_iter):
        sum([smooth_list[i] for i in range(p) if not done[i]]).backward()

        # 初始化新的损失列表
        new_mse_list = []
        new_smooth_list = []
        new_loss_list = []

        # Perform GISTA step for each network.
        for i in range(p):
            # 如果某个网络已经收敛，则跳过该网络，直接将上次的结果复制到新的列表中。
            if done[i]:
                new_mse_list.append(mse_list[i])
                new_smooth_list.append(smooth_list[i])
                new_loss_list.append(loss_list[i])
                continue

            # Prepare for line search.
            step = False 
            lr_it = lr_list[i]
            net = cmlp.networks[i]
            net_copy = cmlp_copy.networks[i]

            while not step:
                # 对当前网络的参数进行梯度下降更新。
                # zip用于从两个可迭代对象各取出一个参数，形成一个元组
                for param, temp_param in zip(net.parameters(),
                                             net_copy.parameters()):
                    temp_param.data = param - lr_it * param.grad

                # 对更新后的参数进行近端操作，以满足非光滑正则化项的约束。
                prox_update(net_copy, lam, lr_it, penalty)

                # 重新计算一遍损失
                mse = loss_fn(net_copy(X[:, :-1]), X[:, lag:, i:i+1])
                ridge = ridge_regularize(net_copy, lam_ridge) # 对除了第一层以外的层进行L2正则化
                smooth = mse + ridge
                with torch.no_grad():
                    nonsmooth = regularize(net_copy, lam, penalty)
                    loss = smooth + nonsmooth
                    # 线搜索的容忍度
                    tol = (0.5 * sigma / lr_it) * sum(
                        [torch.sum((param - temp_param) ** 2)
                         for param, temp_param in
                         zip(net.parameters(), net_copy.parameters())])

                # 如果当前损失小于上一次的损失减去容忍度，则接受这次更新。
                comp = loss_list[i] if monotone else max(last_losses[i])
                # 如果线搜索条件满足
                if not line_search or (comp - loss) > tol:
                    step = True
                    if verbose > 1:
                        print('Taking step, network i = %d, lr = %f'
                              % (i, lr_it))
                        print('Gap = %f, tol = %f' % (comp - loss, tol))

                    # For next iteration.
                    new_mse_list.append(mse)
                    new_smooth_list.append(smooth)
                    new_loss_list.append(loss)

                    # Adjust initial learning rate.
                    lr_list[i] = (
                        (lr_list[i] ** (1 - lr_decay)) * (lr_it ** lr_decay))

                    if not monotone:
                        if len(last_losses[i]) == m:
                            last_losses[i].pop(0)
                        last_losses[i].append(loss)
                # 如果条件不满足，则减小学习率，直到学习率小于lr_min。
                else:
                    lr_it *= r
                    if lr_it < lr_min:
                        done[i] = True # 将该网络标记为已经收敛
                        new_mse_list.append(mse_list[i])
                        new_smooth_list.append(smooth_list[i])
                        new_loss_list.append(loss_list[i])
                        if verbose > 0:
                            print('Network %d converged' % (i + 1))
                        break

            # Clean up.
            net.zero_grad()

            if step:
                # Swap network parameters.
                cmlp.networks[i], cmlp_copy.networks[i] = net_copy, net

        # 更新损失列表，为下一次迭代做准备。
        mse_list = new_mse_list
        smooth_list = new_smooth_list
        loss_list = new_loss_list

        # 检查是否所有网络都已收敛
        if sum(done) == p:
            if verbose > 0:
                print('Done at iteration = %d' % (it + 1))
            break

        # 每隔check_every次迭代，检查一次进度。
        if (it + 1) % check_every == 0:
            # 计算损失
            with torch.no_grad():
                loss_mean = sum(loss_list) / p
                mse_mean = sum(mse_list) / p
                ridge_mean = (sum(smooth_list) - sum(mse_list)) / p
                nonsmooth_mean = (sum(loss_list) - sum(smooth_list)) / p

            train_loss_list.append(loss_mean)
            train_mse_list.append(mse_mean)
            
            if verbose > 0:
                print(('-' * 10 + 'Iter = %d' + '-' * 10) % (it + 1))
                print('Total loss = %f' % loss_mean)
                print('MSE = %f, Ridge = %f, Nonsmooth = %f'
                      % (mse_mean, ridge_mean, nonsmooth_mean))
                print('Variable usage = %.2f%%'
                      % (100 * torch.mean(cmlp.GC().float())))

            # 如果当前未启用线搜索。
            if not line_search:
                # 如果最近两次迭代的损失下降小于switch_tol，则切换到线搜索。
                if train_loss_list[-2] - train_loss_list[-1] < switch_tol:
                    line_search = True
                    if verbose > 0:
                        print('Switching to line search')

    return train_loss_list, train_mse_list


# 使用 Adam 优化器训练模型。
def train_model_adam(cmlp, X, lr, max_iter, lam=0, lam_ridge=0, penalty='H',
                     lookback=5, check_every=100, verbose=1):
    lag = cmlp.lag
    p = X.shape[-1]
    loss_fn = nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(cmlp.parameters(), lr=lr)
    train_loss_list = []

    # For early stopping.
    best_it = None
    best_loss = np.inf
    best_model = None

    for it in range(max_iter):
        # Calculate loss.
        loss = sum([loss_fn(cmlp.networks[i](X[:, :-1]), X[:, lag:, i:i+1])
                    for i in range(p)])

        # Add penalty terms.
        if lam > 0:
            loss = loss + sum([regularize(net, lam, penalty)
                               for net in cmlp.networks])
        if lam_ridge > 0:
            loss = loss + sum([ridge_regularize(net, lam_ridge)
                               for net in cmlp.networks])

        # Take gradient step.
        loss.backward()
        optimizer.step()
        cmlp.zero_grad()

        # Check progress.
        if (it + 1) % check_every == 0:
            mean_loss = loss / p
            train_loss_list.append(mean_loss.detach())

            if verbose > 0:
                print(('-' * 10 + 'Iter = %d' + '-' * 10) % (it + 1))
                print('Loss = %f' % mean_loss)

            # Check for early stopping.
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

# 无正则化的训练过程。
def train_unregularized(cmlp, X, lr, max_iter, lookback=5, check_every=100,
                        verbose=1):
    '''Train model with Adam and no regularization.'''
    lag = cmlp.lag
    p = X.shape[-1]
    loss_fn = nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(cmlp.parameters(), lr=lr)
    train_loss_list = []

    # For early stopping.
    best_it = None
    best_loss = np.inf
    best_model = None

    for it in range(max_iter):
        # Calculate loss.
        pred = cmlp(X[:, :-1])
        loss = sum([loss_fn(pred[:, :, i], X[:, lag:, i]) for i in range(p)])

        # Take gradient step.
        loss.backward()
        optimizer.step()
        cmlp.zero_grad()

        # Check progress.
        if (it + 1) % check_every == 0:
            mean_loss = loss / p
            train_loss_list.append(mean_loss.detach())

            if verbose > 0:
                print(('-' * 10 + 'Iter = %d' + '-' * 10) % (it + 1))
                print('Loss = %f' % mean_loss)

            # Check for early stopping.
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
