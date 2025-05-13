import os
import torch
from model.TCN_granger.granger_utils import (
    lasso_penalty,
    PGD_update,
)
from copy import deepcopy
import matplotlib.pyplot as plt


class CausalFormerTrainer:
    def __init__(self, model, epoch, save_dir, criterion, lr, device, series_num,
                train_loader, valid_loader, penalty_type, lambda_reg, 
                lambda_ridge=0.01, check_every=5, r=0.8, lr_min=1e-8, 
                sigma=0.5, monotone=False, m=10, lr_decay=0.5,
                begin_line_search=True, switch_tol=1e-3, verbose=1):
        self.model = model
        self.epochs = epoch
        self.save_dir = save_dir
        self.criterion = criterion  # 主MSE损失函数
        self.lr = lr
        self.device = device
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.penalty_type = penalty_type
        self.lambda_reg = lambda_reg  # 非平滑正则化参数
        self.lambda_ridge = lambda_ridge  # 输出层的岭正则化参数
        self.series_num = series_num
        self.early_stop = 3
        self.best_loss_result = float('inf') 
        
        # GISTA相关参数
        self.check_every = check_every
        self.r = r  # 线搜索学习率衰减因子
        self.lr_min = lr_min  # 最小学习率
        self.sigma = sigma  # 线搜索参数
        self.monotone = monotone  # 是否要求单调性
        self.m = m  # 用于非单调线搜索的历史损失数量
        self.lr_decay = lr_decay  # 学习率衰减参数
        self.begin_line_search = begin_line_search  # 是否从线搜索开始
        self.switch_tol = switch_tol  # 切换到线搜索的容差
        self.verbose = verbose  # 日志详细程度

    def ridge_regularize(self, model_layers):
        """计算模型除了TCN第一层外的所有参数的L2正则化"""
        ridge_loss = 0.0
        for name, param in model_layers:
            if "tcn_processors" in name and "network_layers.0.conv1.weight" in name:
                continue  # 跳过TCN的第一层权重
            ridge_loss += torch.sum(param ** 2)
        return self.lambda_ridge * ridge_loss
    
    def train_epoch(self):
        self.model.train()
        
        tcn_first_layers = []
        tcn_first_layer_names = []
        for i in range(self.series_num): # 获取所有TCN的第一层权重参数
            layer_name = f"encoder.layers.0.attention.tcn_processors.{i}.network_layers.0.conv1.weight"
            layer = self.model.encoder.layers[0].attention.tcn_processors[i].network_layers[0].conv1.weight
            tcn_first_layers.append(layer)
            tcn_first_layer_names.append(layer_name)
        
        model_copy = deepcopy(self.model)
        lr_list = [self.lr for _ in range(self.series_num)]# 初始化每个TCN的学习率列表
        
        # 线搜索标志
        line_search = self.begin_line_search
        
        # 用于非单调线搜索
        if not self.monotone:
            last_losses = [[float('inf')] for _ in range(self.series_num)]
        
        # 记录每个TCN是否已收敛
        done = [False for _ in range(self.series_num)]
        
        epoch_loss = 0.0
        epoch_mse = 0.0
        epoch_penalty = 0.0
        epoch_ridge = 0.0
        num_batches = 0
        
        train_loss_list = []
        train_mse_list = []
        
        # 批量训练
        for batch_idx, (batch_x, batch_y) in enumerate(self.train_loader):
            # batch_x, batch_x：[batch_size, input_window, series_num, feature]
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            
            # 计算每个TCN网络的损失和梯度
            mse_list = []
            smooth_list = []  # MSE + Ridge
            loss_list = []    # Smooth + NonSmooth
            
            # 计算基础MSE和平滑损失
            for i in range(self.series_num):
                if done[i]:
                    continue
                
                # 前向传播计算MSE
                self.model.zero_grad()
                predictions = self.model(batch_x)
                mse = self.criterion(predictions[:, :, i:i+1, :], batch_y[:, :, i:i+1, :])
                
                # 计算Ridge正则化
                ridge = self.ridge_regularize([
                    (name, param) for name, param in self.model.named_parameters()
                    if f"tcn_processors.{i}" in name
                ])
                
                # 平滑损失 = MSE + Ridge
                smooth = mse + ridge
                
                # 非平滑正则化项
                with torch.no_grad():
                    nonsmooth = lasso_penalty(tcn_first_layers[i], self.lambda_reg, self.penalty_type)
                    total_loss = smooth + nonsmooth
                
                mse_list.append(mse)
                smooth_list.append(smooth)
                loss_list.append(total_loss)
            
            # 计算梯度
            if len(smooth_list) > 0:
                sum(smooth_list).backward()
            
            # 初始化新的损失列表
            new_mse_list = []
            new_smooth_list = []
            new_loss_list = []
            
            # 对每个TCN网络执行GISTA步骤
            for i in range(self.series_num):
                if done[i]:
                    if len(mse_list) > i:
                        new_mse_list.append(mse_list[i])
                        new_smooth_list.append(smooth_list[i])
                        new_loss_list.append(loss_list[i])
                    continue
                
                # 准备线搜索
                step = False
                lr_it = lr_list[i]
                
                # 获取当前TCN和其副本
                current_tcn = tcn_first_layers[i]
                copy_tcn = getattr(model_copy.encoder.layers[0].attention.tcn_processors[i].network_layers[0].conv1, 'weight')
                
                while not step:
                    # 梯度下降更新副本
                    with torch.no_grad():
                        # 更新模型所有参数
                        for name, param in self.model.named_parameters():
                            if param.grad is None:
                                continue
                            
                            copy_param = None
                            for copy_name, cp in model_copy.named_parameters():
                                if name == copy_name:
                                    copy_param = cp
                                    break
                            
                            if copy_param is not None:
                                if name == tcn_first_layer_names[i]:
                                    # 对TCN第一层使用梯度下降（近端梯度下降会在后面应用）
                                    copy_param.data = param - lr_it * param.grad
                                elif f"tcn_processors.{i}" in name:
                                    # 对当前TCN的其他参数使用梯度下降
                                    copy_param.data = param - lr_it * param.grad
                    
                    # 对TCN第一层应用近端梯度下降
                    PGD_update(copy_tcn, self.lambda_reg, lr_it, self.penalty_type)
                    
                    # 重新计算损失
                    predictions = model_copy(batch_x)
                    mse = self.criterion(predictions[:, :, i:i+1, :], batch_y[:, :, i:i+1, :])
                    
                    # 计算Ridge正则化
                    ridge = self.ridge_regularize([
                        (name, param) for name, param in model_copy.named_parameters()
                        if f"tcn_processors.{i}" in name
                    ])
                    
                    # 平滑损失 = MSE + Ridge
                    smooth = mse + ridge
                    
                    with torch.no_grad():
                        nonsmooth = lasso_penalty(copy_tcn, self.lambda_reg, self.penalty_type)
                        loss = smooth + nonsmooth
                        
                        # 计算线搜索的容忍度
                        tol = 0.0
                        for name, param in self.model.named_parameters():
                            if f"tcn_processors.{i}" in name:
                                for copy_name, copy_param in model_copy.named_parameters():
                                    if name == copy_name:
                                        tol += torch.sum((param - copy_param) ** 2)
                                        break
                        tol = (0.5 * self.sigma / lr_it) * tol
                    
                    # 确定是否接受此步骤
                    comp = loss_list[i] if self.monotone else max(last_losses[i])
                    if not line_search or (comp - loss) > tol:
                        step = True
                        if self.verbose > 1:
                            print(f'Taking step, network i = {i}, lr = {lr_it}')
                            print(f'Gap = {comp - loss}, tol = {tol}')
                        
                        # 为下一次迭代准备
                        new_mse_list.append(mse)
                        new_smooth_list.append(smooth)
                        new_loss_list.append(loss)
                        
                        # 调整初始学习率
                        lr_list[i] = (lr_list[i] ** (1 - self.lr_decay)) * (lr_it ** self.lr_decay)
                        
                        if not self.monotone:
                            if len(last_losses[i]) == self.m:
                                last_losses[i].pop(0)
                            last_losses[i].append(loss)
                    else:
                        # 减小学习率并重试
                        lr_it *= self.r
                        if lr_it < self.lr_min:
                            done[i] = True
                            if len(mse_list) > i:
                                new_mse_list.append(mse_list[i])
                                new_smooth_list.append(smooth_list[i])
                                new_loss_list.append(loss_list[i])
                            if self.verbose > 0:
                                print(f'第 {i+1} 个TCN收敛')
                            break
                
                # 如果接受步骤，交换参数
                if step:
                    # 交换模型参数
                    with torch.no_grad():
                        for name, param in self.model.named_parameters():
                            if f"tcn_processors.{i}" in name:
                                for copy_name, copy_param in model_copy.named_parameters():
                                    if name == copy_name:
                                        # 交换参数
                                        temp = param.data.clone()
                                        param.data = copy_param.data.clone()
                                        copy_param.data = temp
                                        break
            
            # 更新损失列表
            mse_list = new_mse_list
            smooth_list = new_smooth_list
            loss_list = new_loss_list
            
            # 检查是否所有网络都已收敛
            if sum(done) == self.series_num:
                if self.verbose > 0:
                    print('所有模型收敛')
                break
            
            # 计算平均损失
            with torch.no_grad():
                if len(loss_list) > 0:
                    loss_mean = sum(loss_list) / len(loss_list)
                    mse_mean = sum(new_mse_list) / len(new_mse_list)
                    ridge_mean = sum([s - m for s, m in zip(new_smooth_list, new_mse_list)]) / len(new_mse_list)
                    nonsmooth_mean = sum([l - s for l, s in zip(new_loss_list, new_smooth_list)]) / len(new_smooth_list)
                    
                    epoch_loss += loss_mean.item()
                    epoch_mse += mse_mean.item()
                    epoch_ridge += ridge_mean.item()
                    epoch_penalty += nonsmooth_mean.item()
                    num_batches += 1
                    
                    train_loss_list.append(loss_mean.item())
                    train_mse_list.append(mse_mean.item())
            
            # 定期检查进度
            if (batch_idx + 1) % self.check_every == 0 and self.verbose > 0:
                # 检查是否需要切换到线搜索
                if not line_search and len(train_loss_list) >= 2:
                    if train_loss_list[-2] - train_loss_list[-1] < self.switch_tol:
                        line_search = True
                        if self.verbose > 0:
                            print('Switching to line search')
        
        # 计算epoch平均损失
        avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else float('inf')
        avg_epoch_mse = epoch_mse / num_batches if num_batches > 0 else float('inf')
        avg_epoch_ridge = epoch_ridge / num_batches if num_batches > 0 else float('inf')
        avg_epoch_penalty = epoch_penalty / num_batches if num_batches > 0 else float('inf')
            
        return avg_epoch_loss, avg_epoch_mse, avg_epoch_ridge, avg_epoch_penalty

    def valid_epoch(self):
        self.model.eval()
        val_loss = 0.0  # 累计验证集的总损失
        val_mse = 0.0   # 累计验证集的MSE
        val_ridge = 0.0 # 累计验证集的Ridge损失
        val_penalty = 0.0 # 累计验证集的非平滑损失
        
        # 获取所有TCN的第一层权重参数
        tcn_first_layers = []
        for i in range(self.series_num):
            tcn_first_layers.append(
                self.model.encoder.layers[0].attention.tcn_processors[i].network_layers[0].conv1.weight
            )
        
        with torch.no_grad():
            for batch_x, batch_y in self.valid_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                # 计算总体损失
                predictions = self.model(batch_x)
                mse = self.criterion(predictions, batch_y)
                
                # 计算Ridge正则化
                ridge = 0.0
                for name, param in self.model.named_parameters():
                    if "tcn_processors" in name and not any(f"network_layers.0.conv1.weight" in name for i in range(self.series_num)):
                        ridge += torch.sum(param ** 2)
                ridge = self.lambda_ridge * ridge
                
                # 计算非平滑正则化
                penalty = 0.0
                for i in range(self.series_num):
                    penalty += lasso_penalty(tcn_first_layers[i], self.lambda_reg, self.penalty_type)

                total_loss = mse + ridge + penalty
                
                val_mse += mse.item() * batch_x.size(0)
                val_ridge += ridge.item() * batch_x.size(0)
                val_penalty += penalty.item() * batch_x.size(0)
                val_loss += total_loss.item() * batch_x.size(0)
            
            # 计算平均值
            dataset_size = len(self.valid_loader.dataset)
            avg_val_mse = val_mse / dataset_size if dataset_size > 0 else float('inf')
            avg_val_ridge = val_ridge / dataset_size if dataset_size > 0 else float('inf')
            avg_val_penalty = val_penalty / dataset_size if dataset_size > 0 else float('inf')
            avg_val_loss = val_loss / dataset_size if dataset_size > 0 else float('inf')
            
        return avg_val_loss, avg_val_mse, avg_val_ridge, avg_val_penalty
    
    def train(self):
        not_improved_count = 0  # 未改进计数器
        train_losses = []  # 训练集总损失
        train_mses = []    # 训练集MSE
        train_ridges = []  # 训练集Ridge损失
        train_penalties = []  # 训练集非平滑正则化损失

        val_losses = []    # 验证集总损失
        val_mses = []      # 验证集MSE
        val_ridges = []    # 验证集Ridge损失
        val_penalties = []  # 验证集非平滑正则化损失
        for epoch in range(1, self.epochs + 1):
            print(f"==第{epoch}轮训练==")
            
            # 训练一个epoch
            epoch_loss, epoch_mse, epoch_ridge, epoch_penalty = self.train_epoch()
            val_loss, val_mse, val_ridge, val_penalty = self.valid_epoch()
            # 记录训练和验证指标
            train_losses.append(epoch_loss)
            train_mses.append(epoch_mse)
            train_ridges.append(epoch_ridge)
            train_penalties.append(epoch_penalty)

            val_losses.append(val_loss)
            val_mses.append(val_mse)
            val_ridges.append(val_ridge)
            val_penalties.append(val_penalty)
            print('本轮训练集： Train_loss = %f, Val_loss = %f, Train_MSE = %f,  Val_MSE = %f, Train_otherLayer_L2 = %f, Val_otherLayer_L2 = %f, Train_lasso_value = %f, Val_lasso_value = %f'
                  % (epoch_loss, val_loss, epoch_mse, val_mse, epoch_ridge, val_ridge, epoch_penalty, val_penalty))
            
            # 使用总损失进行早停判断，而不仅仅是MSE
            if val_loss <= self.best_loss_result:
                self.best_loss_result = val_loss
                not_improved_count = 0
                print(f"==成功改进，最佳总损失：{val_loss}==")
            else:
                not_improved_count += 1
                print("未改进")
                
            # 早停检查
            if not_improved_count > self.early_stop:
                print(f"在第{epoch}轮触发早停，模型的最终总损失：{val_loss}")
                break
            
        # 绘制训练和验证损失曲线
        self.plot_training_curves(train_losses, train_mses, train_ridges, train_penalties,
                                val_losses, val_mses, val_ridges, val_penalties)
                
        return self.best_loss_result
    
    def plot_training_curves(self, train_losses, train_mses, train_ridges, train_penalties,
                             val_losses, val_mses, val_ridges, val_penalties):
        plt.figure(figsize=(12, 8))

        # 绘制总损失曲线
        plt.subplot(2, 2, 1)
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Total Loss')
        plt.legend()

        # 绘制MSE曲线
        plt.subplot(2, 2, 2)
        plt.plot(train_mses, label='Train MSE')
        plt.plot(val_mses, label='Validation MSE')
        plt.xlabel('Epoch')
        plt.ylabel('MSE')
        plt.title('MSE')
        plt.legend()

        # 绘制Ridge损失曲线
        plt.subplot(2, 2, 3)
        plt.plot(train_ridges, label='Train Ridge')
        plt.plot(val_ridges, label='Validation Ridge')
        plt.xlabel('Epoch')
        plt.ylabel('Ridge Loss')
        plt.title('Ridge Loss')
        plt.legend()

        # 绘制非平滑正则化损失曲线
        plt.subplot(2, 2, 4)
        plt.plot(train_penalties, label='Train Penalty')
        plt.plot(val_penalties, label='Validation Penalty')
        plt.xlabel('Epoch')
        plt.ylabel('Penalty')
        plt.title('Non-smooth Penalty')
        plt.legend()

        plt.tight_layout()
        save_path = os.path.join(self.save_dir, 'training_model.png')
        plt.savefig(save_path)
        
        plt.show()