from datetime import datetime
import os
from pathlib import Path
from matplotlib import rcParams
import torch
import torch.optim as optim
from data_loader import TimeSeriesDataloader
import torch.nn as nn
from model.Granger_causalFormer import PredictModel
from model.TCN_granger.granger_utils import (
    lasso_penalty,
    PGD_update
)
import matplotlib.pyplot as plt

rcParams['font.family'] = 'SimHei'
rcParams['axes.unicode_minus'] = False

class CausalFormerTrainer2:
    def __init__(self, model, epoch, save_dir, criterion, lr, device, series_num,
                train_loader, valid_loader, penalty_type, lambda_reg, 
                lambda_ridge=0.01, r=0.8, lr_min=1e-8, sigma=0.5, check_every=5,
                monotone=False, m=10, lr_decay=0.5,
                begin_line_search=False, switch_tol=1e-3, verbose=1):
        self.model = model
        self.epochs = epoch
        self.save_dir = save_dir
        self.criterion = criterion  # 主MSE损失函数
        self.base_lr = lr  # 基础学习率
        self.device = device
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.penalty_type = penalty_type
        self.lambda_reg = lambda_reg  # 非平滑正则化参数
        self.lambda_ridge = lambda_ridge  # 输出层的岭正则化参数
        self.series_num = series_num
        self.early_stop_patience = 3
        self.best_loss_result = float('inf') 
        
        # 线搜索参数
        self.r = r  # 学习率衰减因子
        self.lr_min = lr_min  # 最小学习率
        self.sigma = sigma  # 线搜索停止条件的参数
        self.check_every = check_every  # 检查收敛的频率
        self.monotone = monotone  # 是否要求损失单调下降
        self.m = m  # 非单调线搜索的历史损失数量
        self.lr_decay = lr_decay  # 学习率调整的衰减率
        self.begin_line_search = begin_line_search  # 是否从训练开始就使用线搜索
        self.switch_tol = switch_tol  # 切换到线搜索的容差
        self.verbose = verbose  # 日志输出的详细程度
        
        self.train_losses = []
        self.train_mses = []
        self.train_ridges = []
        self.train_penalties = []

        self.val_losses = []
        self.val_mses = []
        self.val_ridges = []
        self.val_penalties = []
        
        # 为每个TCN网络创建独立的参数组和学习率
        self.tcn_param_groups = [[] for _ in range(series_num)]
        self.tcn_optimizers = []
        self.tcn_lrs = [self.base_lr for _ in range(series_num)]
        self.converged_networks = [False] * series_num
        
        # 非TCN参数
        self.non_tcn_params = []
        _all_tcn_params_identity_set = set()
                        
         # 第一遍：填充tcn_param_groups并识别所有TCN特定的参数
        for i in range(self.series_num):
            # self.tcn_param_groups[i] 已经是初始化好的空列表
            for name, param in model.named_parameters():
                if f"encoder.layers.0.attention.tcn_processors.{i}" in name:
                    self.tcn_param_groups[i].append(param)
                    _all_tcn_params_identity_set.add(param) # 添加参数对象本身到集合中

        # 第二遍：填充non_tcn_params
        # self.non_tcn_params 已经是初始化好的空列表
        for param in model.parameters(): # model.parameters() 返回唯一的参数对象
            if param not in _all_tcn_params_identity_set: # 通过对象ID检查参数是否已在TCN参数集中
                self.non_tcn_params.append(param)
        
        # 为每个TCN创建独立的优化器
        for i in range(series_num):
            self.tcn_optimizers.append(optim.Adam(self.tcn_param_groups[i], lr=self.tcn_lrs[i], weight_decay=0))
        
        # 为非TCN参数创建优化器
        self.non_tcn_optimizer = optim.Adam(self.non_tcn_params, lr=self.base_lr, weight_decay=0)
        
        # 保存模型状态字典，在需要时重新加载
        self.model_state_dict = None
        
        # 如果不使用单调线搜索，则为每个网络保存历史损失
        if not self.monotone:
            self.last_losses = [[] for _ in range(series_num)]
    
    # 岭正则化所有参数
    def ridge_regularize(self):
        """计算模型除了TCN第一层外的所有参数的L2正则化"""
        ridge_loss = 0.0
        for name, param in self.model.named_parameters():
            if any(first_layer_name == name for _, first_layer_name, _ in self.first_layer_params):
                continue  # 跳过TCN的第一层权重
            ridge_loss += torch.sum(param ** 2)
        return self.lambda_ridge * ridge_loss
    
    def train(self):
        line_search = self.begin_line_search  # 是否使用线搜索
        
        for epoch in range(self.epochs):
            epoch_loss, epoch_mse, epoch_ridge, epoch_penalty = self.train_epoch(line_search)
            val_loss, val_mse, val_ridge, val_penalty = self.validate()
            
            # 保存训练和验证的损失
            self.train_losses.append(epoch_loss)
            self.train_mses.append(epoch_mse)
            self.train_ridges.append(epoch_ridge)
            self.train_penalties.append(epoch_penalty)
            
            self.val_losses.append(val_loss)
            self.val_mses.append(val_mse)
            self.val_ridges.append(val_ridge)
            self.val_penalties.append(val_penalty)
            
            if self.verbose > 0:
                print(f"Epoch {epoch+1}/{self.epochs}")
                print(f"Train - loss: {epoch_loss:.6f}, MSE: {epoch_mse:.6f}, "
                      f"Ridge: {epoch_ridge:.6f}, Penalty: {epoch_penalty:.6f}")
                print(f"Valid - loss: {val_loss:.6f}, MSE: {val_mse:.6f}, "
                      f"Ridge: {val_ridge:.6f}, Penalty: {val_penalty:.6f}")
                print(f"网络收敛: {sum(self.converged_networks)}/{len(self.converged_networks)}")
            
            # 如果当前验证损失比之前的最佳损失更好，则保存模型
            if val_loss < self.best_loss_result:
                self.best_loss_result = val_loss
                # if self.save_dir:
                #     self.save_model(epoch)
            
            # 如果所有网络都已收敛，可以提前结束训练
            if all(self.converged_networks) and self.verbose > 0:
                print(f"所有的网络都收敛了")
                break
            
            # 在一定epochs后切换到线搜索
            if not line_search and epoch > 0:
                if self.val_losses[-2] - self.val_losses[-1] < self.switch_tol:
                    line_search = True
                    if self.verbose > 0:
                        print("容差变大，切换到线搜索")
        
        # 训练结束后，绘制损失曲线
        # if self.save_dir:
        #     self.plot_loss_curves()
        
        return self.best_loss_result
    
    def train_epoch(self, use_line_search=True):
        self.model.train()
        
        epoch_loss = 0.0
        epoch_mse = 0.0
        epoch_ridge = 0.0
        epoch_penalty = 0.0
        num_batches = 0
        
        # 批量训练
        for batch_idx, (batch_x, batch_y) in enumerate(self.train_loader):
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            
            # 在每个batch开始时将所有优化器的梯度清零
            self.non_tcn_optimizer.zero_grad()
            for optimizer in self.tcn_optimizers:
                optimizer.zero_grad()
            
            if not use_line_search:
                # 不使用线搜索的情况，使用标准优化方法
                batch_loss, batch_mse, batch_ridge, batch_penalty = self.train_batch_standard(batch_x, batch_y)
            else:
                # 使用线搜索的情况
                batch_loss, batch_mse, batch_ridge, batch_penalty = self.train_batch_line_search(batch_x, batch_y)

            epoch_loss += batch_loss
            epoch_mse += batch_mse
            epoch_ridge += batch_ridge
            epoch_penalty += batch_penalty
            num_batches += 1
        
        # 计算平均损失
        avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else float('inf')
        avg_epoch_mse = epoch_mse / num_batches if num_batches > 0 else float('inf')
        avg_epoch_ridge = epoch_ridge / num_batches if num_batches > 0 else float('inf')
        avg_epoch_penalty = epoch_penalty / num_batches if num_batches > 0 else float('inf')
            
        return avg_epoch_loss, avg_epoch_mse, avg_epoch_ridge, avg_epoch_penalty
    
    def train_batch_standard(self, batch_x, batch_y):
        predictions = self.model(batch_x)

        mse_loss = 0
        for i in range(self.series_num):
            mse_loss += self.criterion(predictions[:, :, i:i+1, :], batch_y[:, :, i:i+1, :])
        mse_loss /= self.series_num
        
        # 计算ridge正则化
        # ridge_loss = self.ridge_regularize()
        ridge_loss = 0
        
        smooth_loss = mse_loss + ridge_loss
        smooth_loss.backward()
        self.non_tcn_optimizer.step()
        nonsmooth_penalty = 0.0
        
        # 对每个TCN应用优化器更新和近端梯度下降（对第一层）
        for i in range(self.series_num):
            # 找到TCN的第一层参数
            first_layer_name = f"encoder.layers.0.attention.tcn_processors.{i}.network_layers.0.conv1.weight"
            first_layer_param = None
            
            for name, param in self.model.named_parameters():
                if name == first_layer_name:
                    first_layer_param = param
                    break
            
            if first_layer_param is not None:
                # 暂存当前梯度
                if first_layer_param.grad is not None:
                    grad_backup = first_layer_param.grad.clone()
                    # 应用普通的优化器步骤
                    self.tcn_optimizers[i].step()
                    # 再应用近端梯度下降到第一层
                    with torch.no_grad():
                        PGD_update(first_layer_param, self.lambda_reg, self.tcn_lrs[i], self.penalty_type)
                    # 计算非平滑正则化值
                    nonsmooth_penalty += lasso_penalty(first_layer_param, self.lambda_reg, self.penalty_type)
            else:
                # 如果没有找到第一层，只进行普通优化
                self.tcn_optimizers[i].step()
        
        # 计算平均非平滑正则化值
        nonsmooth_penalty /= self.series_num
        
        # 重新计算当前模型的总损失
        with torch.no_grad():
            total_loss = smooth_loss + nonsmooth_penalty
        
        return total_loss, mse_loss, ridge_loss, nonsmooth_penalty
    
    def train_batch_line_search(self, batch_x, batch_y):
        # 计算当前损失
        predictions = self.model(batch_x)
        mse_loss = 0
        for i in range(self.series_num):
            mse_loss += self.criterion(predictions[:, :, i:i+1, :], batch_y[:, :, i:i+1, :])
        mse_loss /= self.series_num
        # ridge_loss = self.ridge_regularize()
        ridge_loss = 0
        smooth_loss = mse_loss + ridge_loss
        
        # 计算非平滑正则化值
        nonsmooth_penalty = 0.0
        for i in range(self.series_num):
            first_layer_name = f"encoder.layers.0.attention.tcn_processors.{i}.network_layers.0.conv1.weight"
            for name, param in self.model.named_parameters():
                if name == first_layer_name:
                    nonsmooth_penalty += lasso_penalty(param, self.lambda_reg, self.penalty_type)
                    break
        nonsmooth_penalty /= self.series_num
        
        # 计算总损失
        current_loss = smooth_loss + nonsmooth_penalty
        smooth_loss.backward() # 反向传播梯度
        self.non_tcn_optimizer.step() # 更新非TCN参数
        # self.model_copy.load_state_dict(self.model.state_dict()) # 保存模型状态
        # 保存当前模型状态字典而不是深拷贝整个模型
        self.model_state_dict = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
        
        # 对每个TCN进行线搜索
        for i in range(self.series_num):
            if self.converged_networks[i]:  # 如果网络已收敛，跳过
                continue
                
            # 获取TCN的第一层参数
            first_layer_name = f"encoder.layers.0.attention.tcn_processors.{i}.network_layers.0.conv1.weight"
            first_layer_param = None
            for name, param in self.model.named_parameters():
                if name == first_layer_name:
                    first_layer_param = param # 得到第一层的参数
                    break
            
            if first_layer_param is None or first_layer_param.grad is None:
                print('first_layer_param is None or first_layer_param.grad is None')
                continue
                
            # 准备线搜索
            step_taken = False
            lr_it = self.tcn_lrs[i]
            
            # 获取比较损失
            if not self.monotone:
                if len(self.last_losses[i]) == 0:
                    self.last_losses[i].append(current_loss)
                comp_loss = max(self.last_losses[i])
            else:
                comp_loss = current_loss
            
            # 开始线搜索
            while not step_taken: # 直到step_taken为True时才会停止
                 # 在每次线搜索迭代中，恢复原始模型状态
                if not step_taken and lr_it != self.tcn_lrs[i]:
                    self.model.load_state_dict(self.model_state_dict) # 只有当我们降低学习率且尚未成功时，才需要恢复
                    
                 # 应用梯度下降和近端梯度下降
                with torch.no_grad():
                    for name, param in self.model.named_parameters():
                        if f"encoder.layers.0.attention.tcn_processors.{i}" in name:  # 在TCN中
                            if param.grad is not None:
                                param.data = param.data - lr_it * param.grad  # 应用梯度下降
                                if name == first_layer_name:  # 对第一层应用近端梯度下降
                                    PGD_update(param, self.lambda_reg, lr_it, self.penalty_type)
                
                
                
                # 用更新后的模型计算新的损失
                predictions_updated = self.model(batch_x)
                mse_loss_copy = 0 # 计算新的mse损失
                ridge_loss_copy = 0.0 # 计算新的ridge损失
                for j in range(self.series_num):
                    mse_loss_copy += self.criterion(predictions_updated[:, :, j:j+1, :], batch_y[:, :, j:j+1, :])
                mse_loss_copy /= self.series_num
                # for name, param in self.model_copy.named_parameters():
                #     ridge_loss_copy += torch.sum(param ** 2)
                # ridge_loss_copy *= self.lambda_ridge
                
                # smooth_loss_copy = mse_loss_copy + ridge_loss_copy # 计算新的平滑损失
                smooth_loss_copy = mse_loss_copy # 计算新的平滑损失
                
                # 计算新的非平滑正则化值
                nonsmooth_penalty_copy = 0.0
                for j in range(self.series_num):
                    first_layer_name_j = f"encoder.layers.0.attention.tcn_processors.{j}.network_layers.0.conv1.weight"
                    for name, param in self.model.named_parameters():
                        if name == first_layer_name_j:
                            nonsmooth_penalty_copy += lasso_penalty(param, self.lambda_reg, self.penalty_type)
                            break
                nonsmooth_penalty_copy /= self.series_num

                new_loss = smooth_loss_copy + nonsmooth_penalty_copy # 得到这轮搜索的总损失
                
                # 计算TCN参数差异
                diff_norm = 0.0
                for name, param in self.model.named_parameters():
                    if f"encoder.layers.0.attention.tcn_processors.{i}" in name:
                        # 获取原始参数
                        orig_param = self.model_state_dict[name].clone().detach()
                        diff_norm += torch.sum((param - orig_param) ** 2)
                
                # 计算线搜索容差
                tol = (0.5 * self.sigma / lr_it) * diff_norm
                
                # 检查线搜索条件
                if comp_loss - new_loss > tol:
                    step_taken = True
                    # for name, param in self.model_copy.named_parameters(): # 更新模型参数
                    #     if f"encoder.layers.0.attention.tcn_processors.{i}" in name:
                    #         for orig_name, p in self.model.named_parameters():
                    #             if orig_name == name:
                    #                 p.data.copy_(param.data)
                    #                 break
                    # 更新学习率
                    self.tcn_lrs[i] = (self.tcn_lrs[i] ** (1 - self.lr_decay)) * (lr_it ** self.lr_decay)
                    
                    # 更新历史损失
                    if not self.monotone:
                        if len(self.last_losses[i]) == self.m:
                            self.last_losses[i].pop(0)
                        self.last_losses[i].append(new_loss)
                else:
                    # 减小学习率
                    lr_it *= self.r
                    if lr_it < self.lr_min:
                        # 学习率太小，标记网络已收敛
                        self.converged_networks[i] = True
                        if self.verbose > 0:
                            print(f"  Network {i} 收敛 (lr too small)")
                        step_taken = True
                    else:
                        # 如果没有成功，我们需要在下一次迭代中恢复模型状态
                        self.model.load_state_dict(self.model_state_dict)
        
        # 如果所有网络都已收敛，记录日志
        if all(self.converged_networks) and self.verbose > 0:
            print("所有的网络都收敛了")
            
        if self.verbose > 0 and sum(self.converged_networks) > 0:
            print(f" {sum(self.converged_networks)}/{self.series_num} 的网络收敛了")
        
        # 重新计算当前模型的损失用于返回
        with torch.no_grad():
            predictions = self.model(batch_x)
            
            # 计算MSE损失
            batch_mse = 0
            for i in range(self.series_num):
                batch_mse += self.criterion(predictions[:, :, i:i+1, :], batch_y[:, :, i:i+1, :])
            batch_mse /= self.series_num
            
            # 计算ridge损失
            # ridge_loss = self.ridge_regularize()
            batch_ridge = 0
            
            # 计算非平滑正则化损失
            batch_lasso_penalty = 0.0
            for i in range(self.series_num):
                first_layer_name = f"encoder.layers.0.attention.tcn_processors.{i}.network_layers.0.conv1.weight"
                for name, param in self.model.named_parameters():
                    if name == first_layer_name:
                        batch_lasso_penalty += lasso_penalty(param, self.lambda_reg, self.penalty_type)
                        break
            batch_lasso_penalty /= self.series_num
            
            # 总损失
            batch_loss = batch_mse + batch_ridge + batch_lasso_penalty
        
        return batch_loss, batch_mse, batch_ridge, batch_lasso_penalty
    
    def validate(self):
        """验证模型性能"""
        self.model.eval()
        
        val_loss = 0.0
        val_mse = 0.0
        val_ridge = 0.0
        val_penalty = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_x, batch_y in self.valid_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                predictions = self.model(batch_x)

                mse_loss = 0
                for i in range(self.series_num):
                    mse_loss += self.criterion(predictions[:, :, i:i+1, :], batch_y[:, :, i:i+1, :])
                mse_loss /= self.series_num
                # ridge_loss = self.ridge_regularize()
                ridge_loss = 0

                nonsmooth_penalty = 0.0
                for i in range(self.series_num):
                    first_layer_name = f"encoder.layers.0.attention.tcn_processors.{i}.network_layers.0.conv1.weight"
                    for name, param in self.model.named_parameters():
                        if name == first_layer_name:
                            nonsmooth_penalty += lasso_penalty(param, self.lambda_reg, self.penalty_type)
                            break
                nonsmooth_penalty /= self.series_num
                
                # 总损失
                total_loss = mse_loss + ridge_loss + nonsmooth_penalty
                
                val_loss += total_loss
                val_mse += mse_loss
                val_ridge += ridge_loss
                val_penalty += nonsmooth_penalty
                num_batches += 1
        
        # 计算平均损失
        avg_val_loss = val_loss / num_batches if num_batches > 0 else float('inf')
        avg_val_mse = val_mse / num_batches if num_batches > 0 else float('inf')
        avg_val_ridge = val_ridge / num_batches if num_batches > 0 else float('inf')
        avg_val_penalty = val_penalty / num_batches if num_batches > 0 else float('inf')
        
        return avg_val_loss, avg_val_mse, avg_val_ridge, avg_val_penalty
    
    def save_model(self, epoch):
        """保存模型"""
        save_path = os.path.join(self.save_dir, f"model_epoch_{epoch}.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'non_tcn_optimizer_state_dict': self.non_tcn_optimizer.state_dict(),
            'tcn_optimizers_state_dict': [opt.state_dict() for opt in self.tcn_optimizers],
            'tcn_lrs': self.tcn_lrs,
            'converged_networks': self.converged_networks,
            'train_loss': self.train_losses[-1],
            'val_loss': self.val_losses[-1],
        }, save_path)
    
        if self.verbose > 0:
            print(f"Model saved to {save_path}")
    
    def plot_loss_curves(self):
        """绘制损失曲线"""
        # 创建保存图表的目录
        plots_dir = os.path.join(self.save_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        # 绘制训练和验证的总损失
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.title('Total Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(plots_dir, 'total_loss.png'))
        plt.close()
        
        # 绘制训练和验证的MSE损失
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_mses, label='Train MSE')
        plt.plot(self.val_mses, label='Validation MSE')
        plt.title('MSE Loss')
        plt.xlabel('Epoch')
        plt.ylabel('MSE')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(plots_dir, 'mse_loss.png'))
        plt.close()
        
        # 绘制训练和验证的Ridge正则化损失
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_ridges, label='Train Ridge')
        plt.plot(self.val_ridges, label='Validation Ridge')
        plt.title('Ridge Regularization')
        plt.xlabel('Epoch')
        plt.ylabel('Ridge Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(plots_dir, 'ridge_loss.png'))
        plt.close()
        
        # 绘制训练和验证的非平滑正则化损失
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_penalties, label='Train Penalty')
        plt.plot(self.val_penalties, label='Validation Penalty')
        plt.title('Non-smooth Regularization')
        plt.xlabel('Epoch')
        plt.ylabel('Penalty Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(plots_dir, 'penalty_loss.png'))
        plt.close()