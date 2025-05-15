from datetime import datetime
import os
from pathlib import Path
import copy
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
                lambda_ridge=0.01, verbose=1,
                r=0.8, lr_min=1e-8, sigma=0.5, check_every=5,
                monotone=False, m=10, lr_decay=0.5,
                begin_line_search=False, switch_tol=1e-3):
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
        self.early_stop_patience = 3
        self.best_loss_result = float('inf') 
        self.verbose = verbose  # 日志详细程度
        
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
        
        self.train_losses = []
        self.train_mses = []
        self.train_ridges = []
        self.train_penalties = []

        self.val_losses = []
        self.val_mses = []
        self.val_ridges = []
        self.val_penalties = []
        
        # 创建优化器，排除TCN第一层权重
        self.first_layer_params = []
        other_params = []
        
        for i in range(self.series_num):
            layer_name = f"encoder.layers.0.attention.tcn_processors.{i}.network_layers.0.conv1.weight"
            for name, param in model.named_parameters():
                if name == layer_name:
                    self.first_layer_params.append((i, name, param))
                else:
                    other_params.append(param)
        
        # 为除了TCN第一层之外的所有参数创建优化器
        self.optimizer = optim.Adam(other_params, lr=self.lr, weight_decay=0)  # weight_decay=0 因为我们会手动应用ridge惩罚
        
        # 为每个网络的TCN第一层创建单独的学习率
        self.lr_list = [self.lr for _ in range(len(self.first_layer_params))]
        
        # 创建模型的副本，用于线搜索
        self.model_copy = copy.deepcopy(self.model)
        
        # 如果不使用单调线搜索，则为每个网络的TCN第一层保存历史损失
        if not self.monotone:
            self.last_losses = [[] for _ in range(len(self.first_layer_params))]
    
    def ridge_regularize(self):
        """计算模型除了TCN第一层外的所有参数的L2正则化"""
        ridge_loss = 0.0
        for name, param in self.model.named_parameters():
            if any(first_layer_name == name for _, first_layer_name, _ in self.first_layer_params):
                continue  # 跳过TCN的第一层权重
            ridge_loss += torch.sum(param ** 2)
        return self.lambda_ridge * ridge_loss
    
    def train(self):
        """训练模型"""
        line_search = self.begin_line_search  # 是否使用线搜索
        
        for epoch in range(self.epochs):
            # 训练一个 epoch
            epoch_loss, epoch_mse, epoch_ridge, epoch_penalty = self.train_epoch(line_search)
            
            # 验证
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
            
            # 打印当前 epoch 的训练和验证损失
            if self.verbose > 0:
                print(f"Epoch {epoch+1}/{self.epochs}")
                print(f"Train - Total loss: {epoch_loss:.6f}, MSE: {epoch_mse:.6f}, "
                      f"Ridge: {epoch_ridge:.6f}, Penalty: {epoch_penalty:.6f}")
                print(f"Valid - Total loss: {val_loss:.6f}, MSE: {val_mse:.6f}, "
                      f"Ridge: {val_ridge:.6f}, Penalty: {val_penalty:.6f}")
                # 打印格兰杰因果矩阵的稀疏度
                with torch.no_grad():
                    gc_matrix = self.model.GC()
                    nonzero_ratio = torch.count_nonzero(gc_matrix) / (self.series_num * self.series_num)
                    print(f"Variable usage = {nonzero_ratio.item() * 100:.2f}%")
            
            # 如果当前验证损失比之前的最佳损失更好，则保存模型
            if val_loss < self.best_loss_result:
                self.best_loss_result = val_loss
                # self.save_model(epoch)
            
            # 检查是否需要切换到线搜索
            if not line_search and epoch >= 1:
                # 如果最近两个epoch的训练损失下降幅度小于阈值，则切换到线搜索
                if self.train_losses[-2] - self.train_losses[-1] < self.switch_tol:
                    line_search = True
                    if self.verbose > 0:
                        print("Switching to line search")
        
        # 训练结束后，绘制损失曲线
        self.plot_loss_curves()
        
        return self.train_losses, self.train_mses, self.val_losses, self.val_mses
    
    def train_epoch(self, use_line_search=False):
        """训练一个 epoch"""
        self.model.train()
        
        epoch_loss = 0.0
        epoch_mse = 0.0
        epoch_ridge = 0.0
        epoch_penalty = 0.0
        num_batches = 0
        
        # 批量训练
        for batch_idx, (batch_x, batch_y) in enumerate(self.train_loader):
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            batch_loss, batch_mse, batch_ridge, batch_penalty = self.train_batch(batch_x, batch_y, use_line_search)
            
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
    
    def train_batch(self, batch_x, batch_y, use_line_search=False):
        """训练一个批次"""
        if not use_line_search:
            # 不使用线搜索的情况，使用标准优化方法
            return self.train_batch_standard(batch_x, batch_y)
        else:
            # 使用线搜索的情况
            return self.train_batch_line_search(batch_x, batch_y)
    
    def train_batch_standard(self, batch_x, batch_y):
        """使用标准优化方法训练一个批次"""
        self.optimizer.zero_grad()
        predictions = self.model(batch_x)
        
        # 计算MSE损失
        mse_loss = 0
        for i in range(self.series_num):
            mse_loss += self.criterion(predictions[:, :, i:i+1, :], batch_y[:, :, i:i+1, :])
        mse_loss /= self.series_num
        ridge_loss = self.ridge_regularize()
        smooth_loss = mse_loss + ridge_loss # 总平滑损失（MSE + Ridge）
        smooth_loss.backward() # 反向传播，计算得到所有层的梯度
        self.optimizer.step() # 对除了第一层以外的其他参数进行标准优化器更新
        
        # 对TCN第一层应用近端梯度下降
        for i, name, param in self.first_layer_params:
            # 保存当前梯度
            if param.grad is not None:
                with torch.no_grad():
                    # 先执行标准的梯度下降步骤
                    param.data = param - self.lr * param.grad
                    
                    # 再应用近端梯度下降
                    PGD_update(param, self.lambda_reg, self.lr, self.penalty_type)
        
        # 计算非平滑正则化值（仅用于记录，不影响梯度）
        with torch.no_grad():
            nonsmooth_penalty = 0
            for i, name, param in self.first_layer_params:
                nonsmooth_penalty += lasso_penalty(param, self.lambda_reg, self.penalty_type) # 计算Lasso正则化值
            nonsmooth_penalty /= len(self.first_layer_params) if self.first_layer_params else 1
            
            # 总损失 = 平滑损失 + 非平滑正则化
            total_loss = smooth_loss + nonsmooth_penalty
        
        return total_loss.item(), mse_loss.item(), ridge_loss.item(), nonsmooth_penalty.item()
    
    def train_batch_line_search(self, batch_x, batch_y):
        """使用线搜索训练一个批次"""
        # 首先计算当前模型的损失和梯度
        predictions = self.model(batch_x)
        
        # 计算MSE损失
        mse_loss = 0
        for i in range(self.series_num):
            mse_loss += self.criterion(predictions[:, :, i:i+1, :], batch_y[:, :, i:i+1, :])
        mse_loss /= self.series_num
        ridge_loss = self.ridge_regularize()
        smooth_loss = mse_loss + ridge_loss  # 总平滑损失（MSE + Ridge）
        
        # 计算非平滑正则化值（仅用于记录）
        with torch.no_grad():
            nonsmooth_penalty = 0
            for i, name, param in self.first_layer_params:
                nonsmooth_penalty += lasso_penalty(param, self.lambda_reg, self.penalty_type)
            nonsmooth_penalty /= len(self.first_layer_params) if self.first_layer_params else 1
            
            # 当前总损失
            current_loss = smooth_loss + nonsmooth_penalty
        
        # 反向传播计算梯度
        smooth_loss.backward()
        
        # 对非第一层参数应用标准优化方法
        self.optimizer.step()
        
        # 对TCN第一层分别应用线搜索和近端梯度下降
        total_first_layer_loss = 0
        
        # 更新模型副本
        self.model_copy.load_state_dict(self.model.state_dict())
        
        for idx, (i, name, param) in enumerate(self.first_layer_params):
            if param.grad is None:
                continue
                
            # 获取模型副本中对应的参数
            param_copy = None
            for name_copy, p_copy in self.model_copy.named_parameters():
                if name_copy == name:
                    param_copy = p_copy
                    break
            
            if param_copy is None:
                continue
                
            # 线搜索
            step_taken = False
            lr_it = self.lr_list[idx]
            
            # 如果不使用单调线搜索，需要检查历史损失
            if not self.monotone:
                if len(self.last_losses[idx]) == 0:
                    # 第一次迭代，将当前损失添加到历史损失中
                    self.last_losses[idx].append(current_loss.item())
                
                comp_loss = max(self.last_losses[idx])
            else:
                comp_loss = current_loss.item()
            
            # 线搜索循环
            while not step_taken:
                # 在模型副本上应用梯度下降和近端操作
                with torch.no_grad():
                    # 梯度下降
                    param_copy.data = param.data - lr_it * param.grad
                    
                    # 近端操作
                    PGD_update(param_copy, self.lambda_reg, lr_it, self.penalty_type)
                
                # 用更新后的模型副本计算新的损失
                predictions_copy = self.model_copy(batch_x)
                
                # 计算新的MSE损失
                mse_loss_copy = 0
                for j in range(self.series_num):
                    mse_loss_copy += self.criterion(predictions_copy[:, :, j:j+1, :], batch_y[:, :, j:j+1, :])
                mse_loss_copy /= self.series_num
                
                # 计算新的ridge损失
                ridge_loss_copy = 0.0
                for name_copy, p_copy in self.model_copy.named_parameters():
                    if any(first_layer_name == name_copy for _, first_layer_name, _ in self.first_layer_params):
                        continue
                    ridge_loss_copy += torch.sum(p_copy ** 2)
                ridge_loss_copy *= self.lambda_ridge
                
                # 新的平滑损失
                smooth_loss_copy = mse_loss_copy + ridge_loss_copy
                
                # 新的非平滑正则化损失
                nonsmooth_penalty_copy = lasso_penalty(param_copy, self.lambda_reg, self.penalty_type)
                
                # 新的总损失
                new_loss = smooth_loss_copy + nonsmooth_penalty_copy
                
                # 计算容差
                diff_norm = torch.sum((param - param_copy) ** 2)
                tol = (0.5 * self.sigma / lr_it) * diff_norm
                
                # 检查线搜索条件是否满足
                if comp_loss - new_loss > tol:
                    # 接受这一步
                    step_taken = True
                    
                    # 将参数从副本复制到原始模型
                    param.data.copy_(param_copy.data)
                    
                    # 更新学习率
                    self.lr_list[idx] = (self.lr_list[idx] ** (1 - self.lr_decay)) * (lr_it ** self.lr_decay)
                    
                    # 更新历史损失
                    if not self.monotone:
                        if len(self.last_losses[idx]) == self.m:
                            self.last_losses[idx].pop(0)
                        self.last_losses[idx].append(new_loss.item())
                    
                    # 为返回值累加损失
                    total_first_layer_loss += new_loss.item()
                    
                    if self.verbose > 1:
                        print(f"  Taking step for layer {i}, lr = {lr_it:.6f}")
                        print(f"  Gap = {comp_loss - new_loss.item():.6f}, tol = {tol.item():.6f}")
                else:
                    # 减小学习率
                    lr_it *= self.r
                    
                    # 如果学习率太小，则放弃线搜索
                    if lr_it < self.lr_min:
                        if self.verbose > 1:
                            print(f"  Learning rate too small for layer {i}, giving up line search")
                        
                        # 不更新参数，使用原来的损失
                        total_first_layer_loss += current_loss.item()
                        step_taken = True
        
        # 计算平均损失
        avg_first_layer_loss = total_first_layer_loss / len(self.first_layer_params) if self.first_layer_params else 0
        
        # 重新计算当前模型的损失用于返回
        with torch.no_grad():
            predictions = self.model(batch_x)
            
            # 计算MSE损失
            mse_loss = 0
            for i in range(self.series_num):
                mse_loss += self.criterion(predictions[:, :, i:i+1, :], batch_y[:, :, i:i+1, :])
            mse_loss /= self.series_num
            
            # 计算ridge损失
            ridge_loss = self.ridge_regularize()
            
            # 计算非平滑正则化损失
            nonsmooth_penalty = 0
            for i, name, param in self.first_layer_params:
                nonsmooth_penalty += lasso_penalty(param, self.lambda_reg, self.penalty_type)
            nonsmooth_penalty /= len(self.first_layer_params) if self.first_layer_params else 1
            
            # 总损失
            total_loss = mse_loss + ridge_loss + nonsmooth_penalty
        
        return total_loss.item(), mse_loss.item(), ridge_loss.item(), nonsmooth_penalty.item()
    
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
                
                # 前向传播
                predictions = self.model(batch_x)
                
                # 计算MSE损失
                mse_loss = 0
                for i in range(self.series_num):
                    mse_loss += self.criterion(predictions[:, :, i:i+1, :], batch_y[:, :, i:i+1, :])
                mse_loss /= self.series_num
                
                # 计算ridge正则化
                ridge_loss = self.ridge_regularize()
                
                # 计算非平滑正则化
                nonsmooth_penalty = 0
                for i, name, param in self.first_layer_params:
                    nonsmooth_penalty += lasso_penalty(param, self.lambda_reg, self.penalty_type)
                nonsmooth_penalty /= len(self.first_layer_params) if self.first_layer_params else 1
                
                # 总损失
                total_loss = mse_loss + ridge_loss + nonsmooth_penalty
                
                val_loss += total_loss.item()
                val_mse += mse_loss.item()
                val_ridge += ridge_loss.item()
                val_penalty += nonsmooth_penalty.item()
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
            'optimizer_state_dict': self.optimizer.state_dict(),
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