import os
import torch
import torch.optim as optim
from model.TCN_granger.granger_utils import (
    lasso_penalty,
    PGD_update,
)
import matplotlib.pyplot as plt
import numpy as np


class CausalFormerTrainer:
    def __init__(self, model, epoch, save_dir, criterion, lr, device, series_num,
                train_loader, valid_loader, penalty_type, lambda_reg, 
                lambda_ridge=0.01, verbose=1):
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
        
    def ridge_regularize(self):
        """计算模型除了TCN第一层外的所有参数的L2正则化"""
        ridge_loss = 0.0
        for name, param in self.model.named_parameters():
            if any(first_layer_name == name for _, first_layer_name, _ in self.first_layer_params):
                continue  # 跳过TCN的第一层权重
            ridge_loss += torch.sum(param ** 2)
        return self.lambda_ridge * ridge_loss
    
    def train_epoch(self):
        self.model.train()
        
        epoch_loss = 0.0
        epoch_mse = 0.0
        epoch_penalty = 0.0
        epoch_ridge = 0.0
        num_batches = 0
        
        # 批量训练
        for batch_idx, (batch_x, batch_y) in enumerate(self.train_loader):
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
        
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
                    nonsmooth_penalty += lasso_penalty(param, self.lambda_reg, self.penalty_type)
                nonsmooth_penalty /= len(self.first_layer_params) if self.first_layer_params else 1
                
                # 总损失 = 平滑损失 + 非平滑正则化
                total_loss = smooth_loss + nonsmooth_penalty
                
                # 更新统计信息
                epoch_loss += total_loss.item()
                epoch_mse += mse_loss.item()
                epoch_ridge += ridge_loss.item()
                epoch_penalty += nonsmooth_penalty.item()
                num_batches += 1
        
        # 计算平均损失
        avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else float('inf')
        avg_epoch_mse = epoch_mse / num_batches if num_batches > 0 else float('inf')
        avg_epoch_ridge = epoch_ridge / num_batches if num_batches > 0 else float('inf')
        avg_epoch_penalty = epoch_penalty / num_batches if num_batches > 0 else float('inf')
            
        return avg_epoch_loss, avg_epoch_mse, avg_epoch_ridge, avg_epoch_penalty

    def valid_epoch(self):
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
                
                # 计算Ridge正则化损失
                ridge_loss = self.ridge_regularize()
                
                # 计算非平滑损失
                nonsmooth_penalty = 0
                for i, name, param in self.first_layer_params:
                    nonsmooth_penalty += lasso_penalty(param, self.lambda_reg, self.penalty_type)
                nonsmooth_penalty /= len(self.first_layer_params) if self.first_layer_params else 1
                
                # 总损失
                total_loss = mse_loss + ridge_loss + nonsmooth_penalty
                
                # 更新统计信息
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
    
    def train(self):
        not_improved_count = 0
        train_losses = []
        train_mses = []
        train_ridges = []
        train_penalties = []

        val_losses = []
        val_mses = []
        val_ridges = []
        val_penalties = []
        
        best_model_state = None
        
        for epoch in range(1, self.epochs + 1):
            print(f"==第{epoch}轮训练==")
            
            # 训练和验证
            epoch_loss, epoch_mse, epoch_ridge, epoch_penalty = self.train_epoch()
            val_loss, val_mse, val_ridge, val_penalty = self.valid_epoch()
            
            # 记录指标
            train_losses.append(epoch_loss)
            train_mses.append(epoch_mse)
            train_ridges.append(epoch_ridge)
            train_penalties.append(epoch_penalty)

            val_losses.append(val_loss)
            val_mses.append(val_mse)
            val_ridges.append(val_ridge)
            val_penalties.append(val_penalty)
            
            # 打印进度
            print(f'本轮训练集： Train_loss = {epoch_loss:.6f}, Val_loss = {val_loss:.6f}, '
                  f'Train_MSE = {epoch_mse:.6f}, Val_MSE = {val_mse:.6f}, '
                  f'Train_otherLayer_L2 = {epoch_ridge:.6f}, Val_otherLayer_L2 = {val_ridge:.6f}, '
                  f'Train_Lasso = {epoch_penalty:.6f}, Val_Lasso = {val_penalty:.6f}')
            
            # 模型保存和早停
            if val_loss < self.best_loss_result:
                self.best_loss_result = val_loss
                not_improved_count = 0
                best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                print(f"==成功改进，最佳总损失：{val_loss:.6f}==")
                
                # 保存最佳模型
                # os.makedirs(self.save_dir, exist_ok=True)
                # torch.save(self.model.state_dict(), os.path.join(self.save_dir, 'best_model.pth'))
            else:
                not_improved_count += 1
                print(f"未改进 ({not_improved_count}/{self.early_stop_patience})")
                
            # 早停检查
            if not_improved_count >= self.early_stop_patience:
                print(f"在第{epoch}轮触发早停，模型的最终总损失：{val_loss:.6f}")
                # 恢复最佳模型
                if best_model_state is not None:
                    self.model.load_state_dict(best_model_state)
                break
        
        # 训练结束后绘制损失曲线
        # self.plot_training_curves(train_losses, train_mses, train_ridges, train_penalties,
        #                          val_losses, val_mses, val_ridges, val_penalties)
                
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
        save_path = os.path.join(self.save_dir, 'training_curves.png')
        plt.savefig(save_path)