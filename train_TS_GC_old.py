import os
import sys
import joblib
from matplotlib import rcParams
from datetime import datetime
import torch
import torch.nn as nn # 确保导入nn
from pathlib import Path
import gc
from copy import deepcopy
import numpy as np
from config import *
import pandas as pd
import matplotlib.pyplot as plt

from logger.logger import get_logger, setup_logging
from model.TS_GC import MutiTS_GC

rcParams['font.family'] = 'SimHei'
rcParams['axes.unicode_minus'] = False

# ISTA算法的硬截断，使用软阈值算子 (Soft Thresholding) 把小的权重直接“归零”
def PGD_update(network, lam, lr, penalty):
    hidden, p, lag = network.shape
    if penalty == 'GL': # 组Loss惩罚
        norm = torch.norm(network, dim=(0, 2), keepdim=True) # 得到每一列的L2范数
        # torch.clamp把norm的值限制如果小于lr * lam，就变成lr * lam
        network.data = ((network / torch.clamp(norm, min=(lr * lam)))
                        # 限制如果norm - (lr * lam) 小于 0.0，把其置为0
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

# 稀疏惩罚的结果
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


class TS_GC_Trainer:
    def __init__(self, model, epochs, save_dir, criterion, lr, device, series_num,
                 X_full, Y_full, penalty_type, lasso_param, ridge_param, 
                 check_every=10, lookback=5, logger=None, verbose=1):
        self.model = model
        self.epochs = epochs
        self.save_dir = save_dir
        self.criterion = criterion
        self.logger = logger
        self.base_lr = lr
        self.device = device
        self.check_every = check_every
        
        self.X_full = X_full.to(self.device)
        self.Y_full = Y_full.to(self.device)
        
        self.penalty_type = penalty_type
        self.lasso_param = lasso_param
        self.ridge_param = ridge_param
        self.series_num = series_num
        self.lookback = lookback
        self.verbose = verbose

        self.train_losses = []
        self.best_loss = float('inf')
        self.best_it = None
        self.best_model_state = None

        self.first_layer_params_list = self.model.get_first_layer_weights() # List of weight tensors

    def train(self, save_model=False):
        # 预热参数，前300次不进行稀疏惩罚
        warmup_iters = 300
        
        for ista_iter in range(self.epochs):
            current_smooth_loss, total_mse_loss, ridge_loss_val = self._compute_smooth_loss(self.X_full, self.Y_full)

            # 2. 反向传播计算梯度
            self.model.zero_grad()
            current_smooth_loss.backward()

            with torch.no_grad():
                for param in self.model.parameters():
                    if param.grad is not None:
                        param.data.sub_(self.base_lr * param.grad) # In-place subtraction

            # 4. 第一层权重更新
            if self.lasso_param > 0 and ista_iter > warmup_iters:
                with torch.no_grad():
                    for weight_tensor in self.first_layer_params_list: # 遍历第一层权重张量
                        if weight_tensor is not None:
                             PGD_update(weight_tensor, self.lasso_param, self.base_lr, self.penalty_type)
            
            # 定期检查
            if (ista_iter + 1) % self.check_every == 0:
                with torch.no_grad(): # 在评估时不需要梯度计算
                    eval_smooth_loss, total_mse_loss, ridge_loss_val = self._compute_smooth_loss(self.X_full, self.Y_full)
                    eval_nonsmooth_loss = self._compute_nonsmooth_loss()
                    mean_loss = (eval_smooth_loss + eval_nonsmooth_loss) / self.series_num
                
                self.train_losses.append(mean_loss.item())
                
                if self.verbose > 0:
                    print(f"{'='*10} ISTA Iter = {ista_iter + 1} {'='*10}")
                    print(f"Smooth Loss = {mean_loss.item():.6f}-------MSE Loss = {(total_mse_loss / self.series_num).item():.6f} - ridge_loss = {(ridge_loss_val / self.series_num).item():.6f} - noSmooth Loss = {(eval_nonsmooth_loss / self.series_num).item():.6f}")

                # 非平滑损失为 0 时，提前停止训练
                if eval_nonsmooth_loss == 0:
                    print("非平滑损失为 0，提前停止训练。")
                    if self.best_model_state is not None:
                        self.model.load_state_dict(self.best_model_state)
                    return self.best_loss
            
                # 早停
                if mean_loss < self.best_loss:
                    self.best_loss = mean_loss
                    self.best_it = ista_iter + 1
                    self.best_model_state = deepcopy(self.model.state_dict())
                    if save_model:
                        self.logger.info(f"最优轮数： {self.best_it} ----------最优loss： {self.best_loss:.6f}")
                        torch.save(self.best_model_state, 
                                   os.path.join(self.save_dir, "best_model.pth"))
                        if hasattr(self.model, 'config'):
                             joblib.dump(self.model.config, 
                                       os.path.join(self.save_dir, "model_config.pkl"))
                elif self.best_it is not None and ((ista_iter + 1) - self.best_it) >= self.lookback * self.check_every:
                    if self.verbose > 0:
                        print("Stopping early")
                    if self.best_model_state is not None:
                        self.model.load_state_dict(self.best_model_state)
                    return self.best_loss 
        
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        return self.best_loss

    def _compute_smooth_loss(self, x_data, y_data):
        predictions = self.model(x_data) # Shape: [num_samples, output_window, series_num]
        
        total_mse_loss = 0
        for i in range(self.series_num):
            pred_for_series_i = predictions[:, :, i] # Shape: [num_samples, output_window]
            target_for_series_i = y_data[:, :, i]    # Shape: [num_samples, output_window]
            total_mse_loss += self.criterion(pred_for_series_i, target_for_series_i)
        
        smooth_loss = total_mse_loss + self._ridge_regularize()
        return smooth_loss, total_mse_loss, self._ridge_regularize()

    # 计算非平滑损失
    def _compute_nonsmooth_loss(self):
        total_lasso_loss = 0
        for weight_tensor in self.first_layer_params_list:
            if weight_tensor is not None:
                 total_lasso_loss += lasso_penalty(weight_tensor, self.lasso_param, self.penalty_type)
        return total_lasso_loss

    # 岭回归正则化
    def _ridge_regularize(self):
        ridge_loss = torch.tensor(0.0, device=self.device) # Initialize on correct device
        if self.ridge_param > 0:
            for param in self.model.parameters():
                if param.requires_grad:
                    ridge_loss += torch.sum(param ** 2)
        return self.ridge_param * ridge_loss

    def cleanup(self):
        self.best_model_state = None
        self.train_losses = [float(x) for x in self.train_losses] 
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    def plot_training_curves(self):
        plots_dir = os.path.join(self.save_dir, "training_curves_ista")
        os.makedirs(plots_dir, exist_ok=True)
        
        plt.figure(figsize=(10, 6))
        iterations_plotted = [i * self.check_every for i in range(len(self.train_losses))]
        plt.plot(iterations_plotted, self.train_losses, label='Train Loss (ISTA)')
        plt.title('Training Loss (ISTA)')
        plt.xlabel(f'ISTA Iteration (every {self.check_every} iters)')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(plots_dir, 'ista_training_loss.png'))
        plt.close()

def main():
    run_id = datetime.now().strftime(r'%m-%d_%H-%M-%S')
    save_dir = Path('saved') / run_id
    setup_logging(save_dir)
    train_logger = get_logger() # 日志记录器

    lr = 0.01      
    lasso_param = 0.005 
    ridge_param = 0.001
    penalty_type = 'GSGL' 
    feature_dim = 64
    kernel_size = 5    
    dropout = 0     
    temporal_layers = 3 
    loss_function = nn.MSELoss()
   
    model = MutiTS_GC(
        input_window=INPUT_WINDOW,
        output_window=OUTPUT_WINDOW,
        series_num=SERIES_NUM,
        feature_dim=feature_dim,
        temporal_layers=temporal_layers,
        kernel_size=kernel_size,
        dropout=dropout,
        device=DEVICE
    ).to(DEVICE)

    # 5. Initialize trainer
    trainer = TS_GC_Trainer(
        model=model, 
        epochs=EPOCHS, 
        save_dir=save_dir, 
        criterion=loss_function,
        lr=lr, 
        device=DEVICE,
        series_num=SERIES_NUM,
        X_full=X_DATA,
        Y_full=Y_DATA,
        logger=train_logger,
        penalty_type=penalty_type, 
        lasso_param=lasso_param,
        ridge_param=ridge_param,
        verbose=1
    )
    
    trainer.train(save_model=True)
    trainer.plot_training_curves()

    trainer.cleanup()
    log_message = (
    f"本次所使用的模型参数和训练参数如下：\n"
    f"模型架构:\n"
    f"  - 模型: {model}\n"
    f"模型参数:\n"
    f"  - feature_dim: {feature_dim}\n"
    f"  - kernel_size: {kernel_size}\n"
    f"  - dropout: {dropout}\n"
    f"训练参数:\n"
    f"  - 损失函数: {loss_function}\n"
    f"  - 数据路径: {DATA_PATH}\n"
    f"  - 学习率: {lr}\n"
    f"  - 正则化参数: {ridge_param}\n"
    f"  - 惩罚类型: {penalty_type}\n"
    f"  - Lasso 参数: {lasso_param}\n"
    f"  - 序列数量: {SERIES_NUM}\n"
)

    train_logger.info(log_message)
    
if __name__ == '__main__':
    main()
