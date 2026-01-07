import os
import sys
import joblib
from matplotlib import rcParams
from datetime import datetime
import torch
import torch.nn as nn
from pathlib import Path
import gc
from copy import deepcopy
import numpy as np
from config import *
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset # 新增导入

from util.logger import get_logger, setup_logging
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
                 check_every=10, lookback=5, logger=None, verbose=1, batch_size=128):
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
        self.batch_size = batch_size
        
        self.train_dataset = TensorDataset(self.X_full, self.Y_full)
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0 # 如果是 Windows 系统，建议设为 0；Linux 可设为 4
        )
        
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
        
        if isinstance(self.model, nn.DataParallel):
            self.model_core = self.model.module
        else:
            self.model_core = self.model

        self.first_layer_params_list = self.model_core.get_first_layer_weights() 

    def train(self, save_model=False):
        # 预热参数，前300次不进行稀疏惩罚
        warmup_iters = 100
        
        # 计算 total steps 用于 loss 缩放
        total_batches = len(self.train_loader)
        
        for ista_iter in range(self.epochs):
            self.model.train()
            # 1. 在每个 Epoch 开始时清零梯度 (而不是每个 Batch)
            self.model.zero_grad()
            
            total_mse_in_epoch = 0.0
            
            # --- 修改 2: 梯度累积循环 (Gradient Accumulation) ---
            for batch_idx, (batch_x, batch_y) in enumerate(self.train_loader):
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                # 前向传播
                predictions = self.model(batch_x)
                
                # 计算 MSE Loss
                mse_loss = 0
                for i in range(self.series_num):
                    mse_loss += self.criterion(predictions[:, :, i], batch_y[:, :, i])
                
                # Loss 缩放：为了模拟 Full Batch，我们将 Loss 除以 Batch 数量
                # 这样累加后的梯度 magnitude 与全量训练一致
                loss_to_backward = mse_loss / total_batches
                loss_to_backward.backward() # 梯度累加
                
                total_mse_in_epoch += mse_loss.item() # 记录原始 MSE 用于日志

            # --- 修改 3: 参数更新 (每个 Epoch 做一次，等价于 ISTA) ---
            
            # A. 手动添加 Ridge (L2) 正则化的梯度
            if self.ridge_param > 0:
                with torch.no_grad():
                    for param in self.model.parameters():
                        if param.requires_grad and param.grad is not None:
                            # Ridge 导数: 2 * lambda * w
                            param.grad.add_(2 * self.ridge_param * param.data)

            # B. 梯度下降 (Gradient Descent)
            with torch.no_grad():
                for param in self.model.parameters():
                    if param.grad is not None:
                        param.data.sub_(self.base_lr * param.grad)
            
            # C. 近端算子 (Proximal Step / Soft Thresholding)
            if self.lasso_param > 0 and ista_iter > warmup_iters:
                with torch.no_grad():
                    for weight_tensor in self.first_layer_params_list: # 遍历第一层权重张量
                        if weight_tensor is not None:
                             PGD_update(weight_tensor, self.lasso_param, self.base_lr, self.penalty_type)
            
            # --- 定期检查与评估 ---
            if (ista_iter + 1) % self.check_every == 0:
                # 使用辅助函数分批计算 Loss，防止评估时 OOM
                mean_loss, avg_mse, avg_ridge, avg_nonsmooth = self._evaluate_on_loader()
                
                self.train_losses.append(mean_loss)
                
                if self.verbose > 0:
                    print(f"{'='*10} ISTA Iter = {ista_iter + 1} {'='*10}")
                    print(f"Smooth Loss = {mean_loss:.6f} (MSE: {avg_mse:.6f}, Ridge: {avg_ridge:.6f}) - NonSmooth = {avg_nonsmooth:.6f}")

                # 非平滑损失为 0 时，提前停止训练
                if avg_nonsmooth == 0 and ista_iter > warmup_iters:
                    print("非平滑损失为 0，提前停止训练。")
                    if self.best_model_state is not None:
                        self.model_core.load_state_dict(self.best_model_state)
                    return self.best_loss
            
                # 早停逻辑
                if mean_loss < self.best_loss:
                    self.best_loss = mean_loss
                    self.best_it = ista_iter + 1
                    self.best_model_state = deepcopy(self.model_core.state_dict())
                    if save_model:
                        self.logger.info(f"最优轮数： {self.best_it} ----------最优loss： {self.best_loss:.6f}")
                        torch.save(self.best_model_state, 
                                   os.path.join(self.save_dir, "best_model.pth"))
                        if hasattr(self.model_core, 'config'):
                             joblib.dump(self.model_core.config, 
                                       os.path.join(self.save_dir, "model_config.pkl"))
                elif self.best_it is not None and ((ista_iter + 1) - self.best_it) >= self.lookback * self.check_every:
                    if self.verbose > 0:
                        print("Stopping early")
                    if self.best_model_state is not None:
                        self.model_core.load_state_dict(self.best_model_state)
                    return self.best_loss 
        
        if self.best_model_state is not None:
            self.model_core.load_state_dict(self.best_model_state)
        return self.best_loss

    def _evaluate_on_loader(self):
        """
        辅助函数：分批次在整个数据集上计算 Loss，避免 OOM。
        """
        self.model.eval()
        total_mse_sum = 0.0
        
        # 创建一个临时的 loader 用于评估，不打乱顺序， batch_size 可以稍大一点
        eval_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False)
        
        with torch.no_grad():
            for bx, by in eval_loader:
                bx, by = bx.to(self.device), by.to(self.device)
                preds = self.model(bx)
                
                # 计算当前 batch 所有序列的 MSE 总和
                batch_mse = 0
                for i in range(self.series_num):
                    batch_mse += self.criterion(preds[:, :, i], by[:, :, i])
                
                # criterion 默认是 mean，所以这里还原为 batch 内的总和，或者直接累加 mean * batch_count
                # 为了简单和准确，我们假设 criterion 是 mean 模式
                # 实际上直接累加 batch_mse (它是 batch 均值) 最后除以 loader 长度也是一种近似，
                # 但更准确的是加权平均。
                total_mse_sum += batch_mse.item() * (bx.size(0) / len(self.train_dataset)) * len(eval_loader)
                # 上面公式化简： sum(batch_mean * batch_size) / total_samples 
                # 这里为了保持逻辑简单，直接累加 batch 的 loss，最后除以 batch 数量作为全量 loss 的估计
                
            # 计算平均 MSE (per sample approx)
            avg_mse_loss = total_mse_sum / len(eval_loader)

            # 计算 Ridge (只与权重有关，无需分批)
            ridge_loss = 0.0
            if self.ridge_param > 0:
                for param in self.model.parameters():
                    ridge_loss += torch.sum(param ** 2).item()
            ridge_loss = self.ridge_param * ridge_loss

            # 计算 Nonsmooth (Lasso)
            nonsmooth_loss = self._compute_nonsmooth_loss().item()
            
            # 计算总 Loss (根据原代码逻辑: (Smooth + NonSmooth) / SeriesNum)
            # Smooth = MSE + Ridge
            total_smooth = avg_mse_loss + ridge_loss
            mean_loss = (total_smooth + nonsmooth_loss) / self.series_num
            
            return mean_loss, avg_mse_loss / self.series_num, ridge_loss / self.series_num, nonsmooth_loss / self.series_num

    # 计算非平滑损失
    def _compute_nonsmooth_loss(self):
        total_lasso_loss = 0
        for weight_tensor in self.first_layer_params_list:
            if weight_tensor is not None:
                 total_lasso_loss += lasso_penalty(weight_tensor, self.lasso_param, self.penalty_type)
        return total_lasso_loss

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
    run_id = datetime.now().strftime(r'%Y-%m-%d_%H-%M-%S')
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
    
    # 启用多显卡 DataParallel
    # if torch.cuda.device_count() > 1:
    #     print(f"检测到 {torch.cuda.device_count()} 张显卡，启用 DataParallel 并行训练！")
    #     model = nn.DataParallel(model)
    
    model = model.to(DEVICE)

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
        verbose=1,
        batch_size=BATCH_SIZE
    )
    
    trainer.train(save_model=True)
    trainer.plot_training_curves()
    trainer.cleanup()
    
    # 记录日志 (如果用了DataParallel，要打印 module 的信息)
    log_model_ref = model.module if isinstance(model, nn.DataParallel) else model
    
    log_message = (
    f"本次所使用的模型参数和训练参数如下：\n"
    f"模型架构:\n"
    f"  - 模型: {log_model_ref}\n"
    f"模型参数:\n"
    f"  - feature_dim: {feature_dim}\n"
    f"  - kernel_size: {kernel_size}\n"
    f"  - input_window: {INPUT_WINDOW}\n"
    f"  - output_window: {OUTPUT_WINDOW}\n"
    f"  - dropout: {dropout}\n"
    f"训练参数:\n"
    f"  - 损失函数: {loss_function}\n"
    f"  - 数据路径: {DATA_PATH}\n"
    f"  - 学习率: {lr}\n"
    f"  - Batch Size: {BATCH_SIZE}\n"
    f"  - 正则化参数: {ridge_param}\n"
    f"  - 惩罚类型: {penalty_type}\n"
    f"  - Lasso 参数: {lasso_param}\n"
    f"  - 序列数量: {SERIES_NUM}\n"
)

    train_logger.info(log_message)
    
if __name__ == '__main__':
    main()