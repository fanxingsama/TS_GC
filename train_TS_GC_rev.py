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
import pandas as pd
import matplotlib.pyplot as plt

# 假设 config.py 和 logger 均已正确配置
from config import *
from logger.logger import get_logger, setup_logging
# 确保正确导入您的模型
from model.TS_GC import MutiTS_GC

rcParams['font.family'] = 'SimHei'
rcParams['axes.unicode_minus'] = False

# ==========================================
# 辅助函数：PGD更新和惩罚计算
# ==========================================
def PGD_update(network, lam, lr, penalty):
    """近端梯度下降更新，用于实现稀疏性约束"""
    if penalty == 'GL': # Group Lasso
        norm = torch.norm(network, dim=(0, 2), keepdim=True)
        # 软阈值操作
        network.data = ((network / torch.clamp(norm, min=(lr * lam)))
                        * torch.clamp(norm - (lr * lam), min=0.0))
    elif penalty == 'GSGL': # Group Sparse Group Lasso
        # 这一步稍微复杂，简化处理或保持原样
        norm = torch.norm(network, dim=0, keepdim=True)
        network.data = ((network / torch.clamp(norm, min=(lr * lam)))
                  * torch.clamp(norm - (lr * lam), min=0.0))
        norm = torch.norm(network, dim=(0, 2), keepdim=True)
        network.data = ((network / torch.clamp(norm, min=(lr * lam)))
                  * torch.clamp(norm - (lr * lam), min=0.0))
    # ... 其他惩罚类型保持不变 ...

def lasso_penalty(network, lam, penalty):
    """计算稀疏性损失值（用于比较）"""
    if penalty == 'GL':
        return lam * torch.sum(torch.norm(network, dim=(0, 2)))
    elif penalty == 'GSGL':
        return lam * (torch.sum(torch.norm(network, dim=(0, 2)))
                      + torch.sum(torch.norm(network, dim=0)))
    # ... 其他保持不变 ...
    return 0.0

# ==========================================
# 核心训练器类 (支持时间反演)
# ==========================================
class TS_GC_Trainer:
    def __init__(self, model_origin, model_reverse, epochs, save_dir, criterion, lr, device, series_num,
                 X_full, Y_full, X_rev, Y_rev, penalty_type, lasso_param, ridge_param, 
                 check_every=10, lookback=5, logger=None, verbose=1):
        
        # 保存两个模型：原始模型和反演模型
        self.model_origin = model_origin
        self.model_reverse = model_reverse
        
        self.epochs = epochs
        self.save_dir = save_dir
        self.criterion = criterion
        self.logger = logger
        self.base_lr = lr
        self.device = device
        self.check_every = check_every
        
        # 保存两套数据
        self.X_full = X_full.to(self.device)
        self.Y_full = Y_full.to(self.device)
        self.X_rev = X_rev.to(self.device)
        self.Y_rev = Y_rev.to(self.device)
        
        self.penalty_type = penalty_type
        self.lasso_param = lasso_param
        self.ridge_param = ridge_param
        self.series_num = series_num
        self.lookback = lookback
        self.verbose = verbose

        self.train_losses = []
        self.best_loss = float('inf')
        self.best_it = None
        self.best_model_state_origin = None # 需要分别保存状态
        self.best_model_state_reverse = None

        # 获取两个模型的第一层权重列表，用于PGD更新
        self.params_list_origin = self.model_origin.get_first_layer_weights()
        self.params_list_reverse = self.model_reverse.get_first_layer_weights()
        
        # 将两个模型的所有参数放入同一个优化器
        self.all_params = list(self.model_origin.parameters()) + list(self.model_reverse.parameters())
        self.optimizer = torch.optim.Adam(self.all_params, lr=lr)

    def train(self, save_model=False):
        for ista_iter in range(self.epochs):
            self.model_origin.train()
            self.model_reverse.train()
            
            # --- 1. 前向传播与平滑损失计算 (Adam步) ---
            
            # 原始序列
            loss_smooth_o, mse_o, ridge_o = self._compute_smooth_loss(
                self.model_origin, self.X_full, self.Y_full
            )
            # 反演序列
            loss_smooth_r, mse_r, ridge_r = self._compute_smooth_loss(
                self.model_reverse, self.X_rev, self.Y_rev
            )
            
            # 总平滑损失 (用于反向传播)
            total_loss = loss_smooth_o + loss_smooth_r
            
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            # --- 2. 近端梯度更新 (Proximal Step / PGD) ---
            # 仅针对第一层权重应用软阈值，实现稀疏性
            if self.lasso_param > 0:
                with torch.no_grad():
                    # 更新原始模型权重
                    for w in self.params_list_origin:
                        PGD_update(w, self.lasso_param, self.base_lr, self.penalty_type)
                    # 更新反演模型权重
                    for w in self.params_list_reverse:
                        PGD_update(w, self.lasso_param, self.base_lr, self.penalty_type)

            # --- 3. 定期检查、早停与融合逻辑 ---
            if (ista_iter + 1) % self.check_every == 0:
                with torch.no_grad():
                    # 重新计算各项损失用于评估
                    l_smooth_o, _, _ = self._compute_smooth_loss(self.model_origin, self.X_full, self.Y_full)
                    l_smooth_r, _, _ = self._compute_smooth_loss(self.model_reverse, self.X_rev, self.Y_rev)
                    
                    l_nonsmooth_o = self._compute_nonsmooth_loss(self.params_list_origin)
                    l_nonsmooth_r = self._compute_nonsmooth_loss(self.params_list_reverse)
                    
                    # 记录平均总损失
                    mean_loss = (l_smooth_o + l_smooth_r + l_nonsmooth_o + l_nonsmooth_r) / self.series_num
                    self.train_losses.append(mean_loss.item())

                    if self.verbose > 0:
                        self.logger.info(f"Iter {ista_iter+1}: Total Loss={mean_loss:.5f} | "
                                         f"Origin(MSE={mse_o/self.series_num:.4f}, Lasso={l_nonsmooth_o/self.series_num:.4f}) | "
                                         f"Reverse(MSE={mse_r/self.series_num:.4f}, Lasso={l_nonsmooth_r/self.series_num:.4f})")

                    # 早停检查
                    if mean_loss < self.best_loss:
                        self.best_loss = mean_loss
                        self.best_it = ista_iter + 1
                        self.best_model_state_origin = deepcopy(self.model_origin.state_dict())
                        self.best_model_state_reverse = deepcopy(self.model_reverse.state_dict())
                        
                        # [重要] 实时计算并融合GC矩阵进行观察 (可选)
                        # gc_final = self.get_fused_gc()
                        
                    elif self.best_it is not None and ((ista_iter + 1) - self.best_it) >= self.lookback * self.check_every:
                        self.logger.info("早停触发 (Stopping early)")
                        break
        
        # 恢复最佳模型
        if self.best_model_state_origin is not None:
            self.model_origin.load_state_dict(self.best_model_state_origin)
            self.model_reverse.load_state_dict(self.best_model_state_reverse)
            
            if save_model:
                # 1. 保存模型权重
                torch.save(self.best_model_state_origin, os.path.join(self.save_dir, "best_model_origin.pth"))
                
                # 2. [新增] 保存模型配置 (修复报错的关键)
                if hasattr(self.model_origin, 'config'):
                    joblib.dump(self.model_origin.config, 
                              os.path.join(self.save_dir, "model_config.pkl"))
                else:
                    self.logger.warning("模型没有 config 属性，无法保存配置！")

    def _compute_smooth_loss(self, model, x, y):
        """计算平滑部分损失: MSE + Ridge"""
        preds = model(x) # [Batch, Output_window, Series_num]
        
        mse_loss = 0
        for i in range(self.series_num):
            mse_loss += self.criterion(preds[:, :, i], y[:, :, i])
            
        ridge = self._ridge_regularize(model)
        return mse_loss + ridge, mse_loss, ridge

    def _compute_nonsmooth_loss(self, params_list):
        """计算非平滑部分损失: Lasso"""
        loss = 0
        for w in params_list:
            loss += lasso_penalty(w, self.lasso_param, self.penalty_type)
        return loss

    def _ridge_regularize(self, model):
        """Ridge正则化"""
        loss = 0.0
        if self.ridge_param > 0:
            for param in model.parameters():
                loss += torch.sum(param ** 2)
        return self.ridge_param * loss

    def get_fused_gc(self, threshold=0.05):
        """
        [核心算法] 获取融合后的格兰杰因果矩阵 (基于论文 Algorithm 1)
        """
        self.model_origin.eval()
        self.model_reverse.eval()
        
        with torch.no_grad():
            # 1. 获取损失指标
            _, mse_o, _ = self._compute_smooth_loss(self.model_origin, self.X_full, self.Y_full)
            _, mse_r, _ = self._compute_smooth_loss(self.model_reverse, self.X_rev, self.Y_rev)
            lasso_o = self._compute_nonsmooth_loss(self.params_list_origin)
            lasso_r = self._compute_nonsmooth_loss(self.params_list_reverse)
            
            # 转为 Python float 方便比较
            Lp_o = mse_o.item()
            Lp_r = mse_r.item()
            Ls_o = lasso_o.item()
            Ls_r = lasso_r.item()
            
            # 2. 获取原始GC矩阵 (Tensor -> Numpy)
            # GC() 返回 [P, P]
            G_origin = self.model_origin.GC(ignore_kernel=True).cpu().numpy()
            G_reverse = self.model_reverse.GC(ignore_kernel=True).cpu().numpy()
            
            # 3. 融合逻辑 (论文核心)
            # 情况A: 原始模型在预测和稀疏性上都更优 -> 选原始
            if Lp_o < Lp_r and Ls_o < Ls_r:
                self.logger.info("Selection Strategy: Origin Dominant")
                return G_origin
            
            # 情况B: 反演模型在预测和稀疏性上都更优 -> 选反演
            elif Lp_r < Lp_o and Ls_r < Ls_o:
                self.logger.info("Selection Strategy: Reverse Dominant")
                return G_reverse
            
            # 情况C: 互有优劣 -> 融合
            else:
                self.logger.info("Selection Strategy: Fusion (Mix)")
                diff = np.abs(G_origin - G_reverse)
                
                # [修改点] 动态设定阈值
                if threshold is None:
                    # 例如：设为平均差异，或者 0.3
                    threshold = np.mean(diff) 
                    self.logger.info(f"Auto-adjusted Threshold: {threshold:.4f}")
                
                # 创建结果矩阵
                G_final = np.zeros_like(G_origin)
                
                # 差异小 (< threshold): 取平均
                mask_close = diff < threshold
                G_final[mask_close] = (G_origin[mask_close] + G_reverse[mask_close]) / 2.0
                
                # 差异大 (>= threshold): 取最大值 (保留强信号)
                mask_far = ~mask_close
                G_final[mask_far] = np.maximum(G_origin[mask_far], G_reverse[mask_far])
                
                return G_final

    def cleanup(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# ==========================================
# 主程序
# ==========================================
def main():
    run_id = datetime.now().strftime(r'%m-%d_%H-%M-%S')
    save_dir = Path('saved') / run_id
    setup_logging(save_dir)
    logger = get_logger()

    # --- 参数配置 ---
    lr = 0.01      
    lasso_param = 0.02 
    ridge_param = 0.001
    penalty_type = 'GSGL' 
    feature_dim = 64
    kernel_size = 5    
    dropout = 0     
    temporal_layers = 2 
    loss_function = nn.MSELoss()
    
    # --- 数据准备 ---
    # 假设 X_DATA 形状: [Num_samples, Input_window, Series_num]
    # 假设 Y_DATA 形状: [Num_samples, Output_window, Series_num]
    # 注意：这里的反转是简单的维度翻转 (Time维度是dim=1)
    # 严谨的时间反演应该对整个长序列进行反转后再切分窗口，
    # 但如果您的窗口是滑动切分的，直接翻转窗口内的顺序通常也是有效的近似。
    
    logger.info("Generating Time-Reversed Data...")
    X_rev = torch.flip(X_DATA, dims=[1]) # 翻转时间维
    Y_rev = torch.flip(Y_DATA, dims=[1]) # 翻转时间维

    # --- 模型初始化 ---
    logger.info("Initializing Origin and Reverse Models...")
    
    # 1. 原始模型
    model_origin = MutiTS_GC(
        input_window=INPUT_WINDOW,
        output_window=OUTPUT_WINDOW,
        series_num=SERIES_NUM,
        feature_dim=feature_dim,
        temporal_layers=temporal_layers,
        kernel_size=kernel_size,
        dropout=dropout,
        device=DEVICE
    ).to(DEVICE)

    # 2. 反演模型 (使用相同的架构，权重独立初始化)
    # 使用 deepcopy 确保架构完全一致，但我们需要重置参数还是保持独立随机？
    # 通常独立随机初始化即可。deepcopy模型结构。
    model_reverse = deepcopy(model_origin).to(DEVICE)
    # 如果 deepcopy 复制了权重，建议重置权重，或者让它们从同一起点开始训练均可。
    # 这里保持 deepcopy 的权重作为起点（同一起点），让数据驱动差异。

    # --- 训练器初始化 ---
    trainer = TS_GC_Trainer(
        model_origin=model_origin,
        model_reverse=model_reverse,
        epochs=EPOCHS, 
        save_dir=save_dir, 
        criterion=loss_function,
        lr=lr, 
        device=DEVICE,
        series_num=SERIES_NUM,
        X_full=X_DATA,
        Y_full=Y_DATA,
        X_rev=X_rev,        # 传入反演数据
        Y_rev=Y_rev,        # 传入反演数据
        logger=logger,
        penalty_type=penalty_type, 
        lasso_param=lasso_param,
        ridge_param=ridge_param,
        verbose=1
    )
    
    # --- 开始训练 ---
    trainer.train(save_model=True)
    
    # --- 获取最终融合结果 ---
    final_gc_matrix = trainer.get_fused_gc(threshold=0.05)
    
    # 保存结果
    np.savetxt(save_dir / "final_fused_gc.txt", final_gc_matrix, fmt='%.6f')
    logger.info(f"Final GC Matrix saved. Shape: {final_gc_matrix.shape}")
    
    # 可视化 (可选)
    plt.figure(figsize=(10, 8))
    plt.imshow(final_gc_matrix, cmap='viridis', aspect='auto')
    plt.colorbar()
    plt.title("Fused Granger Causality Matrix")
    plt.savefig(save_dir / "fused_gc_heatmap.png")
    plt.close()

    trainer.cleanup()

if __name__ == '__main__':
    main()