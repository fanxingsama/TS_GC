import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import time
import torch
import numpy as np
import pandas as pd
from logger.logger import get_logger, setup_logging
from TCN_granger.granger_utils import (
    prox_group_lasso,
    prox_group_sparse_group_lasso,
    calculate_group_lasso_penalty,
    calculate_group_sparse_group_lasso_penalty
)

from util import read_json, write_json


class CausalFormerTrainer:
    '''
    model：要训练的模型。
    criterion：损失函数。
    metric_ftns：评估指标函数列表。
    optimizer：优化器。
    config：配置对象，包含训练相关的配置信息。
    train_loader：训练数据加载器。
    valid_loader：验证数据加载器（可选）。
    test_loader：测试数据加载器（可选）。
    data_loader：训练数据加载器。
    valid_data_loader：验证数据加载器（可选）。
    lr_scheduler：学习率调度器（可选）。
    lambda_reg：近端梯度下降里，正则化项的强度。
    alpha_gsgl：GSGL内组稀疏和组内稀疏的比例
    lam：正则化项的权重（默认为 0）。
    len_epoch：每个 epoch 的迭代次数（可选，用于迭代式训练）
    '''
    def __init__(self, model, criterion, metrics_fns, optimizer, config, device, series_num,
                 train_loader, valid_loader, penalty_type, lambda_reg, alpha_gsgl,
                 lr_scheduler=None):
        self.model = model
        self.criterion = criterion
        self.metrics_fns = metrics_fns
        self.optimizer = optimizer
        self.device = device
        self.series_num = series_num
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.penalty_type = penalty_type
        self.lambda_reg = lambda_reg
        self.alpha_gsgl = alpha_gsgl
        self.lr_scheduler = lr_scheduler
        self.config = config
        trainer_conf = self.config['trainer']
        self.epochs = trainer_conf['epochs']
        self.save_dir = trainer_conf['save_dir']

    # 一个训练轮次
    def train_epoch(self):
        self.model.train() # 设置模型为训练模式
        granger_weights_param = self.model.encoder.layers[0].attention.tcn_processor.network_layers[0].conv1.weight
        epoch_loss = 0.0
        epoch_penalty = 0.0
        num_batches = 0

        for batch_x, batch_y in self.train_loader:
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

            # 1. 计算主损失和梯度
            self.model.zero_grad()
            predictions = self.model(batch_x) # 形状: [B, T_out, P, F_out]
            # 确保 target 形状匹配 prediction
            main_loss = self.criterion(predictions, batch_y)
            main_loss.backward() # 计算所有参数的梯度

            epoch_loss += main_loss.item()
            num_batches += 1

            # --- 手动执行近端梯度下降更新 ---
            with torch.no_grad():
                current_penalty = torch.tensor(0.0, device=self.deviceDEVICE)
                current_weights = granger_weights_param.data  # 获取当前权重数据
                if self.penalty_type == 'GL':
                    current_penalty = calculate_group_lasso_penalty(current_weights, self.lambda_reg)
                elif self.penalty_type == 'GSGL':
                    current_penalty = calculate_group_sparse_group_lasso_penalty(current_weights, self.lambda_reg, self.alpha_gsgl)
                epoch_penalty += current_penalty.item()

                # 更新整个模型的参数
                for name, param in self.model.named_parameters():
                    if param.grad is None: continue # 跳过没有梯度的参数

                    # 检查是否是需要正则化的权重
                    is_regularized_weight = (param is granger_weights_param)

                    #对第一层使用近端操作符进行近端更新
                    if is_regularized_weight:
                        w_tilde = param.data - self.lr * param.grad  # 梯度下降公式
                        lambda_gamma = self.lr * self.lambda_reg #  是正则化参数，在近端操作中控制正则化的强度。
                        w_new = torch.zeros_like(w_tilde)  # 初始化新的权重张量

                        if self.penalty_type == 'GL':
                            for j in range(w_tilde.shape[1]): # 遍历输入特征
                                w_new[:, j, :] = prox_group_lasso(w_tilde[:, j, :], lambda_gamma)
                        elif self.penalty_type == 'GSGL':
                            for j in range(w_tilde.shape[1]):
                                w_new[:, j, :] = prox_group_sparse_group_lasso(w_tilde[:, j, :], lambda_gamma, self.alpha_gsgl)
                        param.copy_(w_new) # 更新参数
                    else:
                        # 对其他参数执行标准梯度下降
                        param.copy_(param.data - self.lr * param.grad)

        avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else float('inf')
        avg_epoch_penalty = epoch_penalty / num_batches if num_batches > 0 else float('inf')
            
        return avg_epoch_loss, avg_epoch_penalty

    # 验证轮次
    def valid_epoch(self):
        self.model.eval()
        start_time = time.time()  # 记录验证周期开始时间
        val_mse = 0.0 # 累计验证集的均方误差

        with torch.no_grad(): # 禁用梯度计算，以提高评估效率。
            for batch_x, batch_y in self.val_loader:
                batch_x, batch_y = batch_x.to(self.DEVICE), batch_y.to(self.DEVICE)
                predictions = self.model(batch_x)
                loss = self.criterion(predictions, batch_y) # 验证集损失值
                val_mse += loss.item() * batch_x.size(0) # 乘以批次的大小以得到总的损失值
            avg_val_mse = val_mse / len(self.val_loader.dataset) if len(self.val_loader.dataset) > 0 else float('inf')

        final_avg_val_mse = avg_val_mse
        end_time = time.time()  # 记录验证周期结束时间
        val_time = end_time - start_time  # 计算验证时间
        print(f"验证时间: {val_time:.2f} s")
        print(f"验证结果: {final_avg_val_mse}")
            
        return final_avg_val_mse
    
    # 完整的训练过程
    def train(self):
        not_improved_count = 0  # 未改进计数器，用于记录连续未改进的轮数。
        for epoch in range(1, self.epochs + 1):
            print(f"==================第{epoch}轮训练====================")
            result = self.train_epoch(epoch)
            # 监控模型的性能
            best = False
            improved = self.train_metrics['val_loss'] <= self.mnt_best
            if improved:  # 如果改进，则更新最佳性能值，重置未改进计数器，并标记为最佳轮次。
                self.mnt_best = self.train_metrics['val_loss']
                not_improved_count = 0
                best = True
            else:  # 如果未改进，则增加未改进计数器
                not_improved_count += 1
            # 如果未改进计数器超过早停轮数，则停止训练。
            if not_improved_count > self.early_stop:
                self.logger.info(f"一共训练了{epoch}轮，模型的最终结果：{result}")
                break
            # 只在模型性能有改进时保存
            # self._save_checkpoint(epoch, save_best=best)

    # 
    # def generate_causality_matrix(self):
    #     """
    #     计算并保存格兰杰因果关系矩阵。
    #     """
    #     self.logger.info("正在计算格兰杰因果关系矩阵...")
    #     significance_level = self.config.get('causality_params', {}).get('significance_level', 0.05)
        
    #     causal_matrix = self.model.get_granger_causality_matrix(significance_level=significance_level)
        
    #     if causal_matrix is not None:
    #         if isinstance(causal_matrix, torch.Tensor):
    #             causal_matrix_np = causal_matrix.detach().cpu().numpy()
    #         else:
    #             causal_matrix_np = causal_matrix
                
    #         causal_matrix_path_npy = os.path.join(self.save_dir, "granger_causality_matrix.npy")
    #         np.save(causal_matrix_path_npy, causal_matrix_np)
    #         self.logger.info(f"格兰杰因果关系矩阵 (NumPy 数组) 已保存到 {causal_matrix_path_npy}")

    #         if len(causal_matrix_np.shape) == 2 and causal_matrix_np.shape[0] == self.n_series and causal_matrix_np.shape[1] == self.n_series:
    #             try:
    #                 series_names = getattr(self.train_loader.dataset, 'feature_names', [f'series_{i}' for i in range(self.n_series)])
    #                 if len(series_names) != self.n_series:
    #                     series_names = [f'series_{i}' for i in range(self.n_series)]

    #                 df_causal_matrix = pd.DataFrame(causal_matrix_np, index=series_names, columns=series_names)
    #                 causal_matrix_path_csv = os.path.join(self.save_dir, "granger_causality_matrix.csv")
    #                 df_causal_matrix.to_csv(causal_matrix_path_csv)
    #                 self.logger.info(f"格兰杰因果关系矩阵也已另存为 CSV 到 {causal_matrix_path_csv}")
    #             except Exception as e:
    #                 self.logger.error(f"无法将格兰杰因果关系矩阵另存为 CSV: {e}")
    #         elif len(causal_matrix_np.shape) != 2:
    #              self.logger.warning(f"格兰杰因果关系矩阵不是二维的 (形状: {causal_matrix_np.shape})，无法直接另存为 N,N CSV。")
    #         else:
    #             self.logger.warning(f"格兰杰因果关系矩阵是二维的，但形状 {causal_matrix_np.shape} 与 (N={self.n_series}, N={self.n_series}) 不匹配。不另存为 CSV。")
    #     else:
    #         self.logger.warning("格兰杰因果关系矩阵无法计算或模型返回为 None。")

