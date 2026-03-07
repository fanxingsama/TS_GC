import os
import joblib
from matplotlib import rcParams
from datetime import datetime
import torch
import torch.nn as nn
from pathlib import Path
import gc
from copy import deepcopy
from config import *
import matplotlib.pyplot as plt

from util.logger import get_logger, setup_logging
from model.TS_GC import TS_GC

rcParams['font.family'] = 'SimHei'
rcParams['axes.unicode_minus'] = False

# 软阈值算子优化，由于结构化正则化（如 $L_1$ 范数）在 0 点处不可导，传统的 Adam 或 SGD 优化器无法产生精确的 0 值，所以这里要用ISTA，使用软阈值算子 (Soft Thresholding) 把小的权重直接“归零”，用来优化非光滑问题
def PGD_update(network, lam, lr, penalty):
    hidden, p, lag = network.shape
    # torch.norm(network, dim=(0, 2))：计算特定输入变量对目标变量影响的整体强度
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
    
# 结构化稀疏正则化，强迫模型在学习时进行“优胜劣汰”
def lasso_penalty(network, lam, penalty):
    hidden, p, lag = network.shape
    # torch.norm(network, dim=(0, 2))：计算特定输入变量对目标变量影响的整体强度
    if penalty == 'GL': # 组Loss惩罚
        return lam * torch.sum(torch.norm(network, dim=(0, 2)))
    elif penalty == 'GSGL': # 组稀疏组Lasso惩罚，在组的基础上增加更细粒度的控制，既希望筛选出哪些变量有因果关系，又希望在有关系的变量中剔除不显著的时间点
        return lam * (torch.sum(torch.norm(network, dim=(0, 2)))
                      + torch.sum(torch.norm(network, dim=0)))
    elif penalty == 'H': # 层次Lasso惩罚，针对时序特征最强的约束。它要求如果滞后 1 秒没有关系，那么滞后 2 秒也应该没关系，保证了时间上的连续性和物理一致性
        return lam * sum([torch.sum(torch.norm(network[:, :, :(i+1)], dim=(0, 2)))
                          for i in range(lag)])
    else:
        raise ValueError('unsupported penalty: %s' % penalty)

class TS_GC_Trainer:
    def __init__(self, model, epochs, save_dir, criterion, lr, device, series_num,
                 X_full, Y_full, penalty_type, lasso_param, ridge_param, 
                 check_every=10, lookback=4, verbose=1, apply_mask=False):
        self.model = model
        self.epochs = epochs
        self.save_dir = save_dir
        self.criterion = criterion
        self.base_lr = lr
        self.device = device
        self.check_every = check_every
        self.apply_mask = apply_mask
        
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
        
        self.model_core = self.model
        self.first_layer_params_list = self.model_core.get_first_layer_weights() 
        
    def train(self, save_model=False):
        warmup_epochs = 100  # 定义预热轮次
        for ista_iter in range(self.epochs):
            current_smooth_loss, total_mse_loss, ridge_loss_val = self._compute_smooth_loss(self.X_full, self.Y_full)

            # 2. 梯度计算
            self.model.zero_grad()
            current_smooth_loss.backward()

            # 手动梯度下降
            with torch.no_grad():
                for param in self.model.parameters():
                    if param.grad is not None:
                        param.data.sub_(self.base_lr * param.grad) # In-place subtraction

            # 4. Proximal update for the first layer weights
            if ista_iter >= warmup_epochs and self.lasso_param > 0:
                with torch.no_grad():
                    for weight_tensor in self.first_layer_params_list: # Iterate through the list of weight tensors
                        if weight_tensor is not None:
                             PGD_update(weight_tensor, self.lasso_param, self.base_lr, self.penalty_type)
            
            # 定期检查
            if (ista_iter + 1) % self.check_every == 0:
                with torch.no_grad():
                    eval_smooth_loss, total_mse_loss, ridge_loss_val = self._compute_smooth_loss(self.X_full, self.Y_full)
                    eval_lasso_loss = self._compute_lasso_loss()
                    mean_loss = (eval_smooth_loss + eval_lasso_loss) / self.series_num
                
                self.train_losses.append(mean_loss.item()) # Store as float
                
                if self.verbose > 0:
                    print(f"{'='*10} ISTA Iter = {ista_iter + 1} {'='*10}")
                    print(f"MSE Loss = {(total_mse_loss / self.series_num).item():.6f} - ridge_loss = {(ridge_loss_val / self.series_num).item():.6f} - Lasso Loss = {(eval_lasso_loss / self.series_num).item():.6f}")

                # 非平滑损失为 0 时，提前停止训练
                if eval_lasso_loss == 0:
                    print("Lasso损失为 0，提前停止训练。")
                    if self.best_model_state is not None:
                        self.model.load_state_dict(self.best_model_state)
                    return self.best_loss
            
                # 早停
                if mean_loss < self.best_loss:
                    self.best_loss = mean_loss
                    self.best_it = ista_iter + 1
                    self.best_model_state = deepcopy(self.model.state_dict())
                    if save_model:
                        print(f"最优轮数： {self.best_it} ----------最优loss： {self.best_loss:.6f}")
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

    def _evaluate_full(self):
        """
        辅助函数：全量计算 Loss (比 Loader 更快)
        """
        self.model.eval()
        
        with torch.no_grad():
            preds = self.model(self.X_full)
            
            # 1. MSE
            total_mse = 0
            for i in range(self.series_num):
                total_mse += self.criterion(preds[:, :, i], self.Y_full[:, :, i])
            avg_mse_loss = total_mse.item() # criterion 已经是 mean 了

            # 2. Ridge
            ridge_loss = 0.0
            if self.ridge_param > 0:
                for param in self.model.parameters():
                    ridge_loss += torch.sum(param ** 2).item()
            ridge_loss = self.ridge_param * ridge_loss

            # 3. Nonsmooth (Lasso)
            lasso_loss = self._compute_lasso_loss().item()
            
            # Smooth = MSE + Ridge
            total_smooth = avg_mse_loss + ridge_loss
            # Total = (Smooth + Nonsmooth) / SeriesNum (保持原本的量纲逻辑)
            mean_loss = (total_smooth + lasso_loss) / self.series_num
            
            return mean_loss, avg_mse_loss / self.series_num, ridge_loss / self.series_num, lasso_loss / self.series_num
    
    # 计算并应用软梯度掩码,针对每个子网络的第一层卷积 (first_conv) 的梯度进行缩放。
    def _apply_gradient_mask(self):
        # 1. 获取掩码 (返回的是一个字典 {target_idx: mask_tensor})
        #    建议参数: threshold_ratio=0.5 (保留峰值的一半), suppression_factor=0.1 (非显著区梯度降为10%)
        gradient_masks = self.model_core.get_soft_mask(threshold_ratio=0.5, suppression_factor=0.1)
        
        # 2. 遍历每个目标变量的子网络
        for i in range(self.series_num):
            network = self.model_core.networks[i]
            mask = gradient_masks[i] # 获取对应的掩码 [series_num, kernel_size]
            
            # 3. 确保掩码和权重在同一个设备
            if mask.device != network.first_conv.weight.device:
                mask = mask.to(network.first_conv.weight.device)
            
            # 4. 获取梯度
            grad = network.first_conv.weight.grad
            if grad is not None:
                # 5. 维度对齐与应用
                # grad shape: [out_channels, series_num, kernel_size]
                # mask shape: [series_num, kernel_size]
                # 我们需要把 mask 扩展为 [1, series_num, kernel_size] 以便广播
                mask_expanded = mask.unsqueeze(0)
                
                # 6. 直接修改梯度 (In-place multiplication)
                grad.data.mul_(mask_expanded)
    
    # 计算平滑损失
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
    def _compute_lasso_loss(self):
        total_lasso_loss = 0
        for weight_tensor in self.first_layer_params_list:
            if weight_tensor is not None:
                 total_lasso_loss += lasso_penalty(weight_tensor, self.lasso_param, self.penalty_type)
        return total_lasso_loss

    # 计算正则化损失
    def _ridge_regularize(self):
        ridge_loss = torch.tensor(0.0, device=self.device) # Initialize on correct device
        if self.ridge_param > 0:
            for param in self.model.parameters():
                if param.requires_grad: # Only regularize parameters that are learnable
                    ridge_loss += torch.sum(param ** 2)
        return self.ridge_param * ridge_loss
    
    def cleanup(self):
        self.best_model_state = None
        self.train_losses = [float(x) for x in self.train_losses] 
        del self.X_full
        del self.Y_full
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
    
    model = TS_GC(
        input_window=INPUT_WINDOW,
        output_window=OUTPUT_WINDOW,
        series_num=SERIES_NUM,
        feature_dim=FEATURE_DIM,
        temporal_layers=TEMPORAL_LAYERS,
        kernel_size=KERNAL_SIZE,
        dropout=DROUP_OUT,
        device=DEVICE,
        use_temporal=USE_TEMPORAL, # 引入消融参数
        use_spatial=USE_SPATIAL,   # 引入消融参数
        use_residual=USE_RESIDUAL  # 引入消融参数
    ).to(DEVICE)

    trainer = TS_GC_Trainer(
        model=model, 
        epochs=EPOCHS, 
        save_dir=save_dir, 
        criterion=LOSS_FUNCTION,
        lr=LR, 
        device=DEVICE,
        series_num=SERIES_NUM,
        X_full=X_DATA,
        Y_full=Y_DATA,
        # logger=train_logger,
        penalty_type=PENALTY_TYPE,
        lasso_param=LASSO_PARAM,
        ridge_param=RIDGE_PARAM,
        verbose=1,
        apply_mask = APPLY_MASK
    )
    
    trainer.train(save_model=True)
    trainer.plot_training_curves()
    trainer.cleanup()
    
    log_message = (
    f"本次所使用的模型参数和训练参数如下：\n"
    f"模型架构:\n"
    f"  - 模型: {model}\n"
    f"模型参数:\n"
    f"  - feature_dim: {FEATURE_DIM}\n"
    f"  - kernel_size: {KERNAL_SIZE}\n"
    f"  - input_window: {INPUT_WINDOW}\n"
    f"  - output_window: {OUTPUT_WINDOW}\n"
    f"  - dropout: {DROUP_OUT}\n"
    f"  - 是否用mask: {APPLY_MASK}\n"
    f"训练参数:\n"
    f"  - loss: {LOSS_FUNCTION}\n"
    f"  - 数据路径: {DATA_PATH}\n"
    f"  - lr: {LR}\n"
    f"  - ridge_param: {RIDGE_PARAM}\n"
    f"  - penalty: {PENALTY_TYPE}\n"
    f"  - Lasso 参数: {LASSO_PARAM}\n"
    f"  - 序列数量: {SERIES_NUM}\n"
    f"模型架构消融:\n"
    f"  - 是否使用时间层 (USE_TEMPORAL): {USE_TEMPORAL}\n"
    f"  - 是否使用空间层 (USE_SPATIAL): {USE_SPATIAL}\n"
    f"  - 是否保留原始残差 (USE_RESIDUAL): {USE_RESIDUAL}\n"
)

    train_logger.info(log_message)
    
        # ========== 新增：训练完成后调用测试和根因分析 ==========
    print("\n" + "="*50)
    print("训练完成，开始执行测试...")
    print("="*50)
    # 导入测试模块
    from test_TS_GC import main as test_main
    test_main(save_dir)  # 执行测试
    
    # print("\n" + "="*50)
    # print("测试完成，开始根因分析...")
    # print("="*50)
    # # 导入根因分析模块
    # from max_tree import analyze_root_cause_and_save
    # analyze_root_cause_and_save(save_dir)  # 执行根因分析
    
if __name__ == '__main__':
    main()