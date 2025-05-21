import os
from matplotlib import rcParams
import torch
import torch.optim as optim
from model.granger_tcn import (
    lasso_penalty,
    PGD_update
)
import matplotlib.pyplot as plt

rcParams['font.family'] = 'SimHei'
rcParams['axes.unicode_minus'] = False

class CausalFormerTrainer2:
    def __init__(self, model, epoch, save_dir, criterion, lr, device, series_num,
                train_loader, valid_loader, penalty_type, 
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
        self.lambda_reg = 0.02  # 非平滑正则化参数
        self.lambda_ridge = lambda_ridge  # 输出层的岭正则化参数
        self.series_num = series_num
        self.early_stop_patience = 2  # 修改为2轮没有改善就早停
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
        
        # 早停相关变量
        self.best_val_mse = float('inf')  # 最佳验证集MSE
        self.patience_counter = 0  # 耐心计数器
        self.early_stopped = False  # 是否早停标志
        self.best_epoch = 0  # 最佳epoch
        self.best_model_state = None  # 保存最佳模型状态
        
        
        # 为每个TCN网络创建独立的参数组和学习率
        self.tcn_param_groups_first_layer = [[] for _ in range(series_num)] # 每个TCN的第一层W权重
        self.tcn_param_groups_other_layers = [[] for _ in range(series_num)] # 每个TCN除了第一层W权重的其他权重
        self.tcn_optimizers = []
        self.tcn_lrs = [self.base_lr for _ in range(series_num)]
        self.converged_networks = [False] * series_num
        
        # 非TCN参数
        self.non_tcn_params = []
        _all_tcn_params_identity_set = set()
                        
         # 填充TCN的参数列表
        for i in range(self.series_num):
            first_layer_conv_pattern = f"encoder.layers.0.attention.tcn_processors.{i}.network_layers.0.conv1.weight"
            for name, param in model.named_parameters():
                if f"encoder.layers.0.attention.tcn_processors.{i}" in name: # 参数属于当前的TCN处理器
                    _all_tcn_params_identity_set.add(param) # 记录所有TCN参数
                    if name.startswith(first_layer_conv_pattern):
                        self.tcn_param_groups_first_layer[i].append(param) # 第一层的参数保存
                    else:
                        self.tcn_param_groups_other_layers[i].append(param) # TCN其它层参数保存

        # 第二遍：填充non_tcn_params
        for param in model.parameters(): # model.parameters() 返回唯一的参数对象
            if param not in _all_tcn_params_identity_set: # 通过对象ID检查参数是否已在TCN参数集中
                self.non_tcn_params.append(param)
        
        # 为每个TCN创建独立的优化器
        for i in range(series_num):
            self.tcn_optimizers.append(optim.Adam(self.tcn_param_groups_other_layers[i], lr=self.tcn_lrs[i], weight_decay=0))
        
        # 为非TCN参数创建优化器
        self.non_tcn_optimizer = optim.Adam(self.non_tcn_params, lr=self.base_lr, weight_decay=0)
        
        # 保存模型状态字典，在需要时重新加载
        self.model_state_dict = {}
        
        # 如果不使用单调线搜索，则为每个网络保存历史损失
        if not self.monotone:
            self.last_losses = [[] for _ in range(series_num)]
    
    def train(self):
        line_search = self.begin_line_search  # 是否使用线搜索
        
        for epoch in range(self.epochs):
            epoch_loss, epoch_mse, epoch_ridge, epoch_penalty = self.train_epoch(line_search)
            val_loss, val_mse, val_ridge, val_penalty = self.validate()
            
            # 保存训练和验证的损失
            self.train_losses.append(float(epoch_loss))
            self.train_mses.append(float(epoch_mse))
            self.train_ridges.append(float(epoch_ridge))
            self.train_penalties.append(float(epoch_penalty))
            
            self.val_losses.append(float(val_loss))
            self.val_mses.append(float(val_mse))
            self.val_ridges.append(float(val_ridge))
            self.val_penalties.append(float(val_penalty))
            
            if self.verbose > 0:
                print(f"Epoch {epoch+1}/{self.epochs}")
                print(f"Train - loss: {epoch_loss:.6f}, MSE: {epoch_mse:.6f}, "
                      f"Ridge: {epoch_ridge:.6f}, Penalty: {epoch_penalty:.6f}")
                print(f"Valid - loss: {val_loss:.6f}, MSE: {val_mse:.6f}, "
                      f"Ridge: {val_ridge:.6f}, Penalty: {val_penalty:.6f}")
            
            # 如果当前验证损失比之前的最佳损失更好，则保存模型
            if val_loss < self.best_loss_result:
                self.best_loss_result = val_loss
            
            # 早停检查：基于验证集MSE
            if val_mse < self.best_val_mse:
                self.best_val_mse = val_mse
                self.patience_counter = 0
                self.best_epoch = epoch
                # 保存最佳模型状态
                self.best_model_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
                if self.verbose > 0:
                    print(f"  新的最佳验证MSE: {val_mse:.6f}")
            else:
                self.patience_counter += 1
                if self.verbose > 0:
                    print(f"  验证MSE未改善 ({self.patience_counter}/{self.early_stop_patience})")
                
                # 如果耐心用完，触发早停
                if self.patience_counter >= self.early_stop_patience:
                    if self.verbose > 0:
                        print(f"  早停触发！验证MSE连续{self.early_stop_patience}轮未改善")
                        print(f"  最佳epoch: {self.best_epoch+1}, 最佳验证MSE: {self.best_val_mse:.6f}")
                    self.early_stopped = True
                    # 恢复最佳模型状态
                    if self.best_model_state is not None:
                        self.model.load_state_dict(self.best_model_state)
                        if self.verbose > 0:
                            print("  已恢复最佳模型状态")
                    break
                    
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
                        
            # 清理内存
            if hasattr(self, 'model_state_dict') and self.model_state_dict:
                self.model_state_dict = {}
                # Force garbage collection
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # 训练结束后的信息输出
        if self.verbose > 0:
            if self.early_stopped:
                print(f"\n训练因早停而结束于第{epoch+1}轮")
                print(f"最佳验证MSE: {self.best_val_mse:.6f} (第{self.best_epoch+1}轮)")
            else:
                print(f"\n训练正常结束于第{epoch+1}轮")
        
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
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device) # [batch_size, input_window, series_num, feature]
            
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
        # smooth_loss = mse_loss + ridge_loss
        smooth_loss = mse_loss 
        smooth_loss.backward(retain_graph=True)
        self.non_tcn_optimizer.step()
        nonsmooth_penalty = 0.0
        
        # 对每个TCN应用优化器更新和近端梯度下降（对第一层）
        for i in range(self.series_num):
            self.tcn_optimizers[i].step()
            # 再应用近端梯度下降到第一层
            with torch.no_grad():
                PGD_update(self.tcn_param_groups_first_layer[i][0], self.lambda_reg, self.tcn_lrs[i], self.penalty_type)
            # 计算非平滑正则化值
            nonsmooth_penalty += lasso_penalty(self.tcn_param_groups_first_layer[i][0], self.lambda_reg, self.penalty_type)
        
        # 计算平均非平滑正则化值
        nonsmooth_penalty /= self.series_num
        
        # 重新计算当前模型的总损失
        with torch.no_grad():
            total_loss = smooth_loss + nonsmooth_penalty
        
        return total_loss, mse_loss, ridge_loss, nonsmooth_penalty
    
    def train_batch_line_search(self, batch_x, batch_y):
        # 计算初始预测和损失
        predictions = self.model(batch_x)
        mse_loss = 0
        for i in range(self.series_num):
            mse_loss += self.criterion(predictions[:, :, i:i+1, :], batch_y[:, :, i:i+1, :])
        mse_loss /= self.series_num
        smooth_loss = mse_loss
        
        # 计算非平滑正则化值
        nonsmooth_penalty = 0.0
        for i in range(self.series_num):
            nonsmooth_penalty += lasso_penalty(self.tcn_param_groups_first_layer[i][0], self.lambda_reg, self.penalty_type)
        nonsmooth_penalty /= self.series_num
        
        # 计算总损失
        current_loss = smooth_loss + nonsmooth_penalty
        
        # 保存当前模型状态字典
        self.model_state_dict = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
        
        # 对每个TCN进行线搜索
        for i in range(self.series_num):
            if self.converged_networks[i]:  # 如果网络已收敛，跳过
                continue
            
            # 准备线搜索
            step_taken = False
            lr_it = self.tcn_lrs[i]
            
            # 获取比较损失
            if not self.monotone:
                if len(self.last_losses[i]) == 0:
                        self.last_losses[i].append(float(current_loss.item()))
                comp_loss = max(self.last_losses[i])
            else:
                comp_loss = float(current_loss.item())  # Ensure we use float, not tensor
            
            # 开始线搜索
            while not step_taken:
                # 在每次线搜索迭代中，恢复原始模型状态
                if not step_taken and lr_it != self.tcn_lrs[i]:
                    self.model.load_state_dict(self.model_state_dict)
                    
                # 确保梯度为零
                self.non_tcn_optimizer.zero_grad()
                for opt in self.tcn_optimizers:
                    opt.zero_grad()
                    
                # 计算新的预测和损失 - 不重用之前的计算
                new_predictions = self.model(batch_x)
                new_mse_loss = 0
                for j in range(self.series_num):
                    new_mse_loss += self.criterion(new_predictions[:, :, j:j+1, :], batch_y[:, :, j:j+1, :])
                new_mse_loss /= self.series_num
                new_smooth_loss = new_mse_loss
                
                # 计算梯度
                new_smooth_loss.backward()
                
                # 更新非TCN参数
                self.non_tcn_optimizer.step()
                
                # 对TCN参数i应用近端梯度下降
                with torch.no_grad():
                    PGD_update(self.tcn_param_groups_first_layer[i][0], self.lambda_reg, lr_it, self.penalty_type)
                
                # 用更新后的模型计算新的损失
                with torch.no_grad():
                    predictions_updated = self.model(batch_x)
                    mse_loss_updated = 0
                    for j in range(self.series_num):
                        mse_loss_updated += self.criterion(predictions_updated[:, :, j:j+1, :], batch_y[:, :, j:j+1, :])
                    mse_loss_updated /= self.series_num
                    
                    smooth_loss_updated = mse_loss_updated
                    
                    nonsmooth_penalty_updated = 0.0
                    for j in range(self.series_num):
                        nonsmooth_penalty_updated += lasso_penalty(self.tcn_param_groups_first_layer[j][0], self.lambda_reg, self.penalty_type)
                    nonsmooth_penalty_updated /= self.series_num

                    new_loss = smooth_loss_updated + nonsmooth_penalty_updated
                    
                    # Calculate parameter difference norm - HANDLE DEVICE CORRECTLY
                    diff_norm = 0.0
                    for name, param in self.model.named_parameters():
                        if f"encoder.layers.0.attention.tcn_processors.{i}" in name:
                            # Get original parameter - ENSURE SAME DEVICE
                            orig_param = self.model_state_dict[name]
                            # Ensure same device
                            if orig_param.device != param.device:
                                orig_param = orig_param.to(param.device)
                            diff_norm += torch.sum((param - orig_param) ** 2)
                    
                    # Calculate line search tolerance
                    tol = (0.5 * self.sigma / lr_it) * diff_norm
                    
                    # Accept update if loss decreases sufficiently
                    if comp_loss - new_loss.item() > tol:
                        step_taken = True
                        self.tcn_lrs[i] = (self.tcn_lrs[i] ** (1 - self.lr_decay)) * (lr_it ** self.lr_decay)
                        
                        # Update loss history (as scalar values)
                        if not self.monotone:
                            if len(self.last_losses[i]) == self.m:
                                self.last_losses[i].pop(0)
                            self.last_losses[i].append(float(new_loss.item()))
                    else:
                        lr_it *= self.r
                        if lr_it < self.lr_min:
                            # Learning rate too small, mark network as converged
                            self.converged_networks[i] = True
                            if self.verbose > 0:
                                print(f"  Network {i} 收敛 (lr too small)")
                            step_taken = True
                        else:
                            # Restore model if update not accepted
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
            batch_ridge = 0
            
            # 计算非平滑正则化损失
            batch_lasso_penalty = 0.0
            for i in range(self.series_num):
                batch_lasso_penalty += lasso_penalty(self.tcn_param_groups_first_layer[i][0], self.lambda_reg, self.penalty_type)
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
                    nonsmooth_penalty += lasso_penalty(self.tcn_param_groups_first_layer[i][0], self.lambda_reg, self.penalty_type)
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
    
    def cleanup(self):
        """Clean up resources to prevent memory leaks between trials"""
        self.model_state_dict = {}
        self.best_model_state = None  # 清理最佳模型状态
        
        self.train_losses = [float(x) for x in self.train_losses]
        self.train_mses = [float(x) for x in self.train_mses]
        self.train_ridges = [float(x) for x in self.train_ridges]
        self.train_penalties = [float(x) for x in self.train_penalties]
        self.val_losses = [float(x) for x in self.val_losses]
        self.val_mses = [float(x) for x in self.val_mses]
        self.val_ridges = [float(x) for x in self.val_ridges]
        self.val_penalties = [float(x) for x in self.val_penalties]
        
        # Force garbage collection
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
  
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
            'best_val_mse': self.best_val_mse,
            'early_stopped': self.early_stopped,
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
        # 如果有早停，标记最佳epoch
        if self.early_stopped and hasattr(self, 'best_epoch'):
            plt.axvline(x=self.best_epoch, color='red', linestyle='--', alpha=0.7, label=f'Best Epoch ({self.best_epoch+1})')
            plt.legend()
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
        # 如果有早停，标记最佳epoch
        if self.early_stopped and hasattr(self, 'best_epoch'):
            plt.axvline(x=self.best_epoch, color='red', linestyle='--', alpha=0.7, label=f'Best Epoch ({self.best_epoch+1})')
            plt.legend()
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