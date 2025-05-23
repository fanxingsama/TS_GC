import os
import sys
import joblib
from matplotlib import rcParams
from datetime import datetime
import torch.nn as nn
import torch
from pathlib import Path
import torch.optim as optim
import gc

from MutiTCN.only_tcn import MultiTCNModel
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

from model.granger_tcn import (
    lasso_penalty,
    PGD_update
)
import matplotlib.pyplot as plt
from config import DATA_PATH,BATCH_SIZE, INPUT_WINDOW, OUTPUT_WINDOW, FEATURE_DIM, OUTPUT_DIM, EPOCHS, DEVICE, timeseriesDataLoader, SERIES_NUM
from logger.logger import get_logger, setup_logging
rcParams['font.family'] = 'SimHei'
rcParams['axes.unicode_minus'] = False

class MultiTCNTrainer:
    """
    Multi-TCN模型训练器 - 分离第一层和其他层的优化
    
    Args:
        model: MultiTCNModel实例
        epochs (int): 训练轮数
        save_dir (str): 模型保存目录
        criterion: 损失函数
        lr (float): 学习率
        first_layer_lr (float): 第一层学习率
        device: 计算设备
        series_num (int): 时间序列数量
        train_loader: 训练数据加载器
        valid_loader: 验证数据加载器
        penalty_type (str): 正则化类型 ('GL', 'GSGL', 'H')
        lasso_param (float): Lasso正则化参数
        ridge_param (float): Ridge正则化参数
        logger: 日志记录器
        verbose (int): 详细程度
    """
    def __init__(self, model, epochs, save_dir, criterion, lr, device, series_num,
                 train_loader, valid_loader, penalty_type, lasso_param, logger=None,
                 ridge_param=0.01, verbose=1):
        self.model = model
        self.epochs = epochs
        self.save_dir = save_dir
        self.criterion = criterion
        self.logger = logger
        self.base_lr = lr
        self.device = device
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.penalty_type = penalty_type
        self.lasso_param = lasso_param
        self.ridge_param = ridge_param
        self.series_num = series_num
        self.early_stop_patience = 4
        self.best_mse_result = float('inf')
        self.verbose = verbose

        # 学习率调度器参数
        self.lr_decay_step = 15
        self.lr_decay_gamma = 0.7
        self.use_plateau_scheduler = False
        self.plateau_patience = 5
        self.plateau_factor = 0.5

        # 训练记录
        self.train_losses = []
        self.train_mses = []
        self.train_ridges = []
        self.train_penalties = []

        self.val_losses = []
        self.val_mses = []
        self.val_ridges = []
        self.val_penalties = []

        # 早停相关变量
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.early_stopped = False
        self.best_epoch = 0
        self.best_model_state = None

        # 分离第一层参数和其他参数
        self.first_layer_params = []
        self.other_params = []
        
        # 收集第一层参数
        first_layer_param_names = set()
        for i in range(self.series_num):
            first_layer_conv_pattern = f"tcn_processors.{i}.network_layers.0.conv1.weight"
            first_layer_param_names.add(first_layer_conv_pattern)
        
        # 分类参数
        for name, param in self.model.named_parameters():
            if name in first_layer_param_names:
                self.first_layer_params.append(param)
            else:
                self.other_params.append(param)

        # 只为非第一层参数创建Adam优化器
        self.optimizer = optim.Adam(self.other_params, lr=self.base_lr, weight_decay=0)

            
    def ridge_regularize(self):
        """对非第一层参数进行Ridge正则化"""
        ridge_loss = 0.0
        for param in self.other_params:
            if param is not None:
                ridge_loss += torch.sum(param ** 2)
        return self.ridge_param * ridge_loss
        

    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        epoch_loss = 0.0
        epoch_mse = 0.0
        epoch_ridge = 0.0
        epoch_penalty = 0.0
        num_batches = 0

        for batch_idx, (batch_x, batch_y) in enumerate(self.train_loader):
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            
            # 清零梯度
            if self.optimizer is not None:
                self.optimizer.zero_grad()
            
            # 手动清零第一层梯度
            for param in self.first_layer_params:
                if param.grad is not None:
                    param.grad.zero_()

            # 前向传播
            predictions = self.model(batch_x)
            mse_loss = self.criterion(predictions, batch_y)
            
            # 正则化计算
            ridge_loss = self.ridge_regularize()
            total_loss = mse_loss + ridge_loss
            total_loss.backward()

            self.optimizer.step()

            # 对第一层参数执行手动梯度下降
            for param in self.first_layer_params:
                if param.grad is not None:
                    param.data = param.data - self.base_lr * param.grad

            # 对第一层应用近端优化（稀疏正则化）
            nonsmooth_penalty = 0.0
            for param in self.first_layer_params:
                if param is not None:
                    # 应用近端梯度下降更新
                    PGD_update(param, self.lasso_param, self.base_lr, self.penalty_type)
                    # 计算稀疏惩罚
                    nonsmooth_penalty += lasso_penalty(param, self.lasso_param, self.penalty_type)
            
            if len(self.first_layer_params) > 0:
                nonsmooth_penalty /= len(self.first_layer_params)

            # 总损失（包含正则化）
            total_loss_with_penalty = total_loss + nonsmooth_penalty

            epoch_loss += total_loss_with_penalty.item()
            epoch_mse += mse_loss.item()
            epoch_ridge += ridge_loss.item()
            epoch_penalty += nonsmooth_penalty.item()
            num_batches += 1

        return (epoch_loss / num_batches, epoch_mse / num_batches,
                epoch_ridge / num_batches, epoch_penalty / num_batches)

    def validate(self):
        """验证一个epoch"""
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
                mse_loss = self.criterion(predictions, batch_y)
                ridge_loss = self.ridge_regularize()
                
                # 计算稀疏惩罚
                nonsmooth_penalty = 0.0
                for param in self.first_layer_params:
                    if param is not None:
                        nonsmooth_penalty += lasso_penalty(param, self.lasso_param, self.penalty_type)
                
                if len(self.first_layer_params) > 0:
                    nonsmooth_penalty /= len(self.first_layer_params)

                total_loss = mse_loss + ridge_loss + nonsmooth_penalty

                val_loss += total_loss.item()
                val_mse += mse_loss.item()
                val_ridge += ridge_loss.item()
                val_penalty += nonsmooth_penalty.item()
                num_batches += 1

        return (val_loss / num_batches, val_mse / num_batches,
                val_ridge / num_batches, val_penalty / num_batches)

    def train(self, save_model=False):
        """主训练循环"""
        for epoch in range(self.epochs):
            epoch_loss, epoch_mse, epoch_ridge, epoch_penalty = self.train_epoch()
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
                print(f"Train - Loss: {epoch_loss:.6f}, MSE: {epoch_mse:.6f}, "
                      f"Ridge: {epoch_ridge:.6f}, Penalty: {epoch_penalty:.6f}")
                print(f"Valid - Loss: {val_loss:.6f}, MSE: {val_mse:.6f}, "
                      f"Ridge: {val_ridge:.6f}, Penalty: {val_penalty:.6f}")
                if self.optimizer is not None:
                    current_lr = self.optimizer.param_groups[0]['lr']
                    print(f"Adam Learning Rate: {current_lr:.2e}")
            
            # 早停检查
            if val_mse < self.best_mse_result:
                self.best_mse_result = val_mse
                self.patience_counter = 0
                self.best_epoch = epoch
                # 保存最佳模型状态
                self.best_model_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
                
                # 保存模型
                if save_model:
                    os.makedirs(self.save_dir, exist_ok=True)
                    torch.save(self.best_model_state, os.path.join(self.save_dir, "best_model.pth"))
                    joblib.dump(self.model.config, os.path.join(self.save_dir, "model_config.pkl"))
                    
                if self.verbose > 0:
                    print(f"  New best validation MSE: {val_mse:.6f}")
            else:
                self.patience_counter += 1
                if self.verbose > 0:
                    print(f"  No improvement ({self.patience_counter}/{self.early_stop_patience})")
                
                # 早停
                if self.patience_counter >= self.early_stop_patience:
                    if self.verbose > 0:
                        print(f"  Early stopping triggered! No improvement for {self.early_stop_patience} epochs")
                        print(f"  Best epoch: {self.best_epoch+1}, Best validation MSE: {self.best_mse_result:.6f}")
                    self.early_stopped = True
                    # 恢复最佳模型状态
                    if self.best_model_state is not None:
                        self.model.load_state_dict(self.best_model_state)
                        if self.verbose > 0:
                            print("  Restored best model state")
                    break
        
        # 训练结束信息
        if self.verbose > 0:
            if self.early_stopped:
                print(f"\nTraining stopped early at epoch {epoch+1}")
                print(f"Best validation MSE: {self.best_mse_result:.6f} (epoch {self.best_epoch+1})")
            else:
                print(f"\nTraining completed at epoch {epoch+1}")
        
        return self.best_mse_result
    
    def cleanup(self):
        """清理资源以防止内存泄漏"""
        self.best_model_state = None  # 清理最佳模型状态
        
        # 确保损失列表是float类型
        self.train_losses = [float(x) for x in self.train_losses]
        self.train_mses = [float(x) for x in self.train_mses]
        self.train_ridges = [float(x) for x in self.train_ridges]
        self.train_penalties = [float(x) for x in self.train_penalties]
        self.val_losses = [float(x) for x in self.val_losses]
        self.val_mses = [float(x) for x in self.val_mses]
        self.val_ridges = [float(x) for x in self.val_ridges]
        self.val_penalties = [float(x) for x in self.val_penalties]
        
        # 强制垃圾回收
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
  
    def plot_training_curves(self):
        """绘制损失曲线"""
        # 创建保存图表的目录
        plots_dir = os.path.join(self.save_dir, "训练下降曲线")
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
        
        if self.verbose > 0:
            print(f"Loss curves saved to {plots_dir}")

def main():
    run_id = datetime.now().strftime(r'%m-%d_%H-%M-%S')
    save_dir = Path('saved') / run_id
    setup_logging(save_dir)
    train_logger = get_logger() # 日志记录器

    train_loader, val_loader, test_loader = timeseriesDataLoader.split_sampler() # 得到训练集、验证集和测试集的数据加载器

    dropout = 0
    tcn_channels = 256
    kernel_size = 4
    # 训练参数
    criterion = nn.L1Loss()
    lr = 0.0002 
    lasso_param = 10
    penalty_type = 'GL'
    
    log_message = (
    f"本次所使用的模型参数和训练参数如下：\n"
    f"模型参数:\n"
    f"  - tcn_channels: {tcn_channels}\n"
    f"  - kernel_size: {kernel_size}\n"
    f"  - dropout: {dropout}\n"
    f"训练参数:\n"
    f"  - EPOCHS: {EPOCHS}\n"
    f"  - BATCH_SIZE: {BATCH_SIZE}\n"
    f"  - DEVICE: {DEVICE}\n"
    f"  - 数据路径: {DATA_PATH}\n"
    f"  - 输入窗口长度: {INPUT_WINDOW}\n"
    f"  - 输出窗口长度: {OUTPUT_WINDOW}\n"
    f"  - 特征维度: {FEATURE_DIM}\n"
    f"  - 输出维度: {OUTPUT_DIM}\n"
    f"  - 学习率: {lr}\n"
    f"  - Lasso 参数: {lasso_param}\n"
    f"  - 正则化类型: {penalty_type}\n"
    f"  - 序列数量: {SERIES_NUM}\n"
)

    train_logger.info(log_message)
    
    model = MultiTCNModel(
        input_window=INPUT_WINDOW,
        output_window=OUTPUT_WINDOW,
        series_num=SERIES_NUM,
        feature_dim=FEATURE_DIM,
        output_dim=OUTPUT_DIM,
        device=DEVICE,
        tcn_channels=tcn_channels,
        kernel_size=kernel_size,
        dropout=dropout,
    ).to(DEVICE)
    
    causalFormerTrainer = MultiTCNTrainer(
        model=model, 
        epochs=EPOCHS, 
        save_dir=save_dir, 
        criterion=criterion,
        lr=lr, 
        device=DEVICE,
        train_loader=train_loader, 
        valid_loader=val_loader, 
        series_num=SERIES_NUM, 
        logger=train_logger,
        penalty_type=penalty_type, 
        lasso_param=lasso_param
    )
    causalFormerTrainer.train(save_model=True)
    causalFormerTrainer.plot_training_curves()

if __name__ == '__main__':
    main()