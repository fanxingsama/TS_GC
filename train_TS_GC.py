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
import pandas as pd
import matplotlib.pyplot as plt

# 假设 MutiTS_GC 在名为 model_TS_GC.py 的文件中，或者在 Canvas 中提供的 TS_GC.py 文件中
# from model.TS_GC import MutiTS_GC # 根据您的项目结构调整
# 如果 TS_GC.py 与此脚本在同一目录或Python路径中:
from model.TS_GC import MutiTS_GC
from util import simulate_var


rcParams['font.family'] = 'SimHei'
rcParams['axes.unicode_minus'] = False

def PGD_update(weight_tensor, lam, lr, penalty):
    # weight_tensor 的形状: [out_channels, in_channels/groups, kernel_size]
    # 对应到 cmlp.py 中的 W (hidden, p, lag)
    # out_channels -> hidden (feature_dim)
    # in_channels/groups -> p (series_num)
    # kernel_size -> lag (kernel_size for first_conv)
    feature_dim, series_num_in_group, kernel_s = weight_tensor.shape

    if penalty == 'GL':
        norm = torch.norm(weight_tensor, dim=(0, 2), keepdim=True) # Group Lasso over feature_dim and kernel_size
        weight_tensor.data = ((weight_tensor / torch.clamp(norm, min=(lr * lam)))
                              * torch.clamp(norm - (lr * lam), min=0.0))
    elif penalty == 'GSGL':
        # GSGL typically has two norms.
        # Norm over series_num_in_group (dim=1 for Conv1D weights if groups=1)
        norm1 = torch.norm(weight_tensor, dim=1, keepdim=True) # Group over input series
        weight_tensor.data = ((weight_tensor / torch.clamp(norm1, min=(lr * lam)))
                              * torch.clamp(norm1 - (lr * lam), min=0.0))
        # Norm over feature_dim and kernel_size (dim=(0,2))
        norm2 = torch.norm(weight_tensor, dim=(0, 2), keepdim=True) # Group over output features and kernel taps
        weight_tensor.data = ((weight_tensor / torch.clamp(norm2, min=(lr * lam)))
                              * torch.clamp(norm2 - (lr * lam), min=0.0))
    elif penalty == 'H': # Hierarchical over kernel_size
        for i in range(kernel_s):
            # Norm over feature_dim and first (i+1) kernel taps, and all input series
            norm = torch.norm(weight_tensor[:, :, :(i + 1)], dim=(0, 1, 2), keepdim=True)
            # Corrected: norm should be taken across relevant dimensions for hierarchical sparsity on kernel taps
            # A common way for hierarchical on kernel is to sum norms of weights up to a certain lag
            # For Conv1D weights [out_channels, in_channels, kernel_size]
            # Hierarchical on kernel_size means penalizing groups of [out_channels, in_channels, 0:k]
            norm_hier = torch.norm(weight_tensor[:, :, :(i + 1)].flatten(start_dim=0, end_dim=1), p=2, dim=0, keepdim=True) # Norm over [out_channels, in_channels] for each tap group
            
            # The original cmlp.py PGD for H was:
            # norm = torch.norm(W[:, :, :(i + 1)], dim=(0, 2), keepdim=True)
            # W.data[:, :, :(i+1)] = ( (W.data[:, :, :(i+1)] / torch.clamp(norm, min=(lr * lam))) * torch.clamp(norm - (lr * lam), min=0.0))
            # Here, W has shape (hidden, p, lag).
            # For our weight_tensor (feature_dim, series_num, kernel_s):
            # dim 0 is feature_dim (hidden), dim 1 is series_num (p), dim 2 is kernel_s (lag)
            # So, dim=(0,1) for norm on W[:, :, :(i+1)] if we want to group over feature_dim and series_num for each lag group
            
            current_slice = weight_tensor[:, :, :(i + 1)]
            # Norm for hierarchical group lasso on kernel taps: for each (feature_dim, series_num) pair, take norm over kernel taps up to i
            # This interpretation might be different from cmlp.py if its 'p' is not series_num for the first layer.
            # Let's stick to cmlp.py's interpretation of W's dims for H penalty:
            # W.shape = (hidden, p, lag) -> (feature_dim, series_num, kernel_size)
            # norm over dim (0,2) means norm over (feature_dim, kernel_size_slice) for each series_num 'p'
            # This seems unusual for typical Conv1D weight interpretation.
            # Let's assume the PGD_update from cmlp.py is intended to be applied as is,
            # mapping W's dims to weight_tensor's dims.
            # W (hidden, p, lag) -> weight_tensor (out_channels, in_channels, kernel_size)
            # So, dim=(0,2) in cmlp.py corresponds to dim=(0,2) for weight_tensor.
            norm = torch.norm(current_slice, dim=(0, 2), keepdim=True)
            slice_data = current_slice.data
            slice_data = ((slice_data / torch.clamp(norm, min=(lr*lam))) * torch.clamp(norm - (lr*lam), min=0.0))
            weight_tensor.data[:, :, :(i+1)] = slice_data

    else:
        raise ValueError('unsupported penalty: %s' % penalty)

def lasso_penalty(weight_tensor, lam, penalty):
    feature_dim, series_num_in_group, kernel_s = weight_tensor.shape
    if penalty == 'GL':
        return lam * torch.sum(torch.norm(weight_tensor, dim=(0, 2)))
    elif penalty == 'GSGL':
        return lam * (torch.sum(torch.norm(weight_tensor, dim=(0, 2)))
                      + torch.sum(torch.norm(weight_tensor, dim=1))) # sum of norms over input series
    elif penalty == 'H':
        # Original cmlp.py: sum([torch.sum(torch.norm(W[:, :, :(i+1)], dim=(0, 2))) for i in range(lag)])
        return lam * sum([torch.sum(torch.norm(weight_tensor[:, :, :(i+1)], dim=(0, 2)))
                          for i in range(kernel_s)])
    else:
        raise ValueError('unsupported penalty: %s' % penalty)


def create_sequences(data_df, input_window, output_window, series_cols):
    X_list, Y_list = [], []
    data_np = data_df[series_cols].values
    num_timesteps, num_series = data_np.shape

    for i in range(num_timesteps - input_window - output_window + 1):
        X_list.append(data_np[i : i + input_window, :])
        Y_list.append(data_np[i + input_window : i + input_window + output_window, :])

    X = torch.tensor(np.array(X_list), dtype=torch.float32)
    Y = torch.tensor(np.array(Y_list), dtype=torch.float32)
    # Expected X: [num_samples, input_window, series_num]
    # Expected Y: [num_samples, output_window, series_num]
    return X, Y

class TS_GC_Trainer:
    def __init__(self, model, epochs, save_dir, criterion, lr, device, series_num,
                 X_full, Y_full, penalty_type, lasso_param, ridge_param, 
                 check_every, lookback=5, logger=None, verbose=1):
        self.model = model
        self.epochs = epochs # These are ISTA iterations
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

        other_params_names = set()
        for i in range(self.series_num):
            other_params_names.add(f"networks.{i}.first_conv.weight")

        self.other_params_for_ridge = []
        for name, param in self.model.named_parameters():
            is_first_layer_weight = False
            for fl_param in self.first_layer_params_list:
                if param is fl_param: # Check for object identity
                    is_first_layer_weight = True
                    break
            if not is_first_layer_weight:
                 self.other_params_for_ridge.append(param)


    def train(self, save_model=False):
        for ista_iter in range(self.epochs):
            current_smooth_loss, total_mse_loss, ridge_loss_val = self._compute_smooth_loss(self.X_full, self.Y_full)

            # 2. Backward pass for the smooth part
            self.model.zero_grad()
            current_smooth_loss.backward()

            with torch.no_grad():
                for param in self.model.parameters():
                    if param.grad is not None:
                        param.data.sub_(self.base_lr * param.grad) # In-place subtraction

            # 4. Proximal update for the first layer weights
            if self.lasso_param > 0:
                with torch.no_grad():
                    for weight_tensor in self.first_layer_params_list: # Iterate through the list of weight tensors
                        if weight_tensor is not None: # Should always be not None
                             PGD_update(weight_tensor, self.lasso_param, self.base_lr, self.penalty_type)
            
            # 5. Monitoring and Early Stopping (operates on full dataset metrics)
            if (ista_iter + 1) % self.check_every == 0:
                with torch.no_grad(): # Ensure no gradients are computed for monitoring
                    # Re-calculate smooth loss with updated parameters
                    eval_smooth_loss, total_mse_loss, ridge_loss_val = self._compute_smooth_loss(self.X_full, self.Y_full)
                    eval_nonsmooth_loss = self._compute_nonsmooth_loss()
                    mean_loss = (eval_smooth_loss + eval_nonsmooth_loss) / self.series_num
                
                self.train_losses.append(mean_loss.item()) # Store as float
                
                if self.verbose > 0:
                    print(f"{'='*10} ISTA Iter = {ista_iter + 1} {'='*10}")
                    print(f"Smooth Loss = {mean_loss.item():.6f}-------MSE Loss = {total_mse_loss.item():.6f} - Ridge Loss = {ridge_loss_val.item():.6f}")

                if mean_loss < self.best_loss:
                    self.best_loss = mean_loss
                    self.best_it = ista_iter + 1
                    self.best_model_state = deepcopy(self.model.state_dict())
                    if save_model:
                        os.makedirs(self.save_dir, exist_ok=True)
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
                        if self.verbose > 0:
                            print("Restored best model state for ISTA.")
                    return self.train_losses
        
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        return self.train_losses

    def _compute_smooth_loss(self, x_data, y_data):
        predictions = self.model(x_data) # Shape: [num_samples, output_window, series_num]
        
        total_mse_loss = 0
        for i in range(self.series_num):
            pred_for_series_i = predictions[:, :, i] # Shape: [num_samples, output_window]
            target_for_series_i = y_data[:, :, i]    # Shape: [num_samples, output_window]
            mse_i = self.criterion(pred_for_series_i, target_for_series_i)
            total_mse_loss += mse_i
        
        ridge_loss_val = self._ridge_regularize()
        return total_mse_loss + ridge_loss_val, total_mse_loss, ridge_loss_val

    def _compute_nonsmooth_loss(self):
        total_lasso_loss = 0
        # self.first_layer_params_list contains the actual weight tensors
        for weight_tensor in self.first_layer_params_list:
            if weight_tensor is not None:
                 total_lasso_loss += lasso_penalty(weight_tensor, self.lasso_param, self.penalty_type)
        return total_lasso_loss

    def _ridge_regularize(self):
        ridge_loss = torch.tensor(0.0, device=self.device) # Initialize on correct device
        if self.ridge_param > 0:
            for param in self.other_params_for_ridge:
                if param.requires_grad: # Only regularize parameters that are learnable
                    ridge_loss += torch.sum(param ** 2)
        return self.ridge_param * ridge_loss

    def cleanup(self):
        self.best_model_state = None
        self.train_losses = [float(x) for x in self.train_losses] # Ensure they are floats
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
  
    def plot_training_curves(self):
        plots_dir = os.path.join(self.save_dir, "training_curves_ista")
        os.makedirs(plots_dir, exist_ok=True)
        
        plt.figure(figsize=(10, 6))
        # X-axis for plot should be based on check_every
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
    DATA_PATH = 'data/simu_data/series_data.csv' # 数据路径
    INPUT_WINDOW = 10   # 例如: 使用过去50个时间点
    OUTPUT_WINDOW = 1   # 例如: 预测未来1个时间点
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    EPOCHS = 20000       # ISTA iterations
    LR = 0.01         # 学习率 for ISTA
    LASSO_PARAM = 0.001  # Lasso (L1) 正则化参数
    RIDGE_PARAM = 0.01 # Ridge (L2) 正则化参数 for other layers
    PENALTY_TYPE = 'GL' # 'GL', 'GSGL', or 'H'
    CHECK_EVERY = 10    # 每多少次ISTA迭代记录一次损失
    LOOKBACK = 4        # 早停参数: (lookback * check_every) iterations without improvement
    
    FEATURE_DIM = 64    # MutiTS_GC 参数
    KERNEL_SIZE = 3     # MutiTS_GC 参数
    DROPOUT = 0.1       # MutiTS_GC 参数
    TEMPORAL_LAYERS = 2 # MutiTS_GC 参数

    run_id = datetime.now().strftime(r'%m-%d_%H-%M-%S')
    save_dir = Path('saved') / run_id

    # 1. Load data
    data_df = pd.read_csv(DATA_PATH)
    SERIES_NUM = data_df.shape[1] # Infer from data
    series_columns = data_df.columns.tolist()

    X_full, Y_full = create_sequences(data_df, INPUT_WINDOW, OUTPUT_WINDOW, series_columns)
   
    model = MutiTS_GC(
        input_window=INPUT_WINDOW,
        output_window=OUTPUT_WINDOW,
        series_num=SERIES_NUM,
        feature_dim=FEATURE_DIM,
        temporal_layers=TEMPORAL_LAYERS,
        kernel_size=KERNEL_SIZE,
        dropout=DROPOUT,
        device=DEVICE
    ).to(DEVICE)
    
    # 4. Initialize loss function
    loss_function = nn.MSELoss(reduction='mean') # Or your preferred loss

    # 5. Initialize trainer
    trainer = TS_GC_Trainer(
        model=model, 
        epochs=EPOCHS, 
        save_dir=save_dir, 
        criterion=loss_function,
        lr=LR, 
        device=DEVICE,
        series_num=SERIES_NUM,
        X_full=X_full,
        Y_full=Y_full,
        penalty_type=PENALTY_TYPE, 
        lasso_param=LASSO_PARAM,
        ridge_param=RIDGE_PARAM,
        check_every=CHECK_EVERY,
        lookback=LOOKBACK,
        verbose=1
    )
    
    # 6. Train model
    print("Starting ISTA training...")
    train_losses = trainer.train(save_model=True)
    
    if train_losses: # Check if training actually ran
        trainer.plot_training_curves()
        print(f"Training completed. Best loss: {trainer.best_loss:.6f} at iteration {trainer.best_it}")
    else:
        print("Training did not produce any loss records.")

    trainer.cleanup()
    
if __name__ == '__main__':
    main()
