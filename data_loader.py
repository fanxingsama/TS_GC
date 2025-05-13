import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
import torch
from sklearn import preprocessing



class TimeSeriesDataloader():
    '''
    input_window: 输入序列长度，每次预测的时候使用多少个数据点
    output_window: 输出序列长度，预测多少个数据点
    '''
    def __init__(self, data_dir=None, gc_dir=None, batch_size=64, DATA_SEED=42, 
                 input_window=None, output_window=1):
        self.DATA_SEED = DATA_SEED
        self.batch_size = batch_size
        self.input_window = input_window
        self.output_window = output_window
        
        self.df_a = pd.read_csv(data_dir) # 形状: [series_len, series_num]
        self.df_b = pd.read_csv(gc_dir, header=None) # 读取真实的格兰杰因果矩阵
        self.series_num = self.df_a.shape[1] # 获取序列数量
        
        X_np = self.df_a.values  # 获取所有的数据点
        scaler = preprocessing.MinMaxScaler(feature_range=(0, 1)) # 归一化
        X_np = scaler.fit_transform(X_np)

        # T：时间序列数量，P：时间序列长度
        self.X_np = X_np[:, :, np.newaxis] # 给序列增加一个新的维度，最后一个维度是序列的特征数量，X_np表示每一个序列的形状: [series_len, series_num, 1]
        
    #  划分数据集
    def split_sampler(self):
        # 分割数据
        X_train_val_np, X_test_np = train_test_split(self.X_np, test_size=0.2, random_state=self.DATA_SEED, shuffle=False)
        X_train_np, X_val_np = train_test_split(X_train_val_np, test_size=0.25, random_state=self.DATA_SEED, shuffle=False)
        
        # --- 构造模型可接收的输入 ---
        X_train_seq, y_train_seq = create_sequences(X_train_np, self.input_window, self.output_window) # [num_samples, input_window, series_num, feature_num]
        X_val_seq, y_val_seq = create_sequences(X_val_np, self.input_window, self.output_window)
        X_test_seq, y_test_seq = create_sequences(X_test_np, self.input_window, self.output_window)

        # 转换为 PyTorch 张量
        X_train_tensor = torch.tensor(X_train_seq, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train_seq, dtype=torch.float32)
        X_val_tensor = torch.tensor(X_val_seq, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val_seq, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test_seq, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test_seq, dtype=torch.float32)
        
        # 创建数据集加载器
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        
        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True) # 为了确保 PGD 状态的一致性，丢弃最后一个批次（如果处理得当可以移除）
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        
        # 返回训练集和验证集的采样器
        return train_loader, val_loader, test_loader # [batch_size, num_samples, input_window, series_num, feature_num]
        
# --- 辅助函数：创建序列 (适配 CausalFormer 输入输出) ---
def create_sequences(data, input_window, output_window): # data:  [series_len, series_num, feature_num]。
    xs, ys = [], []
    total_len = len(data)
    
    # 确保有足够的数据创建序列
    if total_len < input_window + output_window:
        raise ValueError(f"数据长度 {total_len} 小于输入窗口 {input_window} 加输出窗口 {output_window} 的总和")
    
    # 滑动窗口创建序列
    for i in range(total_len - input_window - output_window + 1):
        x_window = data[i:i+input_window] # [input_window, series_num, feature_num]
        y_window = data[i+input_window:i+input_window+output_window]

        xs.append(x_window)
        ys.append(y_window)
    # xs: [num_samples, input_window, series_num, feature_num]，其中num_samples是分成的样本数量
    # ys: [num_samples, output_window, series_num, feature_num]
    return np.array(xs), np.array(ys)