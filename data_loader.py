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
        
        self.X_np = self.df_a.values  # 获取所有的数据点
        
    #  划分数据集
    def split_sampler(self):
        # 首先进行数据划分（不进行归一化）
        X_train_val_np, X_test_np = train_test_split(self.X_np, test_size=0.2, random_state=self.DATA_SEED, shuffle=False)
        X_train_np, X_val_np = train_test_split(X_train_val_np, test_size=0.25, random_state=self.DATA_SEED, shuffle=False)
        
        # 初始化每个时间序列的缩放器列表
        scalers = []
        
        # 创建转换后的数组
        X_train_scaled = np.zeros_like(X_train_np)
        X_val_scaled = np.zeros_like(X_val_np)
        X_test_scaled = np.zeros_like(X_test_np)
        
        # 对每个时间序列单独进行归一化
        for i in range(self.series_num):
            scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
            # 只使用训练集拟合缩放器
            scaler.fit(X_train_np[:, i].reshape(-1, 1))
            scalers.append(scaler)
            
            # 应用相同的缩放器到所有数据集的对应列
            X_train_scaled[:, i] = scaler.transform(X_train_np[:, i].reshape(-1, 1)).flatten()
            X_val_scaled[:, i] = scaler.transform(X_val_np[:, i].reshape(-1, 1)).flatten()
            X_test_scaled[:, i] = scaler.transform(X_test_np[:, i].reshape(-1, 1)).flatten()
        
        # 存储缩放器，以便后续使用（如反归一化）
        self.scalers = scalers
        
        # 添加特征维度
        X_train_scaled = X_train_scaled[:, :, np.newaxis]  # 形状: [series_len, series_num, 1]
        X_val_scaled = X_val_scaled[:, :, np.newaxis]
        X_test_scaled = X_test_scaled[:, :, np.newaxis]
        
        # --- 构造模型可接收的输入 ---
        X_train_seq, y_train_seq = create_sequences(X_train_scaled, self.input_window, self.output_window)
        X_val_seq, y_val_seq = create_sequences(X_val_scaled, self.input_window, self.output_window)
        X_test_seq, y_test_seq = create_sequences(X_test_scaled, self.input_window, self.output_window)

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
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        
        # 返回训练集和验证集的采样器
        return train_loader, val_loader, test_loader
    
    def inverse_transform(self, data, series_indices=None):
        """
        将归一化的数据转换回原始尺度
        
        参数:
        - data: 形状为 [batch_size, sequence_len, series_num, 1] 的数据
        - series_indices: 指定要转换的时间序列索引列表，默认转换所有序列
        
        返回:
        - 逆变换后的数据
        """
        if series_indices is None:
            series_indices = range(self.series_num)
            
        # 确保scalers已经初始化
        if not hasattr(self, 'scalers'):
            raise ValueError("请先调用split_sampler方法初始化scalers")
            
        data_copy = data.copy()
        
        # 处理不同维度的情况
        if data.ndim == 4:  # [batch_size, sequence_len, series_num, 1]
            for i, idx in enumerate(series_indices):
                scaler = self.scalers[idx]
                # 处理每个批次和每个时间步
                for b in range(data.shape[0]):
                    for t in range(data.shape[1]):
                        data_copy[b, t, idx, 0] = scaler.inverse_transform(
                            data[b, t, idx, 0].reshape(-1, 1)
                        ).flatten()[0]
        elif data.ndim == 3:  # [sequence_len, series_num, 1]
            for i, idx in enumerate(series_indices):
                scaler = self.scalers[idx]
                for t in range(data.shape[0]):
                    data_copy[t, idx, 0] = scaler.inverse_transform(
                        data[t, idx, 0].reshape(-1, 1)
                    ).flatten()[0]
        else:
            raise ValueError(f"不支持的数据维度: {data.ndim}")
            
        return data_copy
      
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