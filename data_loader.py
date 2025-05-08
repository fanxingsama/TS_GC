import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
import torch



class TimeSeriesDataloader():
    def __init__(self, data_dir=None, gc_dir=None, batch_size=64, DATA_SEED=42, 
                 input_window=None, output_window=None):
        self.DATA_SEED = DATA_SEED
        self.batch_size = batch_size
        self.input_window = input_window
        self.output_window = output_window
        
        df_a = pd.read_csv(data_dir)
        self.df_b = pd.read_csv(gc_dir, header=None) # 读取真实的格兰杰因果矩阵
        
        X_np = df_a.values  # 获取所有的数据点
        series_names = df_a.columns.tolist() # 获取序列名称
        self.series_num = len(series_names) # 获取序列数量
        
        self.series_to_idx = {name: i for i, name in enumerate(series_names)} # 创建从序列名称到索引的映射
        self.X_np = X_np[:, :, np.newaxis] # 给序列增加一个新的维度，最后一个维度是序列的特征数量，X_np表示每一个序列的形状: [T, P, 1]
        
    #  划分数据集
    def split_sampler(self):
        # 分割数据
        X_train_val_np, X_test_np = train_test_split(self.X_np, test_size=0.2, random_state=self.DATA_SEED, shuffle=False)
        X_train_np, X_val_np = train_test_split(X_train_val_np, test_size=0.25, random_state=self.DATA_SEED, shuffle=False)
        
        # --- 构造模型可接收的输入 ---
        X_train_seq, y_train_seq = create_sequences(X_train_np, self.input_window, self.output_window)
        X_val_seq, y_val_seq = create_sequences(X_val_np, self.input_window, self.output_window)
        X_test_sql, y_test_seq = create_sequences(X_test_np, self.input_window, self.output_window)

        # 转换为 PyTorch 张量
        X_train_tensor = torch.tensor(X_train_seq, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train_seq, dtype=torch.float32) # 形状: [N, T_out, P, F_out]
        X_val_tensor = torch.tensor(X_val_seq, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val_seq, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test_sql, dtype=torch.float32)
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
        return train_loader, val_loader, test_loader
        
    
    # 从文件中读取真实的格兰杰因果关系
    def get_true_granger(self):
        series_to_idx = self.series_to_idx
        GC_true_np = np.zeros((self.series_num, self.series_num), dtype=int) # 初始化真实的格兰杰因果矩阵
        # 读取真实的格兰杰因果矩阵
        for _, row in self.df_b.iterrows():
            cause_name = row.iloc[0]  # 第一列为 "因"
            effect_name = row.iloc[1] # 第二列为 "果"
            # lag_value = row.iloc[2] if df_b.shape[1] > 2 else None # 第三列为 "延迟" (可选)
            str_cause_name = str(cause_name)
            str_effect_name = str(effect_name)

            if str_cause_name in series_to_idx and str_effect_name in series_to_idx:
                idx_cause = series_to_idx[str_cause_name]
                idx_effect = series_to_idx[str_effect_name]
                GC_true_np[idx_effect, idx_cause] = 1 # 1 表示 idx_effect 受 idx_cause 的影响
        return GC_true_np # 返回真实的格兰杰因果矩阵
    
# --- 辅助函数：创建序列 (适配 CausalFormer 输入输出) ---
def create_sequences(data, input_seq_len, output_seq_len):
    """
    Args:
        data (np.array): 输入数据，形状 [时间步, 序列数量, 特征数量]。
        input_seq_len (int): 输入序列长度 (输入窗口)。
        output_seq_len (int): 输出序列长度 (输出窗口)。
    Returns:
        Tuple[np.array, np.array]: X (输入序列), Y (目标序列)
            X 形状: [样本数量, input_seq_len, 序列数量, 特征数量]
            Y 形状: [样本数量, output_seq_len, 序列数量, 特征数量]
    """
    xs, ys = [], []
    total_len = len(data)

    for i in range(total_len - input_seq_len - output_seq_len + 1):
        x = data[i:(i + input_seq_len)]  # 提取输入序列
        y = data[(i + input_seq_len):(i + input_seq_len + output_seq_len)]  # 提取目标序列
        xs.append(x)  # 将输入序列添加到列表 xs 中
        ys.append(y)  # 将目标序列添加到列表 ys 中
    return np.array(xs), np.array(ys)