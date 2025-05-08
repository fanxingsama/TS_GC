import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn import preprocessing

# 输入：数据集的路径和真实因果图路径
# 输出：划分好的dataloader和格兰杰因果图


class TimeseriesDataLoader(DataLoader):
    def __init__(self, data_dir=None, dataset=None, batch_size=32, time_step=None, 
                 output_window=None, feature_dim=None, output_dim=None,
                 shuffle=True, validation_split=0.0, num_workers=1, collate_fn=default_collate):
        '''
        data_dir：数据文件的路径。可选，当提供dataset时忽略此参数。
        dataset：预先构建的数据集。可选，如果提供则忽略data_dir。
        batch_size：每个批次的样本数量。
        time_step：输入窗口的大小，即每个样本的时间步长。
        output_window：输出窗口的大小，即预测的时间步长。
        feature_dim：每个时间步的特征维度。
        output_dim：输出的维度。
        shuffle：是否随机打乱数据，默认为True。
        validation_split：验证集的比例或数量，默认为0.0。
        num_workers：加载数据时使用的子进程数量，默认为1。
        collate_fn：用于将多个样本组合成一个批次的函数，默认为default_collate。
        '''
        # 处理数据集
        if dataset is None and data_dir is not None:
            # 加载数据
            self.data_dir = data_dir
            self.df_data = pd.read_csv(self.data_dir)
            self.data_len = len(self.df_data.index)
            self.data = self.df_data.values.astype('float32')
            
            # 初始化参数
            self.time_step = time_step
            self.output_window = output_window
            self.series_num = self.data.shape[1]
            self.feature_dim = feature_dim
            self.output_dim = output_dim
            
            # 归一化数据
            scaler = preprocessing.MinMaxScaler(feature_range=(0.5, 1)) # 将数据缩放到 [0.5, 1] 范围内。
            self.data = scaler.fit_transform(self.data)
            
            # 构造输入样本
            self.dataset = []
            assert self.time_step < len(self.data) + 1, "确保输入窗口长度小于数据长度"
            assert self.output_window < self.time_step, "确保输出窗口长度小于输入窗口长度"
            
            '''
            遍历数据，构造每个样本，每个样本包含两个部分：
            输入数据：从 i - self.time_step 到 i 的时间步长内的数据，形状为 (time_step, series_num, feature_dim)。
            输出数据：从 i - self.output_window 到 i 的时间步长内的数据，形状为 (output_window, series_num, output_dim)。
            '''
            for i in range(self.time_step, len(self.data) + 1):
                self.dataset.append((
                    self.data[i - self.time_step:i].reshape(self.time_step, self.series_num, self.feature_dim),
                    self.data[i - self.output_window:i].reshape(self.output_window, self.series_num, self.output_dim)
                ))
        else:
            self.dataset = dataset
        
        # 初始化验证集分割参数
        self.validation_split = validation_split
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.batch_idx = 0
        self.n_samples = len(self.dataset)
        
        # 创建训练集和验证集的采样器
        self.sampler, self.valid_sampler = self._split_sampler(self.validation_split)
        
        # 初始化DataLoader参数
        self.init_kwargs = {
            'dataset': self.dataset,
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'collate_fn': collate_fn,
            'num_workers': num_workers
        }
        
        # 初始化父类DataLoader
        super().__init__(sampler=self.sampler, **self.init_kwargs)
    
    #  划分数据集
    def _split_sampler(self, split):
        if split == 0.0: # 如果 split 为 0.0，表示不划分验证集，返回 None
            return None, None
        
        idx_full = np.arange(self.n_samples) # 生成一个包含所有样本索引的数组 idx_full。
        
        np.random.seed(0)
        np.random.shuffle(idx_full)
        
        # 根据split的类型，划分数据集
        # 如果 split 是整数，表示验证集的样本数量；如果 split 是浮点数，表示验证集的比例。
        if isinstance(split, int):
            assert split > 0
            assert split < self.n_samples, "验证集大小被配置为大于整个数据集。"
            len_valid = split
        else:
            len_valid = int(self.n_samples * split)
        
        # 划分验证集和训练集的索引
        valid_idx = idx_full[0:len_valid]
        train_idx = np.delete(idx_full, np.arange(0, len_valid))
        
        # 使用 SubsetRandomSampler（子随机采样器） 创建训练集和验证集的采样器
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)
        
        # 关闭 shuffle 选项，因为采样器已经提供了随机性
        self.shuffle = False
        self.n_samples = len(train_idx)
        
        # 返回训练集和验证集的采样器
        return train_sampler, valid_sampler
    
    # 创建验证集的DataLoader
    def split_validation(self):
        # 如果没有验证集采样器，返回None
        if self.valid_sampler is None:
            return None
        else:
            # 使用验证集采样器创建一个新的 DataLoader 实例，返回验证集的数据加载器。
            return DataLoader(sampler=self.valid_sampler, **self.init_kwargs)