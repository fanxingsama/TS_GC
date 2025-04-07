from base import BaseDataLoader
import pandas as pd
import torch
import math
from sklearn import preprocessing

class TimeseriesDataLoader(BaseDataLoader):
    '''
    data_dir：数据文件的路径。
    batch_size：每个批次的样本数量。
    time_step：输入窗口的大小，即每个样本的时间步长。
    output_window：输出窗口的大小，即预测的时间步长。
    feature_dim：每个时间步的特征维度。
    output_dim：输出的维度。
    shuffle：是否打乱数据，默认为 True。
    validation_split：用于验证的数据比例，默认为 0.0。
    num_workers：加载数据时使用的线程数，默认为 1。
    training：是否为训练模式，默认为 True。
    '''
    def __init__(self, data_dir, batch_size, time_step, output_window, feature_dim, output_dim, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        # 加载数据
        self.data_dir = data_dir
        self.df_data = pd.read_csv(self.data_dir)
        self.data_len = len(self.df_data.index)
        self.data = self.df_data.values.astype('float32')

        # 初始化参数
        self.batch_size = batch_size
        self.time_step = time_step
        self.output_window = output_window
        self.series_num = self.data.shape[1]
        self.feature_dim = feature_dim
        self.output_dim = output_dim
        
        # 归一化数据
        scaler = preprocessing.MinMaxScaler(feature_range=(0.5,1 )) # 将数据缩放到 [0.5, 1] 范围内。
        self.data = scaler.fit_transform(self.data)
        
        # 构造输入样本
        self.dataset = []
        assert self.time_step<len(self.data)+1, "确保输入窗口长度小于数据长度"
        assert self.output_window<self.time_step, "确保输出窗口长度小于输入窗口长度"
        '''
        遍历数据，构造每个样本，每个样本包含两个部分：
        输入数据：从 i - self.time_step 到 i 的时间步长内的数据，形状为 (time_step, series_num, feature_dim)。
        输出数据：从 i - self.output_window 到 i 的时间步长内的数据，形状为 (output_window, series_num, output_dim)。
        '''
        for i in range(self.time_step,len(self.data)+1):
            self.dataset.append((self.data[i-self.time_step:i].reshape(self.time_step,self.series_num,self.feature_dim) ,
                                 self.data[i-self.output_window:i].reshape(self.output_window,self.series_num,self.output_dim)))
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers) # 调用父类的构造函数，把构造好的数据传递给父类