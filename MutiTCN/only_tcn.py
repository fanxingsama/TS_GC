import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
sys.path.append("../")
from model.granger_tcn import GrangerTCN

class MultiTCNModel(nn.Module):
    """
    基于多个TCN的预测模型
    
    Args:
        input_window (int): 输入窗口长度
        output_window (int): 输出窗口长度  
        series_num (int): 时间序列数量
        feature_dim (int): 输入特征维度
        output_dim (int): 输出维度
        tcn_channels (int): TCN隐藏通道数
        kernel_size (int): 卷积核大小
        dropout (float): Dropout率
        device: 计算设备
    """
    def __init__(self, input_window, output_window, series_num, feature_dim, output_dim, 
                 tcn_channels, kernel_size, dropout, device):
        super(MultiTCNModel, self).__init__()
        self.input_window = input_window
        self.output_window = output_window
        self.series_num = series_num
        self.feature_dim = feature_dim
        self.output_dim = output_dim
        self.device = device
        
        # 配置信息
        self.config = {
            'input_window': self.input_window,
            'output_window': self.output_window,
            'series_num': self.series_num,
            'feature_dim': self.feature_dim,
            'output_dim': self.output_dim,
            'tcn_channels': tcn_channels,
            'kernel_size': kernel_size,
            'dropout': dropout
        }
        
        # 为每个时间序列创建单独的TCN
        self.tcn_processors = nn.ModuleList()
        for i in range(self.series_num):
            self.tcn_processors.append(
                GrangerTCN(input_series_num=self.series_num - 1,  # 排除目标序列自身
                          output_size=tcn_channels,
                          TCN_hidden_channels=tcn_channels,
                          kernel_size=kernel_size,
                          dropout=dropout)
            )
        
        # 特征融合层
        self.feature_fusion = nn.Linear(tcn_channels, tcn_channels)
        
        # 预测头
        self.prediction_head = nn.Sequential(
            nn.Linear(tcn_channels, tcn_channels // 2),
            nn.PReLU(),
            nn.Dropout(dropout),
            nn.Linear(tcn_channels // 2, output_dim)
        )
        
        self.init_weights()
    
    def init_weights(self):
        """初始化权重"""
        nn.init.kaiming_normal_(self.feature_fusion.weight, mode='fan_in', nonlinearity='leaky_relu')
        if self.feature_fusion.bias is not None:
            nn.init.constant_(self.feature_fusion.bias, 0)
            
        for layer in self.prediction_head:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='leaky_relu')
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
    
    def forward(self, x):
        """
        Args:
            x: [batch_size, input_window, series_num, feature_dim]
        Returns:
            output: [batch_size, output_window, series_num, output_dim]
        """
        
        # 如果特征维度大于1，只取第一个特征；如果为1，则去掉最后一个维度
        if self.feature_dim == 1:
            x_tcn_input = x.squeeze(-1)  # [batch_size, input_window, series_num]
        else:
            x_tcn_input = x[..., 0]  # [batch_size, input_window, series_num]
        
        x_tcn_input = x_tcn_input.permute(0, 2, 1)  # [batch_size, series_num, input_window]
        
        # 为每个时间序列运行对应的TCN
        all_tcn_outputs = []
        for i in range(self.series_num):
            # 创建掩码，排除目标序列
            mask = torch.ones(self.series_num, dtype=torch.bool, device=self.device)
            mask[i] = False
            other_series = x_tcn_input[:, mask, :]  # [batch_size, series_num-1, input_window]
            
            # 通过TCN处理
            tcn_out = self.tcn_processors[i](other_series)  # [batch_size, tcn_channels, input_window]
            all_tcn_outputs.append(tcn_out)
        
        # 堆叠所有TCN输出
        stacked_outputs = torch.stack(all_tcn_outputs, dim=1)  # [batch_size, series_num, tcn_channels, input_window]
        
        # 取最后一个时间步的特征用于预测
        last_step_features = stacked_outputs[:, :, :, -1]  # [batch_size, series_num, tcn_channels]
        
        # 特征融合
        fused_features = self.feature_fusion(last_step_features)  # [batch_size, series_num, tcn_channels]
        fused_features = F.relu(fused_features)
        
        # 预测
        predictions = self.prediction_head(fused_features)  # [batch_size, series_num, output_dim]
        
        # 调整输出形状以匹配期望的输出格式
        predictions = predictions.unsqueeze(1)  # [batch_size, 1, series_num, output_dim]
        
        # 如果需要多步预测，重复输出
        if self.output_window > 1:
            predictions = predictions.repeat(1, self.output_window, 1, 1)
        
        return predictions  # [batch_size, output_window, series_num, output_dim]
    
    def GC(self, threshold=False, ignore_kernel=True, weight_threshold=0.0):
        """
        获取格兰杰因果矩阵
        
        Args:
            threshold (bool): 是否将权重转换为二值
            ignore_kernel (bool): 是否忽略卷积核维度
            weight_threshold (float): 权重阈值
            
        Returns:
            gc_matrix: 格兰杰因果矩阵
        """
        device_to_use = self.device
        
        if ignore_kernel:
            gc_matrix = torch.zeros(self.series_num, self.series_num, device=device_to_use)
        else:
            # 获取卷积核大小
            kernel_size = self.tcn_processors[0].get_first_block_conv1_weights().shape[2]
            gc_matrix = torch.zeros(self.series_num, self.series_num, kernel_size, device=device_to_use)
        
        for i in range(self.series_num):  # 目标序列
            # 获取第i个TCN的第一层卷积权重
            weights = self.tcn_processors[i].get_first_block_conv1_weights()
            # weights.shape: [output_channels, input_series_num-1, kernel_size]
            
            current_tcn_channel_idx = 0
            for j in range(self.series_num):  # 源序列
                if i == j:  # 跳过自身
                    continue
                    
                if current_tcn_channel_idx < weights.shape[1]:
                    if ignore_kernel:
                        # 计算所有卷积核位置的L2范数
                        gc_matrix[i, j] = torch.norm(weights[:, current_tcn_channel_idx, :])
                    else:
                        # 保留每个卷积核位置的信息
                        for k_idx in range(weights.shape[2]):
                            gc_matrix[i, j, k_idx] = torch.norm(weights[:, current_tcn_channel_idx, k_idx])
                
                current_tcn_channel_idx += 1
        
        if threshold:
            return (gc_matrix > weight_threshold).int()
        else:
            return gc_matrix