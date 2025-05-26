import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TS_GC(nn.Module):
    """单个序列的时空处理器"""
    def __init__(self, series_num, feature_dim, temporal_layers, kernel_size, dropout):
        super(TS_GC, self).__init__()
        self.first_conv = nn.Conv1d(series_num, feature_dim, kernel_size, padding=kernel_size//2)
        
        # 时间特征提取 - 多层扩张卷积
        self.temporal_layers = nn.ModuleList()
        for i in range(temporal_layers):
            dilation = 2 ** i
            self.temporal_layers.append(
                nn.Sequential(
                    nn.Conv1d(feature_dim, feature_dim, kernel_size, 
                             padding=(kernel_size-1)*dilation//2, dilation=dilation),
                    nn.BatchNorm1d(feature_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                )
            )
        
        # 空间特征提取 - 多头注意力
        self.spatial_attention = nn.MultiheadAttention(
            embed_dim=feature_dim, num_heads=4, dropout=dropout, batch_first=True
        )
        self.spatial_norm = nn.LayerNorm(feature_dim)
        
        # 特征融合
        self.feature_fusion = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 预测头
        self.prediction_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim // 2, 1)  # 输出单个值
        )
    
    def forward(self, x):
        """
        Args:
            x: [batch_size, series_num, input_window]
        Returns:
            prediction: [batch_size, 1]
        """
        # 第一层特征提取
        features = self.first_conv(x)  # [batch_size, feature_dim, input_window]
        features = F.relu(features)
        
        # 时间特征提取
        for temporal_layer in self.temporal_layers:
            residual = features
            features = temporal_layer(features)
            features = features + residual  # 残差连接
        
        # 空间特征提取
        # 转换维度: [batch_size, input_window, feature_dim]
        spatial_input = features.permute(0, 2, 1)
        spatial_output, _ = self.spatial_attention(spatial_input, spatial_input, spatial_input)
        spatial_output = self.spatial_norm(spatial_input + spatial_output)
        
        # 转换回来: [batch_size, feature_dim, input_window]
        features = spatial_output.permute(0, 2, 1)
        
        # 特征融合
        fused_features = self.feature_fusion(features)  # [batch_size, feature_dim]
        
        # 预测
        prediction = self.prediction_head(fused_features)  # [batch_size, 1]
        
        return prediction
    
    def get_first_conv_weights(self):
        """获取第一层Conv1d的权重"""
        return self.first_conv.weight

class MutiTS_GC(nn.Module):
    def __init__(self, input_window, output_window, series_num,
                 feature_dim=64, temporal_layers=3, kernel_size=3, dropout=0.1, device='cpu'):
        super(MutiTS_GC, self).__init__()
        self.input_window = input_window
        self.output_window = output_window
        self.series_num = series_num
        self.feature_dim = feature_dim
        self.device = device
        
        # 配置信息
        self.config = {
            'input_window': self.input_window,
            'output_window': self.output_window,
            'series_num': self.series_num,
            'feature_dim': feature_dim,
            'temporal_layers': temporal_layers,
            'kernel_size': kernel_size,
            'dropout': dropout
        }
        
        # 为每个时间序列创建单独的处理器
        self.processors = nn.ModuleList()
        for i in range(self.series_num):
            self.processors.append(
                TS_GC(
                    series_num=self.series_num,
                    feature_dim=feature_dim,
                    temporal_layers=temporal_layers,
                    kernel_size=kernel_size,
                    dropout=dropout
                )
            )
        
        self.init_weights()
    
    def init_weights(self):
        """权重初始化"""
        for processor in self.processors:
            for module in processor.modules():
                if isinstance(module, nn.Conv1d):
                    nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
                elif isinstance(module, nn.Linear):
                    nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        """
        Args:
            x: [batch_size, input_window, series_num]
        Returns:
            output: [batch_size, output_window, series_num]
        """
        x_input = x.permute(0, 2, 1)  # [batch_size, series_num, input_window]
        
        # 为每个时间序列运行对应的处理器
        all_predictions = []
        for i in range(self.series_num):
            prediction = self.processors[i](x_input)  # [batch_size, 1]
            all_predictions.append(prediction)
        
        # 组合所有预测结果
        predictions = torch.cat(all_predictions, dim=1)  # [batch_size, series_num]
        predictions = predictions.unsqueeze(1)  # [batch_size, 1, series_num]
        
        return predictions  # [batch_size, output_window, series_num]
    
    def get_first_layer_weights(self):
        all_weights = []
        for i in range(self.series_num):
            weights = self.processors[i].get_first_conv_weights()
            all_weights.append(weights)
        return all_weights
    
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
            kernel_size = self.processors[0].first_conv.weight.shape[2]
            gc_matrix = torch.zeros(self.series_num, self.series_num, kernel_size, device=device_to_use)
        
        for i in range(self.series_num):  # 目标序列
            # 获取第i个处理器的第一层卷积权重
            weights = self.processors[i].first_conv.weight
            # weights.shape: [feature_dim, series_num, kernel_size]
            
            for j in range(self.series_num):  # 源序列
                if ignore_kernel:
                    # 计算所有卷积核位置和特征维度的L2范数
                    gc_matrix[i, j] = torch.norm(weights[:, j, :])
                else:
                    # 保留每个卷积核位置的信息
                    for k_idx in range(weights.shape[2]):
                        gc_matrix[i, j, k_idx] = torch.norm(weights[:, j, k_idx])
        
        if threshold:
            return (gc_matrix > weight_threshold).int()
        else:
            return gc_matrix