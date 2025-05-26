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
        
        # 空间特征提取 - 使用自适应平均池化和全连接
        self.spatial_processor = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Linear(feature_dim, feature_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Unflatten(1, (feature_dim, 1))
            )
        
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
        origin_features = self.first_conv(x)
        origin_features = F.relu(origin_features)
        
        # 时间特征提取
        time_features = origin_features
        for temporal_layer in self.temporal_layers:
            residual = time_features
            time_features = temporal_layer(time_features)
            time_features = time_features + residual  # 残差连接
        
        # 空间特征提取
        space_features = self.spatial_processor(origin_features)
        combined_features = time_features + space_features # 特征融合
        combined = 0.8 * origin_features + 0.2 * combined_features
        
        fused_features = self.feature_fusion(combined)
        prediction = self.prediction_head(fused_features)
        
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
        if ignore_kernel: # 如果忽略滞后
            GC = [torch.norm(net.get_first_conv_weights(), dim=(0, 2))
                  for net in self.processors]
        else:
            GC = [torch.norm(net.get_first_conv_weights(), dim=0)
                  for net in self.processors]
        GC = torch.stack(GC) 
        if threshold:  # 如果需要进行阈值处理
            return (GC > 0).int() # 变成0和1
        else:
            return GC