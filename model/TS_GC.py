import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Permute(nn.Module):
    def __init__(self, *dims):
        super().__init__()
        self.dims = dims
        
    def forward(self, x):
        return x.permute(*self.dims)

class SelectLastTimeStep(nn.Module):
    def forward(self, x):
        # x: [batch_size, feature_dim, seq_len]
        return x[:, :, -1] # [batch_size, feature_dim]

class TS_GC(nn.Module):
    def __init__(self, series_num, feature_dim, temporal_layers, kernel_size, dropout, output_window):
        super(TS_GC, self).__init__()
        self.first_conv = nn.Conv1d(series_num, feature_dim, kernel_size, padding=kernel_size//2)
        
        # 时间层
        self.temporal_layers = nn.ModuleList()
        for i in range(temporal_layers):
            dilation = 2 ** i
            self.temporal_layers.append(
                nn.Sequential(
                    nn.Conv1d(feature_dim, feature_dim, kernel_size, 
                             padding=(kernel_size-1)*dilation//2, dilation=dilation),
                    nn.BatchNorm1d(feature_dim),
                    nn.PReLU(),
                    nn.Dropout(dropout)
                )
            )
        
        # 空间层
        self.spatial_processor = nn.Sequential(
                nn.AdaptiveAvgPool1d(1), # 输出: [batch_size, feature_dim, 1]
                nn.Flatten(), # 输出: [batch_size, feature_dim]
                nn.Linear(feature_dim, feature_dim), # 输出: [batch_size, feature_dim]
                nn.PReLU(),
                nn.Dropout(dropout),
                nn.Unflatten(1, (feature_dim, 1)) # 输出: [batch_size, feature_dim, 1]
            )
        
        self.feature_fusion = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), # 输出: [batch_size, feature_dim, 1] (假设输入是 [batch_size, feature_dim, seq_len])
            nn.Flatten(), # 输出: [batch_size, feature_dim]
            nn.Linear(feature_dim, feature_dim), # 输出: [batch_size, feature_dim]
            nn.PReLU(),
            nn.Dropout(dropout)
        )
        
        self.prediction_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2), # 输出: [batch_size, feature_dim // 2]
            nn.PReLU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim // 2, output_window) # 输出: [batch_size, output_window]
        )
    
    def forward(self, x):
        # x: [batch_size, series_num, input_window]
        origin_features = self.first_conv(x) # origin_features: [batch_size, feature_dim, input_window]
        origin_features = F.relu(origin_features) # origin_features: [batch_size, feature_dim, input_window]
        
        time_features = origin_features # time_features: [batch_size, feature_dim, input_window]
        for temporal_layer in self.temporal_layers:
            residual = time_features # residual: [batch_size, feature_dim, input_window]
            time_features = temporal_layer(time_features) # time_features: [batch_size, feature_dim, input_window]
            time_features = time_features + residual # time_features: [batch_size, feature_dim, input_window]
        
        space_features = self.spatial_processor(origin_features) # space_features: [batch_size, feature_dim, 1]
        # space_features 广播到 time_features 的形状
        combined_features = time_features + space_features # combined_features: [batch_size, feature_dim, input_window] (通过广播)
        
        # combined 的计算方式也依赖于上述的维度对齐
        combined = 0.5 * origin_features + 0.5 * combined_features # combined: [batch_size, feature_dim, input_window]
        
        fused_features = self.feature_fusion(combined) # fused_features: [batch_size, feature_dim]
        prediction = self.prediction_head(fused_features) # prediction: [batch_size, output_window]
        
        return prediction
    
    def get_first_conv_weights(self):
        return self.first_conv.weight

class MutiTS_GC(nn.Module):
    def __init__(self, input_window, output_window, series_num,
                 feature_dim, temporal_layers, kernel_size, dropout, device):
        super(MutiTS_GC, self).__init__()
        self.input_window = input_window
        self.output_window = output_window
        self.series_num = series_num 
        self.feature_dim = feature_dim
        self.device = device
        
        self.config = {
            'input_window': self.input_window,
            'output_window': self.output_window,
            'series_num': self.series_num,
            'feature_dim': feature_dim,
            'temporal_layers': temporal_layers,
            'kernel_size': kernel_size,
            'dropout': dropout
        }
        
        self.networks = nn.ModuleList()
        for i in range(self.series_num):
            self.networks.append(
                TS_GC(
                    series_num=self.series_num,
                    feature_dim=feature_dim,
                    temporal_layers=temporal_layers,
                    kernel_size=kernel_size,
                    dropout=dropout,
                    output_window=self.output_window 
                )
            )

    def forward(self, x):
        # x: [batch_size, input_window, series_num] 
        x_input = x.permute(0, 2, 1) # x_input: [batch_size, series_num, input_window] (调整以匹配Conv1d的channels_in)
        
        all_predictions = []
        for i in range(self.series_num):
            # channel_mask = torch.ones(self.series_num, dtype=torch.bool, device=x.device)
            # channel_mask[i] = False
            
            # input_for_network_i = x_input[:, channel_mask, :] # 遮蔽掉目标序列的值
            
            # prediction = self.networks[i](input_for_network_i) # prediction : [batch_size, output_window]
            prediction = self.networks[i](x_input) # prediction : [batch_size, output_window]
            all_predictions.append(prediction)
        
        # all_predictions 是一个长度为 series_num 的列表
        # 每个元素是形状为 [batch_size, output_window] 的张量
        predictions = torch.stack(all_predictions, dim=2) # predictions: [batch_size, output_window, series_num]
        
        return predictions
    
    def get_first_layer_weights(self):
        all_weights = []
        for i in range(self.series_num):
            weights = self.networks[i].get_first_conv_weights()
            all_weights.append(weights)
        return all_weights
    
    def GC(self, threshold=False, ignore_kernel=True, weight_threshold=0.0):
        if ignore_kernel:
            GC_val = [torch.norm(net.get_first_conv_weights(), dim=(0, 2))
                      for net in self.networks]
        else:
            GC_val = [torch.norm(net.get_first_conv_weights(), p = 1,dim=0)
                      for net in self.networks]
        GC_val = torch.stack(GC_val) 
        if threshold:
            return (GC_val > weight_threshold).int() 
        else:
            return GC_val
