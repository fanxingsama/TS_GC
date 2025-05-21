import numpy as np
import torch
import torch.nn as nn
import math
from .TCN_granger.granger_tcn import GrangerTCN
Linear = nn.Linear
LayerNorm = nn.LayerNorm
Dropout = nn.Dropout
Softmax = nn.Softmax
LeakyReLU = nn.LeakyReLU

class SimplifiedGrangerModel(nn.Module):
    def __init__(self, config, tcn_channels, kernel_size, dropout):
        super().__init__()
        self.series_num = config['data_loader']['args']['series_num']
        self.input_window = config['data_loader']['args']['input_window']
        
        # 只使用TCN，不使用复杂的注意力机制
        self.tcn_processors = nn.ModuleList()
        for i in range(self.series_num):
            self.tcn_processors.append(
                GrangerTCN(
                    input_series_num=self.series_num - 1,  # 排除自身
                    output_size=1,  # 简化输出
                    TCN_hidden_channels=tcn_channels,
                    kernel_size=kernel_size,
                    dropout=dropout
                )
            )
        
        self.prediction_layer = nn.Linear(self.input_window, 1)
    
    def forward(self, x):
        # x: [batch_size, input_window, series_num, feature_dim]
        batch_size = x.size(0)
        predictions = []
        
        for i in range(self.series_num):
            # 排除目标序列
            mask = torch.ones(self.series_num, dtype=torch.bool)
            mask[i] = False
            other_series = x[:, :, mask, 0].permute(0, 2, 1)  # [batch, other_series, time]
            
            # TCN处理
            tcn_out = self.tcn_processors[i](other_series)  # [batch, output_size, time]
            
            # 预测
            pred = self.prediction_layer(tcn_out.mean(dim=1))  # [batch, 1]
            predictions.append(pred)
        
        # 组合预测结果
        predictions = torch.stack(predictions, dim=2)  # [batch, 1, series_num]
        return predictions.unsqueeze(-1)  # [batch, 1, series_num, 1]