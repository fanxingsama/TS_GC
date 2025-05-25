import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from model.granger_tcn import GrangerTCN

def activation_helper(activation):
    if activation == 'relu':
        return nn.ReLU()
    elif activation == 'prelu':
        return nn.PReLU()
    elif activation == 'leaky_relu':
        return nn.LeakyReLU()
    elif activation == 'tanh':
        return nn.Tanh()
    elif activation == 'sigmoid':
        return nn.Sigmoid()
    else:
        return nn.ReLU()

class MLP(nn.Module):
    def __init__(self, series_num, kernel_size, hidden, activation):
        super(MLP, self).__init__()
        self.activation = activation_helper(activation)
        # 使用卷积来构建模型的第一层，输入维度为num_series，输出维度为hidden[0]，卷积核大小是滞后值
        layer = nn.Conv1d(series_num, hidden[0], kernel_size) 
        # 输入数据的形状(batch_size, series_num, T)，经过第一层后，输出的形状 (batch_size, hidden[0], T - lag + 1)。
        modules = [layer]

        for d_in, d_out in zip(hidden, hidden[1:] + [1]):
            layer = nn.Conv1d(d_in, d_out, 1) 
            modules.append(layer)

        # 把这些层注册到模型中
        self.layers = nn.ModuleList(modules)

    def forward(self, X):
        X = X.transpose(2, 1) # 变成[batch_size, series_num, series_length]
        for i, fc in enumerate(self.layers):
            if i != 0: # 如果当前层不是第一层
                X = self.activation(X) # 使用激活函数进行激活
            X = fc(X)

        return X.transpose(2, 1)

    def get_first_layer_weights(self):
        """获取第一层权重"""
        return self.layers[0].weight

class cMLP(nn.Module):
    def __init__(self, series_num, kernel_size, mlp_hidden, mlp_activation):
        super(cMLP, self).__init__()
        self.networks = nn.ModuleList([
            MLP(series_num, kernel_size, mlp_hidden, mlp_activation)
            for _ in range(series_num)])

    def forward(self, X):
        return torch.cat([network(X) for network in self.networks], dim=2)

    def GC(self, threshold=True, ignore_lag=True):
        if ignore_lag: # 如果忽略滞后
            GC = [torch.norm(net.layers[0].weight, dim=(0, 2))
                  for net in self.networks]
        else:
            GC = [torch.norm(net.layers[0].weight, dim=0)
                  for net in self.networks]
        GC = torch.stack(GC) # 堆叠成一个张量
        if threshold: # 如果需要进行阈值处理
            return (GC > 0).int() # 变成0和1
        else:
            return GC
        
    def get_first_layer_params(self):
        return [mlp.layers[0].weight for mlp in self.networks]
         

class TCN_cMLP_Model(nn.Module):
    """
    基于TCN特征提取 + 多个MLP预测的模型结构
    
    Args:
        input_window (int): 输入窗口长度
        output_window (int): 输出窗口长度  
        series_num (int): 时间序列数量
        feature_dim (int): 输入特征维度
        output_dim (int): 输出维度
        tcn_channels (int): TCN隐藏通道数
        kernel_size (int): TCN卷积核大小
        dropout (float): Dropout率
        mlp_hidden (list): MLP隐藏层维度列表
        mlp_activation (str): MLP激活函数
        device: 计算设备
    """
    def __init__(self, input_window, output_window, series_num, feature_dim, output_dim, 
                 tcn_channels, kernel_size, dropout, mlp_hidden, mlp_activation='prelu', device=None):
        super(TCN_cMLP_Model, self).__init__()
        self.input_window = input_window
        self.output_window = output_window
        self.series_num = series_num
        self.feature_dim = feature_dim
        self.output_dim = output_dim
        self.device = device
        self.mlp_hidden = mlp_hidden
        
        # 配置信息
        self.config = {
            'input_window': self.input_window,
            'output_window': self.output_window,
            'series_num': self.series_num,
            'feature_dim': self.feature_dim,
            'output_dim': self.output_dim,
            'tcn_channels': tcn_channels,
            'kernel_size': kernel_size,
            'dropout': dropout,
            'mlp_hidden': mlp_hidden,
            'mlp_activation': mlp_activation
        }
        
        # 共享的TCN特征提取器
        self.tcn_feature_extractor = GrangerTCN(
            input_series_num=self.series_num,
            output_size=tcn_channels,
            TCN_hidden_channels=tcn_channels,
            kernel_size=kernel_size,
            dropout=dropout
        )
        
        # 特征维度适配层
        # TCN输出是 [batch_size, tcn_channels, sequence_length]
        # 我们需要将其转换为适合MLP的输入格式
        self.feature_adapter = nn.Linear(tcn_channels, self.series_num)
        
        # 组件化MLP，每个序列一个MLP
        self.cmlp = cMLP(
            num_series=self.series_num,
            hidden=mlp_hidden,
            activation=mlp_activation,
            kernel_size=kernel_size
        )
        
        # 输出适配层
        self.output_adapter = nn.Linear(1, output_dim)
        
        self.init_weights()
    
    def init_weights(self):
        """初始化权重"""
        nn.init.kaiming_normal_(self.feature_adapter.weight, mode='fan_in', nonlinearity='leaky_relu')
        if self.feature_adapter.bias is not None:
            nn.init.constant_(self.feature_adapter.bias, 0)
            
        nn.init.kaiming_normal_(self.output_adapter.weight, mode='fan_in', nonlinearity='leaky_relu')
        if self.output_adapter.bias is not None:
            nn.init.constant_(self.output_adapter.bias, 0)
    
    def forward(self, x):
        """
        Args:
            x: [batch_size, input_window, series_num, feature_dim]
        Returns:
            output: [batch_size, output_window, series_num, output_dim]
        """
        batch_size = x.shape[0]
        
        # 处理输入特征维度
        if self.feature_dim == 1:
            x_tcn_input = x.squeeze(-1)  # [batch_size, input_window, series_num]
        else:
            x_tcn_input = x[..., 0]  # [batch_size, input_window, series_num]
        
        x_tcn_input = x_tcn_input.permute(0, 2, 1)  # [batch_size, series_num, input_window]
        
        # 通过TCN提取特征
        tcn_features = self.tcn_feature_extractor(x_tcn_input) 
        
        # 转换特征格式以适配MLP
        tcn_features = tcn_features.permute(0, 2, 1)  # [batch_size, input_window, tcn_channels]
        adapted_features = self.feature_adapter(tcn_features)  # [batch_size, input_window, series_num]
        
        # 通过组件化MLP进行预测
        combined_output = self.cmlp(adapted_features)  # [batch_size, T-lag+1, series_num]
        
        # 取最后一个时间步的输出用于预测
        if combined_output.shape[1] > 0:
            predictions = combined_output[:, -1:, :]  # [batch_size, 1, series_num]
        else:
            predictions = torch.zeros(batch_size, 1, self.series_num, device=x.device)
        
        # 适配输出维度
        predictions = self.output_adapter(predictions.unsqueeze(-1))  # [batch_size, 1, series_num, output_dim]
        
        # 如果需要多步预测，重复输出
        if self.output_window > 1:
            predictions = predictions.repeat(1, self.output_window, 1, 1)
        
        return predictions  # [batch_size, output_window, series_num, output_dim]
    
    def GC(self, threshold=True, ignore_kernel=True, weight_threshold=0.0):
        """
        计算Granger因果关系矩阵
        
        Args:
            threshold (bool): 是否返回阈值化的二值矩阵
            ignore_kernel (bool): 是否忽略卷积核维度
            weight_threshold (float): 权重阈值
        
        Returns:
            torch.Tensor: Granger因果关系矩阵
        """
        if ignore_kernel:
            # 计算每个MLP第一层权重的范数，忽略lag维度
            GC = [torch.norm(mlp.layers[0].weight, dim=(0, 2)) for mlp in self.cmlp.networks]
        else:
            # 保留lag维度
            GC = [torch.norm(mlp.layers[0].weight, dim=0) for mlp in self.cmlp.networks]
        
        GC = torch.stack(GC)  # 堆叠成张量
        
        # [series_num, series_num] 或 [series_num, series_num, lag]
        if threshold:
            return (GC > weight_threshold).int()  # 转换为二值矩阵
        else:
            return GC
    
    def get_first_layer_params(self):
        """获取所有MLP第一层的参数"""
        return [mlp.layers[0].weight for mlp in self.cmlp.networks]
