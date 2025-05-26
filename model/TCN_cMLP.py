import torch
import torch.nn as nn
import torch.nn.functional as F
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
    def __init__(self, input_dim, hidden_dims, activation='prelu', output_dim=1, kernel_size=1):
        super(MLP, self).__init__()
        self.activation = activation_helper(activation)
        self.kernel_size = kernel_size
        
        # 第一层使用1D卷积来保持与训练代码的兼容性
        # 输入: [batch_size, input_dim, 1] -> 输出: [batch_size, hidden_dims[0], 1]
        self.first_layer = nn.Conv1d(input_dim, hidden_dims[0], kernel_size=kernel_size)
        
        # 后续层使用全连接层
        layers = []
        dims = hidden_dims + [output_dim]
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
        
        self.layers = nn.ModuleList(layers)
    
    def forward(self, x):
        # x: [batch_size, input_dim]
        # 为卷积层添加时间维度
        x = x.unsqueeze(-1)  # [batch_size, input_dim, 1]
        
        # 第一层卷积
        x = self.first_layer(x)  # [batch_size, hidden_dims[0], 1]
        x = self.activation(x)
        
        # 移除时间维度用于全连接层
        x = x.squeeze(-1)  # [batch_size, hidden_dims[0]]
        
        # 后续全连接层
        for i, layer in enumerate(self.layers):
            x = layer(x)
            # 在最后一层之前应用激活函数
            if i < len(self.layers) - 1:
                x = self.activation(x)
        return x
    
    def get_first_layer_weights(self):
        return self.first_layer.weight

class cMLP(nn.Module):
    def __init__(self, input_dim, series_num, mlp_hidden, mlp_activation='prelu', output_dim=1, kernel_size=1):
        super(cMLP, self).__init__()
        self.series_num = series_num
        self.input_dim = input_dim
        self.kernel_size = kernel_size
        
        # 为每个目标序列创建一个MLP
        self.networks = nn.ModuleList([
            MLP(input_dim, mlp_hidden, mlp_activation, output_dim, kernel_size)
            for _ in range(series_num)
        ])
    
    def forward(self, x):
        """
        Args:
            x: [batch_size, input_dim] - 保持序列对应关系的特征
        Returns:
            outputs: [batch_size, series_num, output_dim]
        """
        outputs = []
        for i, network in enumerate(self.networks):
            output = network(x)  # [batch_size, output_dim]
            outputs.append(output)
        
        # 堆叠所有输出
        return torch.stack(outputs, dim=1)  # [batch_size, series_num, output_dim]
    
    def GC(self, threshold=True, weight_threshold=0.0, ignore_lag=True):
        """
        计算Granger因果关系矩阵
        
        Args:
            threshold (bool): 是否返回阈值化的二值矩阵
            weight_threshold (float): 权重阈值
            ignore_lag (bool): 是否忽略lag维度（与训练代码兼容）
        
        Returns:
            torch.Tensor: Granger因果关系矩阵 [series_num, series_num]
        """
        if ignore_lag:
            # 计算每个MLP第一层权重的范数，忽略kernel维度
            GC = [torch.norm(net.first_layer.weight, dim=(0, 2)) for net in self.networks]
        else:
            # 保留所有维度
            GC = [torch.norm(net.first_layer.weight, dim=0) for net in self.networks]
        
        GC = torch.stack(GC)  # [series_num, input_dim] 堆叠成张量
        
        if threshold:
            return (GC > weight_threshold).int()
        else:
            return GC
        
    def get_first_layer_params(self):
        """获取所有MLP第一层的参数"""
        return [mlp.first_layer.weight for mlp in self.networks]


class TCN_cMLP_Model(nn.Module):
    def __init__(self, input_window, output_window, series_num, feature_dim, output_dim, 
                 tcn_channels, kernel_size, dropout, mlp_hidden, mlp_activation='prelu', device=None):
        super(TCN_cMLP_Model, self).__init__()
        self.input_window = input_window
        self.output_window = output_window
        self.series_num = series_num
        self.feature_dim = feature_dim
        self.output_dim = output_dim
        self.device = device
        
        # TCN特征提取器
        self.tcn_feature_extractor = GrangerTCN(
            input_series_num=series_num,
            output_size=series_num,  # 输出维度等于序列数，保持对应关系
            TCN_hidden_channels=tcn_channels,
            kernel_size=kernel_size,
            dropout=dropout
        )
        self.feature_fusion = nn.Linear(series_num * input_window, series_num)
        
        # 时间维度池化层
        self.temporal_pooling = nn.AdaptiveAvgPool1d(1) 
        
        # 组件化MLP - 输入维度直接对应序列数
        self.cmlp = cMLP(
            input_dim=series_num,
            series_num=series_num,
            mlp_hidden=mlp_hidden,
            mlp_activation=mlp_activation,
            output_dim=1,
            kernel_size=1
        )
            
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
    
    def forward(self, x):
        if self.feature_dim == 1:
            x_input = x.squeeze(-1)
        else:
            x_input = x[..., 0]
        
        x_input = x_input.permute(0, 2, 1)  # [batch_size, series_num, input_window]
        
        # TCN特征提取 - 保持序列对应关系
        tcn_features = self.tcn_feature_extractor(x_input)  # [batch_size, series_num, input_window]
        
        # 时间维度池化，得到每个序列的代表性特征
        pooled_features = self.temporal_pooling(tcn_features).squeeze(-1)  # [batch_size, series_num]
        # fused_features = self.feature_fusion(tcn_features.flatten(1))
        
        # MLP预测 - 输入特征直接对应原始序列
        mlp_outputs = self.cmlp(pooled_features)  # [batch_size, series_num, 1]
        
        predictions = mlp_outputs.unsqueeze(1)
        
        return predictions
    
    def GC(self, threshold=True, ignore_kernel=True, weight_threshold=0.0):
        """
        计算Granger因果关系矩阵
        在这个版本中，MLP输入直接对应原始序列，因此权重分析是有意义的
        """
        return self.cmlp.GC(threshold=threshold, weight_threshold=weight_threshold, ignore_lag=ignore_kernel)

    def get_first_layer_params(self):
        """获取所有MLP第一层的参数"""
        return self.cmlp.get_first_layer_params()