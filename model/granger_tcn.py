import torch
import torch.nn as nn
import torch.nn.functional as F

# 裁剪掉填充的长度
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        """
            chomp_size (int): 需要从序列末尾裁剪掉的元素数量。
        """
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): 输入张量，形状 [batch, channels, time]。
        Returns:
            torch.Tensor: 裁剪后的张量。
        """
        return x[:, :, :-self.chomp_size].contiguous()

# TCN 的基本模块，包含两个膨胀因果卷积层和一个残差连接。
class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout):
        """
        Args:
            n_inputs (int): 输入通道数。
            n_outputs (int): 卷积后想从数据中得到多少个不同的特征。
            kernel_size (int): 卷积核大小。
            stride (int): 卷积步长。
            dilation (int): 卷积膨胀系数。
            padding (int): 卷积填充量。
            dropout (float): Dropout 比率。
        """
        super(TemporalBlock, self).__init__()
        # w形状：[out_ch, in_ch, kernel_size]
        # n_outputs也代表了用多少个卷积核进行处理，得到多少维矩阵
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation) 
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.PReLU()
        self.dropout1 = nn.Dropout(dropout)  

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.PReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2
        )

        # 使用 1x1 卷积调整残差连接
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.PReLU()
        self.init_weights() 
    # He初始化
    def init_weights(self):
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_in', nonlinearity='leaky_relu')
        if self.conv1.bias is not None:
            nn.init.constant_(self.conv1.bias, 0)
        nn.init.kaiming_normal_(self.conv2.weight, mode='fan_in', nonlinearity='leaky_relu')
        if self.conv2.bias is not None:
            nn.init.constant_(self.conv2.bias, 0)

        if self.downsample is not None:
            nn.init.kaiming_normal_(self.downsample.weight, mode='fan_in', nonlinearity='leaky_relu')
            if self.downsample.bias is not None:
                nn.init.constant_(self.downsample.bias, 0)

    def forward(self, x):
        out = self.net(x) # x: [batch_size, series_num, sequence_length]
        '''
        输入通道数和输出通道数相同时，直接使用输入张量 x 作为残差连接。
        输入通道数和输出通道数不同时，通过 1x1 卷积层对输入张量进行通道调整，以确保残差连接的输出张量的通道数与卷积层的输出通道数一致。
        '''
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res) # [batch_size, n_outputs, sequence_length]

class GrangerTCN(nn.Module):
    def __init__(self, input_series_num, output_size, TCN_hidden_channels, kernel_size, dropout):
        super(GrangerTCN, self).__init__()
        self.network_layers = nn.ModuleList()
        
        # 第一个 TemporalBlock
        dilation_size = 1 
        padding = (kernel_size - 1) * dilation_size
        self.network_layers.append(
            TemporalBlock(input_series_num, 
                          TCN_hidden_channels, 
                          kernel_size, 
                          stride=1,
                          dilation=dilation_size,
                          padding=padding,
                          dropout=dropout)
        )
        
        # 第二个 TemporalBlock
        dilation_size = 2 
        padding = (kernel_size - 1) * dilation_size
        self.network_layers.append(
            TemporalBlock(TCN_hidden_channels, 
                          output_size, 
                          kernel_size, 
                          stride=1,
                          dilation=dilation_size,
                          padding=padding,
                          dropout=dropout)
        )
    def get_first_block_conv1_weights(self):
        return self.network_layers[0].conv1.weight

    def forward(self, x):
        # x：[batch_size, series_num, input_window]
        for layer in self.network_layers:
            x = layer(x)
        # 输出为每个时刻该序列的output_size个特征的数据
        return x # [batch_size, series_num, input_window]
    

