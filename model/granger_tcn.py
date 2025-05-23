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
        # 第一个卷积层
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation) # w形状：[out_ch, in_ch, kernel_size]，相当于一个卷积核同时对in_ch条序列进行卷积
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.PReLU()
        self.dropout1 = nn.Dropout(dropout)

        # 第二个卷积层
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.PReLU()
        self.dropout2 = nn.Dropout(dropout)

        # 包含两个卷积层的序列网络
        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2
        )

        # 如果输入输出通道数不同，使用 1x1 卷积调整残差连接
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.PReLU()
        # 初始化权重（可选，但通常是好的实践）
        self.init_weights() 

    # 稀疏初始化
    # def init_weights(self):
    #     # 对第一个卷积层进行稀疏初始化
    #     weight = self.conv1.weight
    #     weight = weight.view(weight.shape[0], -1)  # 展平为二维
    #     nn.init.sparse_(weight, sparsity=0.9, std=0.01)
    #     weight = weight.view_as(self.conv1.weight)  # 重塑回原来的形状
    #     self.conv1.weight.data = weight
    #     if self.conv1.bias is not None:
    #         nn.init.constant_(self.conv1.bias, 0)

    #     # 对第二个卷积层进行稀疏初始化
    #     weight = self.conv2.weight
    #     weight = weight.view(weight.shape[0], -1)
    #     nn.init.sparse_(weight, sparsity=0.9, std=0.01)
    #     weight = weight.view_as(self.conv2.weight)
    #     self.conv2.weight.data = weight
    #     if self.conv2.bias is not None:
    #         nn.init.constant_(self.conv2.bias, 0)

    #     # 如果有下采样层，同样进行稀疏初始化
    #     if self.downsample is not None:
    #         weight = self.downsample.weight
    #         weight = weight.view(weight.shape[0], -1)
    #         nn.init.sparse_(weight, sparsity=0.9, std=0.01)
    #         weight = weight.view_as(self.downsample.weight)
    #         self.downsample.weight.data = weight
    #         if self.downsample.bias is not None:
    #             nn.init.constant_(self.downsample.bias, 0)
    
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
        """
        Args:
            input_size (int): 输入的时序序列的数量，代表除了目标时间序列外的其他序列数量 (n-1)。
            output_size (int): 输出通道数，对应最终的输出维度。
            TCN_hidden_channels (int or list): 每个 TemporalBlock 的输出通道数，如果有超过2个卷积块，就应该是个list，[tcn_channels, tcn_channels, ...]，表示每一层卷积块的输出维度和下一层的输入维度。
            若 TCN_hidden_channels 设计为递增（如 [32, 64, 128]），深层块能捕捉更复杂的模式，但需通过跳跃连接中的 1x1 卷积对齐输入输出维度。这会增加参数量和计算成本，但可能提升模型表达能力。
            
            kernel_size (int): 卷积核大小。
            dropout (float): Dropout 比率。
        """
        super(GrangerTCN, self).__init__()
        self.network_layers = nn.ModuleList()
        
        # 第一个 TemporalBlock
        dilation_size = 1  # 第一个block的膨胀系数为1
        padding = (kernel_size - 1) * dilation_size
        self.network_layers.append(
            TemporalBlock(input_series_num, TCN_hidden_channels, kernel_size, stride=1,
                          dilation=dilation_size,
                          padding=padding,
                          dropout=dropout)
        )
        
        # 第二个 TemporalBlock
        dilation_size = 2  # 第二个block的膨胀系数为2
        padding = (kernel_size - 1) * dilation_size
        self.network_layers.append(
            # 通过隐藏通道数个特征，提取到output_size个特征
            TemporalBlock(TCN_hidden_channels, output_size, kernel_size, stride=1,
                          dilation=dilation_size,
                          padding=padding,
                          dropout=dropout)
        )
    def get_first_block_conv1_weights(self):
        return self.network_layers[0].conv1.weight

    def forward(self, x):
        # x：[batch_size, series_num, sequence_length]
        for layer in self.network_layers:
            x = layer(x)
        return x  # [batch_size, output_feature, sequence_length]
    
    
# 对神经网络的第一层权重矩阵进行近端更新，作用于函数，直接对函数的参数进行稀疏性约束，使得某些参数被设置为零，从而实现稀疏性。
def PGD_update(network, lam, lr, penalty):
    hidden, p, lag = network.shape
    if penalty == 'GL': # 组Loss惩罚
        norm = torch.norm(network, dim=(0, 2), keepdim=True)
        network.data = ((network / torch.clamp(norm, min=(lr * lam)))
                  * torch.clamp(norm - (lr * lam), min=0.0))
    elif penalty == 'GSGL': # 组稀疏组Lasso惩罚
        norm = torch.norm(network, dim=0, keepdim=True)
        network.data = ((network / torch.clamp(norm, min=(lr * lam)))
                  * torch.clamp(norm - (lr * lam), min=0.0))
        norm = torch.norm(network, dim=(0, 2), keepdim=True)
        network.data = ((network / torch.clamp(norm, min=(lr * lam)))
                  * torch.clamp(norm - (lr * lam), min=0.0))
    elif penalty == 'H': # 层次Lasso惩罚
        for i in range(lag):
            norm = torch.norm(network[:, :, :(i + 1)], dim=(0, 2), keepdim=True)
            network.data[:, :, :(i+1)] = (
                (network.data[:, :, :(i+1)] / torch.clamp(norm, min=(lr * lam)))
                * torch.clamp(norm - (lr * lam), min=0.0))
    else:
        raise ValueError('unsupported penalty: %s' % penalty)

# 稀疏惩罚的结果
def lasso_penalty(network, lam, penalty):
    hidden, p, lag = network.shape
    if penalty == 'GL': # 组Loss惩罚
        return lam * torch.sum(torch.norm(network, dim=(0, 2)))
    elif penalty == 'GSGL': # 组稀疏组Lasso惩罚
        return lam * (torch.sum(torch.norm(network, dim=(0, 2)))
                      + torch.sum(torch.norm(network, dim=0)))
    elif penalty == 'H': # 层次Lasso惩罚
        return lam * sum([torch.sum(torch.norm(network[:, :, :(i+1)], dim=(0, 2)))
                          for i in range(lag)])
    else:
        raise ValueError('unsupported penalty: %s' % penalty)

