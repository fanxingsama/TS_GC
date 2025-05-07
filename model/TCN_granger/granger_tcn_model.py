import torch
import torch.nn as nn
import torch.nn.functional as F
# 确保 granger_utils.py 在 Python 路径中或同一目录下
# 导入所需的惩罚计算函数 (如果需要在模型内部计算惩罚)
# from granger_utils import calculate_group_lasso_penalty, calculate_group_sparse_group_lasso_penalty

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
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        """
        Args:
            n_inputs (int): 输入通道数。
            n_outputs (int): 输出通道数。
            kernel_size (int): 卷积核大小。
            stride (int): 卷积步长。
            dilation (int): 卷积膨胀系数。
            padding (int): 卷积填充量。
            dropout (float): Dropout 比率。
        """
        super(TemporalBlock, self).__init__()
        # 第一个卷积层
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding) # 裁剪以确保因果性
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        # 第二个卷积层
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding) # 裁剪以确保因果性
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        # 包含两个卷积层的序列网络
        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2
        )

        # 如果输入输出通道数不同，使用 1x1 卷积调整残差连接
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        # 初始化权重（可选，但通常是好的实践）
        self.init_weights()

    def init_weights(self):
        """初始化卷积层的权重。"""
        # 对于 ReLU 激活函数，通常使用 He 初始化 (kaiming_normal_)
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_in', nonlinearity='relu')
        if self.conv1.bias is not None:
            nn.init.constant_(self.conv1.bias, 0)
        nn.init.kaiming_normal_(self.conv2.weight, mode='fan_in', nonlinearity='relu')
        if self.conv2.bias is not None:
            nn.init.constant_(self.conv2.bias, 0)

        # 如果存在，初始化下采样层
        if self.downsample is not None:
             nn.init.kaiming_normal_(self.downsample.weight, mode='fan_in', nonlinearity='relu')
             if self.downsample.bias is not None:
                nn.init.constant_(self.downsample.bias, 0)


    def forward(self, x):
        """
        Args:
            x (torch.Tensor): 输入张量，形状 [batch, channels, time]。
        Returns:
            torch.Tensor: TemporalBlock 的输出，形状 [batch, out_channels, time]。
        """
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

# Granger-TCN
class GrangerTCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels_list, kernel_size=3, dropout=0.2):
        """
        Args:
            input_size (int): 输入特征（时间序列）的数量。对应 Conv1d 的 in_channels。
            output_size (int): *注意：此参数在此修改版中不再直接用于最终预测，
                               而是表示最后一个 TCN 块的输出通道数。*
                               但为了与原始签名兼容，我们保留它，并确保它等于 num_channels_list[-1]。
            num_channels_list (list of int): 一个列表，其中每个元素指定了对应 TemporalBlock 的输出通道数。
                                             列表的长度决定了 TCN 的层数/块数。
            kernel_size (int): 卷积核大小。
            dropout (float): Dropout 比率。
        """
        super(GrangerTCN, self).__init__()

        # 验证 output_size 是否与最后一个通道数匹配
        if output_size != num_channels_list[-1]:
            print(f"警告: GrangerTCN 的 output_size ({output_size}) 与 num_channels_list 的最后一个元素 ({num_channels_list[-1]}) 不匹配。将使用 num_channels_list 的最后一个元素作为 TCN 的最终输出通道数。")
            # output_size = num_channels_list[-1] # 内部使用最后一个通道数

        self.network_layers = nn.ModuleList() # 使用 ModuleList 存储网络层
        num_levels = len(num_channels_list)

        # 构建 TCN 层
        for i in range(num_levels):
            dilation_size = 2 ** i # 膨胀系数指数增长
            # 确定当前层的输入通道数
            in_channels = input_size if i == 0 else num_channels_list[i-1]
            out_channels = num_channels_list[i]
            # 计算因果卷积所需的填充量
            padding = (kernel_size - 1) * dilation_size
            self.network_layers.append(
                TemporalBlock(in_channels, out_channels, kernel_size, stride=1,
                              dilation=dilation_size,
                              padding=padding,
                              dropout=dropout)
            )
    def get_first_block_conv1_weights(self):
        """
        返回第一个 TemporalBlock 中第一个卷积层 (conv1) 的权重张量。
        这是 Group Lasso/GSGL 惩罚的目标。
        形状: [out_channels_first_block, input_size, kernel_size]
        """
        # 检查网络层列表是否为空以及第一个块是否有 conv1 属性
        if len(self.network_layers) > 0 and hasattr(self.network_layers[0], 'conv1'):
            return self.network_layers[0].conv1.weight
        else:
            print("警告: 无法获取第一个 TCN 块的 conv1 权重。")
            return None

    def forward(self, x):
        """
        Args:
            x: 输入张量，形状为 [batch_size, input_size, sequence_length]。
               (注意：与原始示例不同，这里假设输入已经是 [batch, channels, time] 格式)
               或者如果输入是 [batch_size, sequence_length, input_size]，则需要先转置。
        Returns:
            输出张量，是最后一个 TCN 块的完整序列输出。
            形状为 [batch_size, num_channels_last_block, sequence_length]。
        """
        # 将输入传递给所有 TCN 块
        for layer in self.network_layers:
            x = layer(x)

        return x
        # --- 结束修改 ---
