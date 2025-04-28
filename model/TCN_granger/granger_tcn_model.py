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

# Granger-TCN (修改后用于集成)
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

        # --- 移除或注释掉原始的最终线性层 ---
        # 因为最终的预测将在外部的 PredictModel 中完成。
        # self.linear = nn.Linear(num_channels_list[-1], output_size)
        # nn.init.xavier_uniform_(self.linear.weight)
        # if self.linear.bias is not None:
        #     nn.init.constant_(self.linear.bias, 0)
        # --- 结束移除 ---

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
        # --- 输入形状检查与调整 (如果需要) ---
        # 假设调用者已经将数据调整为 [batch_size, input_size, sequence_length]
        # if x.shape[1] != self.network_layers[0].conv1.in_channels:
        #     # 如果输入是 [batch, seq_len, features], 转置
        #     if x.shape[2] == self.network_layers[0].conv1.in_channels:
        #          x = x.transpose(1, 2)
        #     else:
        #         raise ValueError(f"输入形状 {x.shape} 与 TCN 期望的输入通道数 {self.network_layers[0].conv1.in_channels} 不符。")
        # --- 结束调整 ---


        # 将输入传递给所有 TCN 块
        for layer in self.network_layers:
            x = layer(x)

        # --- 返回最后一个 TCN 块的输出 (完整序列) ---
        # 不应用原始的 self.linear 或只取最后一个时间步
        return x
        # --- 结束修改 ---

# --- 示例用法 (展示修改后的行为) ---
if __name__ == '__main__':
    # 超参数
    batch_size = 16
    sequence_length = 50
    input_features = 10  # 输入时间序列的数量 (TCN 的 input_size)
    # output_size 现在表示最后一个 TCN 块的通道数
    tcn_channels = [32, 64, 64] # TCN 各层通道数
    final_tcn_channels = tcn_channels[-1] # 最后一个块的输出通道
    kernel_size = 3
    dropout = 0.1

    # 创建模拟输入数据 [batch, channels, time]
    dummy_input = torch.randn(batch_size, input_features, sequence_length)

    # 实例化修改后的模型
    # output_size 参数应等于最后一个通道数
    model = GrangerTCN(input_features, final_tcn_channels, tcn_channels, kernel_size, dropout)

    model.eval() # 设置为评估模式以禁用 dropout
    with torch.no_grad():
        tcn_output = model(dummy_input)

    print(f"修改后的 GrangerTCN 输出形状: {tcn_output.shape}")
    # 期望形状: [batch_size, final_tcn_channels, sequence_length]
    # 例如: [16, 64, 50]
    expected_shape = (batch_size, final_tcn_channels, sequence_length)
    print(f"期望形状: {expected_shape}")
    assert tcn_output.shape == expected_shape, "输出形状不符合预期！"

    # 检查是否能获取权重
    weights = model.get_first_block_conv1_weights()
    if weights is not None:
        print(f"第一个块 conv1 权重的形状: {weights.shape}")
        # 期望形状: [tcn_channels[0], input_features, kernel_size]
        # 例如: [32, 10, 3]
        expected_weight_shape = (tcn_channels[0], input_features, kernel_size)
        print(f"期望权重形状: {expected_weight_shape}")
        assert weights.shape == expected_weight_shape, "权重形状不符合预期！"
    else:
        print("未能获取权重。")
