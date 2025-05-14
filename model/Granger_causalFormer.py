import torch
import torch.nn as nn
import math
from .TCN_granger.granger_tcn_model import GrangerTCN


Linear = nn.Linear
LayerNorm = nn.LayerNorm
Dropout = nn.Dropout
Softmax = nn.Softmax
LeakyReLU = nn.LeakyReLU

# 把多变量序列的数据进行嵌入
class Embedding(nn.Module):
    """
    参数：
        series_num (int): 输入中的时间序列数量。
        input_window (int): 输入时间序列窗口的长度。
        feature_dim (int): 时间序列中每个特征的维度。
        d_model (int): 嵌入向量的维度。在论文中表示为 D_QK。
        drop_prob (float): 用于正则化的 Dropout 概率。
        device (torch.device): 计算设备（'cpu' 或 'cuda'）。
    """
    def __init__(self, series_num, input_window, feature_dim, d_model, drop_prob, device):
        super().__init__()
        self.series_num = series_num
        self.input_window = input_window
        self.feature_dim = feature_dim
        self.d_model = d_model
        self.drop_prob = drop_prob
        self.device = device
        self.feature_emb = Linear(in_features=self.input_window * self.feature_dim, out_features=self.d_model, bias=True)
        self.feature_emb.weight.data.normal_(0, math.sqrt(2.0 / (self.input_window * self.feature_dim + self.d_model)))
        if self.feature_emb.bias is not None:
             nn.init.constant_(self.feature_emb.bias, 0)
        self.norm = LayerNorm(self.d_model)
        self.drop_out = Dropout(self.drop_prob)

    def forward(self, x):
        # 输入 x: [batch_size, series_num, input_window, feature_dim]
        batch_size = x.size(0)
        x_flat = x.view(batch_size, self.series_num, -1) # 展平，[batch_size, series_num, input_window * feature_dim]
        embedding = self.feature_emb(x_flat)
        embedding = self.norm(embedding)
        embedding = self.drop_out(embedding)
        return embedding
        # [batch_size, series_num, d_model]

# 多变量因果注意力，实现注意力机制 计算核心 的模块。它接收已经准备好的Q/K/V，之后进行注意力计算，输出是注意力机制权重
class MultiVariateCausalAttention(nn.Module):
    """
    Args:
        d_tensor (int): 每个注意力头的维度 (d_model / n_head)。
        tau (float): softmax 的温度超参数。
    """
    def __init__(self, d_tensor, tau):
        super().__init__()
        self.d_tensor = d_tensor
        self.tau = tau
        self.softmax = Softmax(dim=-1)

    def forward(self, q, k, v):
        """
            q/k:[batch_size, head, series_num, d_tensor]。
            v (torch.Tensor): 值张量 [batch_size, head, series_num, input_window, d_tensor]。
        Returns:
            out:[batch_size, head, series_num, input_window, d_tensor]。
            attn_weights: [batch_size, head, series_num, series_num]。
        """
        batch_size, n_head, series_num, d_tensor = q.shape
        _batch_size, _n_head, _series_num, input_window, d_tensor_v = v.shape
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_tensor)
        attn_weights = self.softmax(scores / self.tau)

        # v_reshaped = v.view(batch_size, n_head, series_num, -1)
        v_reshaped = v.reshape(batch_size, n_head, series_num, -1)

        out_reshaped = torch.matmul(attn_weights, v_reshaped)

        # out = out_reshaped.view(batch_size, n_head, series_num, input_window, feature_dim_v)
        out = out_reshaped.reshape(batch_size, n_head, series_num, input_window, d_tensor_v) # Use reshape

        return out, attn_weights

# 多头注意力，内部实现了多变量因果注意力，这里使用了TCN
class MultiHeadAttention(nn.Module):
    """
    Args:
        series_num (int): 输入中的时间序列数量。
        input_window (int): 输入时间序列窗口的长度。
        feature_dim (int): 输入时间序列中每个特征的维度。
        d_model (int): 嵌入向量的维度 (D_QK)。
        n_head (int): 注意力头的数量 (h)。
        tcn_channels (list): GrangerTCN 的通道列表。
        tcn_kernel_size (int): GrangerTCN 的核大小。
        tcn_dropout (float): GrangerTCN 的 dropout。
        tau (float): 注意力 softmax 的温度超参数。
        device (torch.device): 计算设备。
    """
    def __init__(self, series_num, input_window, feature_dim, d_model, n_head,
                 tcn_channels, tcn_kernel_size, tcn_dropout,
                 tau, device):
        super().__init__()
        self.series_num = series_num
        self.input_window = input_window
        self.feature_dim = feature_dim
        self.d_model = d_model
        self.n_head = n_head
        self.d_tensor = d_model // n_head
        self.tau = tau
        self.device = device

        self.Wq = Linear(in_features=self.d_model, out_features=self.d_model, bias=True) # 将输入特征投影到不同的表示空间，加入线性之后方便训练W矩阵
        self.Wk = Linear(in_features=self.d_model, out_features=self.d_model, bias=True) 
        self.Wv = Linear(in_features=tcn_channels, out_features=self.d_model) # 将TCN的输出output_size投影到与d_model相同的维度，前提是tcn_channels为一个整数，而不是列表
        self.w_concat = Linear(in_features=self.d_model, out_features=self.feature_dim, bias=False) # 将多头注意力的输出从d_model变回原始维度feature_dim
        self.Wq.weight.data.normal_(0, math.sqrt(2.0 / (self.d_model + self.d_model))) # He 初始化
        self.Wk.weight.data.normal_(0, math.sqrt(2.0 / (self.d_model + self.d_model))) 
        self.Wv.weight.data.normal_(0, math.sqrt(2.0 / (tcn_channels + self.d_model)))
        self.w_concat.weight.data.normal_(0, math.sqrt(2.0 / (self.d_model + self.feature_dim)))

        # 为每个时间序列创建单独的TCN
        self.tcn_processors = nn.ModuleList()
        for i in range(self.series_num):
            self.tcn_processors.append(
                GrangerTCN(input_series_num=self.series_num - 1, # 每个TCN的输入是除了目标序列之外的所有序列 (series_num - 1)
                          output_size=tcn_channels, # TCN所提取到的特征数
                          TCN_hidden_channels=tcn_channels, # TCN中间层的通道数
                          kernel_size=tcn_kernel_size, 
                          dropout=tcn_dropout)
            )

        # 构建注意力对象
        self.count_attention = MultiVariateCausalAttention(self.d_tensor, self.tau)

    # 将输入张量拆分为多个头，以便进行多头注意力计算
    def split(self, tensor):
        batch_size = tensor.size(0)
        if tensor.ndim == 3: # 3个维度的情况，针对Q, K矩阵，[batch_size, series_num, d_model]
            length, d_model = tensor.size(1), tensor.size(2)
            d_tensor = d_model // self.n_head
            tensor = tensor.view(batch_size, length, self.n_head, d_tensor).transpose(1, 2).contiguous() # Add contiguous()
        elif tensor.ndim == 4: # 4个维度的情况，针对V矩阵: [batch_size, series_num, series_len, d_model]
            n_series, seq_len, f_v = tensor.size(1), tensor.size(2), tensor.size(3)
            fv_head = f_v // self.n_head
            tensor = tensor.view(batch_size, n_series, seq_len, self.n_head, fv_head).permute(0, 3, 1, 2, 4).contiguous() # Add contiguous()
        else:
            raise ValueError(f"Unsupported tensor ndim for split: {tensor.ndim}")
        return tensor

    # 将多个头的输出张量连接在一起，以便进行后续处理
    def concat(self, tensor):
        batch_size, head, n_series, seq_len, fv_head = tensor.size()
        f_v = head * fv_head
        tensor = tensor.permute(0, 2, 3, 1, 4).contiguous().view(batch_size, n_series, seq_len, f_v) # Add contiguous() before view
        return tensor

    def forward(self, q_emb, k_emb, x_input):
        """
        Args:
            q_emb (torch.Tensor): 查询嵌入 [batch_size, series_num, d_model]
            k_emb (torch.Tensor): 键嵌入 [batch_size, series_num, d_model]
            x_input (torch.Tensor): 时间序列 [batch_size, series_num, input_window, feature_dim]
        Returns:
            torch.Tensor: 多头注意力的输出 [batch_size, series_num, input_window, feature_dim]。
        """
        q, k = self.Wq(q_emb), self.Wk(k_emb) # Q和K的线性变换

        # 如果特征维度大于1，则只取第一个特征维度作为 TCN 的输入，如果特征维度为1，则直接去掉最后一个维度
        x_tcn_input = x_input.squeeze(-1) if self.feature_dim == 1 else x_input[..., 0] # 去掉feature_dim维度，最终的输入维度：[batch_size, series_num, input_window]

        # 对于每个时间序列，使用单独的TCN处理除了目标序列之外的所有序列。
        all_tcn_outputs = []
        for i in range(self.series_num):
            # 将所有除了目标序列之外的序列作为输入
            mask = torch.ones(self.series_num, dtype=torch.bool, device=self.device) 
            mask[i] = False 
            other_series = x_tcn_input[:, mask, :]
            
            tcn_out = self.tcn_processors[i](other_series)  # tcn_out:[batch_size, output_size, series_len]
            all_tcn_outputs.append(tcn_out)
        
        # 堆叠所有TCN输出
        stacked_outputs = torch.stack(all_tcn_outputs, dim=1)  # stacked_outputs:[batch_size, series_num, output_size, series_len]
        tcn_output = stacked_outputs.permute(0, 3, 1, 2)  # 调整后堆叠的输出，方便线性变换，[batch_size, series_len, series_num, output_size]
        
        # 使用线性层 Wv 将 TCN 的输出投影到与 d_model 相同的维度。
        v_temp = self.Wv(tcn_output)  # [batch_size, series_len, series_num, d_model]
        v = v_temp.permute(0, 2, 1, 3)  # 维度顺序调整，[batch_size, series_num, series_len, d_model]

        q, k, v = self.split(q), self.split(k), self.split(v) # 将Q、K、V拆分为多头，以便进行多头注意力计算

        out, attn_weights = self.count_attention(q, k, v) # 计算多头注意力
        out = self.concat(out) # 将多头注意力的输出拼接回原始维度，[batch_size, series_num, input_window, d_model]
        out = self.w_concat(out) # 使用线性层将输出投影到特征维度，[batch_size, series_num, input_window, feature_dim]

        return out # [batch_size, series_num, input_window, feature_dim]

# 位置前馈层
class PositionwiseFeedForward(nn.Module):
    """
    Args:
        dim (int): 输入和输出维度。
        hidden (int): 中间隐藏层维度 (d_FFN)。
        drop_prob (float): Dropout 概率。
    """
    def __init__(self, dim, hidden, drop_prob=0.1):
        super().__init__()
        self.linear1 = Linear(dim, hidden, bias=True)
        self.linear2 = Linear(hidden, dim, bias=True)
        self.activation = LeakyReLU()
        self.dropout = Dropout(drop_prob)
        self.linear1.weight.data.normal_(0, math.sqrt(2.0 / (dim + hidden)))
        self.linear2.weight.data.normal_(0, math.sqrt(2.0 / (hidden + dim)))

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

# 编码器层
class EncoderLayer(nn.Module):
    """
    Args:
        series_num (int): 时间序列数量。
        input_window (int): 输入窗口长度。
        feature_dim (int): 输入特征维度。
        d_model (int): QK 嵌入维度。
        n_head (int): 注意力头数。
        tcn_channels (list): GrangerTCN 通道列表。
        tcn_kernel_size (int): GrangerTCN 核大小。
        tcn_dropout (float): GrangerTCN dropout。
        ffn_hidden (int): FFN 隐藏层维度。
        drop_prob (float): Dropout 概率。
        tau (float): 注意力 softmax 温度。
        device (torch.device): 计算设备。
    """
    def __init__(self, series_num, input_window, feature_dim, d_model, n_head,
                 tcn_channels, tcn_kernel_size, tcn_dropout,
                 ffn_hidden, drop_prob, tau, device):
        super().__init__()
        self.attention = MultiHeadAttention(series_num, input_window, feature_dim, d_model, n_head,
                                            tcn_channels, tcn_kernel_size, tcn_dropout,
                                            tau, device)
        
        self.norm1 = LayerNorm([input_window, feature_dim])
        self.dropout1 = Dropout(drop_prob)
        self.ffn = PositionwiseFeedForward(dim=feature_dim, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm2 = LayerNorm([input_window, feature_dim])
        self.dropout2 = Dropout(drop_prob)

    def forward(self, x_embedding, x):
        """
        Args:
            x_embedding (torch.Tensor): 输入嵌入 [batch_size, series_num, d_model]
            x (torch.Tensor): [batch_size, series_num, input_window, feature_dim]
        Returns:
            torch.Tensor: 编码器层输出 [batch_size, series_num, input_window, feature_dim]。
        """
        attn_output = self.attention(q_emb=x_embedding, k_emb=x_embedding, x_input=x)
        x_res = x + self.dropout1(attn_output)
        x_norm1 = self.norm1(x_res)
        ffn_output = self.ffn(x_norm1)
        x_res2 = x_norm1 + self.dropout2(ffn_output)
        x_norm2 = self.norm2(x_res2)
        return x_norm2

# 编码器
class Encoder(nn.Module):
    """
    Args:
        series_num, input_window, feature_dim, d_model, n_head: 见 EncoderLayer。
        tcn_channels, tcn_kernel_size, tcn_dropout: GrangerTCN 参数。
        n_layers (int): 编码器层数。
        ffn_hidden, drop_prob, tau, device: 见 EncoderLayer。
    """
    def __init__(self, series_num, input_window, feature_dim, d_model, n_head,
                 tcn_channels, tcn_kernel_size, tcn_dropout,
                 n_layers, ffn_hidden, drop_prob, tau, device):
        super().__init__()
        self.emb = Embedding(series_num=series_num,
                             input_window=input_window,
                             feature_dim=feature_dim,
                             d_model=d_model,
                             drop_prob=drop_prob,
                             device=device)

        self.layers = nn.ModuleList([EncoderLayer(series_num=series_num,
                                                  input_window=input_window,
                                                  feature_dim=feature_dim,
                                                  d_model=d_model,
                                                  n_head=n_head,
                                                  tcn_channels=tcn_channels,
                                                  tcn_kernel_size=tcn_kernel_size,
                                                  tcn_dropout=tcn_dropout,
                                                  ffn_hidden=ffn_hidden,
                                                  drop_prob=drop_prob,
                                                  tau=tau,
                                                  device=device)
                                     for _ in range(n_layers)])

    def forward(self, x):
        # x: [batch_size, series_num, input_window, feature_dim]
        embedding = self.emb(x) # [batch_size, series_num, d_model]
        out = x
        for layer in self.layers:
            out = layer(embedding, out)
        return out
        # 输出 x: [batch_size, series_num, input_window, feature_dim]

# 最终预测模型
class PredictModel(nn.Module):
    """
    Args:
        config (dict): 配置字典。
        d_model (int): QK 嵌入维度。
        n_head (int): 注意力头数。
        tcn_channels (list): GrangerTCN 通道列表。
        tcn_kernel_size (int): GrangerTCN 核大小。
        tcn_dropout (float): GrangerTCN dropout。
        n_layers (int): 编码器层数
        ffn_hidden (int): FFN 隐藏层维度
        drop_prob (float): Dropout 概率
        tau (float): 注意力 softmax 温度
    """
    def __init__(self, config, d_model, n_head, tcn_channels, tcn_kernel_size, tcn_dropout, n_layers, ffn_hidden, drop_prob, tau):
        super().__init__()
        self.config = config
        self.data_feature = config['data_loader']['args']
        self.input_window = self.data_feature.get('input_window')
        self.output_window = self.data_feature.get('output_window')
        self.series_num = self.data_feature.get('series_num')
        self.feature_dim = self.data_feature.get('feature_dim')
        self.output_dim = self.data_feature.get('output_dim')

        device_str = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device(device_str)

        self.encoder = Encoder(series_num=self.series_num,
                               input_window=self.input_window,
                               feature_dim=self.feature_dim,
                               d_model=d_model,
                               n_head=n_head,
                               tcn_channels=tcn_channels,
                               tcn_kernel_size=tcn_kernel_size,
                               tcn_dropout=tcn_dropout,
                               n_layers=n_layers,
                               ffn_hidden=ffn_hidden,
                               drop_prob=drop_prob,
                               tau=tau,
                               device=self.device)

        self.fc = Linear(in_features=self.feature_dim, out_features=self.output_dim, bias=True)
        self.fc.weight.data.normal_(0, math.sqrt(2.0 / (self.feature_dim + self.output_dim)))
        if self.fc.bias is not None:
            nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        # x = [batch_size, input_window, series_num, feature]
        x = x.permute(0, 2, 1, 3)  # 维度交换位置，适配其他层所需的形状，[batch_size, series_num, input_window, feature_dim]
        encoder_out = self.encoder(x) # [batch_size, series_num, input_window, feature_dim]
        last_step_out = encoder_out[:, :, -1, :] # 提取input_window上最后一个时间步的输出，用于预测未来。[batch_size, series_num, feature_dim]
        out = self.fc(last_step_out) # 全连接，将 feature_dim 的特征向量映射到输出值维度 output_dim（这里是1，表示单步预测单个值）。[batch_size, series_num, output_dim]
        out = out.unsqueeze(1) # [batch_size, 1, series_num, output_dim]
        if self.output_window > 1:
            out = out.repeat(1, self.output_window, 1, 1)
        return out # [batch_size, output_window, series_num, output_dim]
    
    def GC(self, threshold=True, ignore_kernel=True):
        gc_weights = []
        
        for i in range(self.series_num):
            weights = self.encoder.layers[0].attention.tcn_processors[i].get_first_block_conv1_weights() # [output_channels, input_channels, kernel_size]
            full_weights = torch.zeros(self.series_num, *weights.shape[1:], device=weights.device) # 目标序列的完整因果矩阵
            
            # 填充除了目标序列之外的所有序列的权重
            mask = torch.ones(self.series_num, dtype=torch.bool, device=weights.device)
            mask[i] = False # 排除目标序列
            full_weights[mask] = weights.reshape(-1, *weights.shape[1:]) # 使用 mask 将权重 weights 填充到 full_weights 中，排除目标序列自身。
            if ignore_kernel:
                # 计算跨输出通道和卷积核大小维度的范数
                series_gc = torch.norm(full_weights, dim=(1, 2))
            else:
                # 计算仅跨输出通道维度的范数
                series_gc = torch.norm(full_weights, dim=1)
            
            gc_weights.append(series_gc)

        gc_matrix = torch.stack(gc_weights) # GC 权重堆叠起来
        
        if threshold:
            return (gc_matrix > 0).int()
        else:
            return gc_matrix # [series_num, series_num]，如果ignore_kernel=False，形状为 [series_num, series_num, kernel_size]