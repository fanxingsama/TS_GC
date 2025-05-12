import torch
import torch.nn as nn
import math
from .TCN_granger.granger_tcn_model import GrangerTCN


Linear = nn.Linear
LayerNorm = nn.LayerNorm
Dropout = nn.Dropout
Softmax = nn.Softmax
LeakyReLU = nn.LeakyReLU


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
        x_flat = x.view(batch_size, self.series_num, -1)
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
        Args:
            q (torch.Tensor): 查询张量 [batch_size, head, series_num, d_tensor]。
            k (torch.Tensor): 键张量 [batch_size, head, series_num, d_tensor]。
            v (torch.Tensor): 值张量 [batch_size, head, series_num, input_window, feature_dim_v]。
        Returns:
            torch.Tensor: 注意力输出 [batch_size, head, series_num, input_window, feature_dim_v]。
            torch.Tensor: 注意力权重 [batch_size, head, series_num, series_num]。
        """
        batch_size, n_head, series_num, d_tensor = q.shape
        _batch_size, _n_head, _series_num, input_window, feature_dim_v = v.shape
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_tensor)
        attn_weights = self.softmax(scores / self.tau)

        # *** FIX: Replace view with reshape for non-contiguous tensor ***
        # v_reshaped = v.view(batch_size, n_head, series_num, -1)
        v_reshaped = v.reshape(batch_size, n_head, series_num, -1)

        out_reshaped = torch.matmul(attn_weights, v_reshaped)

        # *** FIX: Replace view with reshape for non-contiguous tensor ***
        # out = out_reshaped.view(batch_size, n_head, series_num, input_window, feature_dim_v)
        out = out_reshaped.reshape(batch_size, n_head, series_num, input_window, feature_dim_v) # Use reshape

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

        self.Wq = Linear(in_features=self.d_model, out_features=self.d_model, bias=True)
        self.Wk = Linear(in_features=self.d_model, out_features=self.d_model, bias=True)
        self.Wq.weight.data.normal_(0, math.sqrt(2.0 / (self.d_model + self.d_model)))
        self.Wk.weight.data.normal_(0, math.sqrt(2.0 / (self.d_model + self.d_model)))

        # V的处理，为每个时间序列创建单独的TCN
        self.tcn_processors = nn.ModuleList()
        for i in range(self.series_num):
            # 每个TCN的输入是除了目标序列之外的所有序列 (series_num - 1)
            tcn_input_size = self.series_num - 1
            tcn_output_size = tcn_channels  # 使用相同的输出通道数
            
            self.tcn_processors.append(
                GrangerTCN(input_size=tcn_input_size,
                          output_size=tcn_output_size,
                          num_channels=tcn_channels,
                          kernel_size=tcn_kernel_size,
                          dropout=tcn_dropout)
            )

        self.v_feature_dim = self.d_model
        self.Wv_proj = Linear(tcn_channels, self.v_feature_dim)
        self.Wv_proj.weight.data.normal_(0, math.sqrt(2.0 / (tcn_channels + self.v_feature_dim)))
        
        # 构建注意力对象
        self.attention = MultiVariateCausalAttention(self.d_tensor, self.tau)

        self.w_concat = Linear(in_features=self.v_feature_dim, out_features=self.feature_dim, bias=False)
        self.w_concat.weight.data.normal_(0, math.sqrt(2.0 / (self.v_feature_dim + self.feature_dim)))

    def split(self, tensor):
        batch_size = tensor.size(0)
        if tensor.ndim == 3: # Q, K: [B, N, D]
            length, d_model = tensor.size(1), tensor.size(2)
            d_tensor = d_model // self.n_head
            # Use reshape instead of view after transpose for safety
            tensor = tensor.view(batch_size, length, self.n_head, d_tensor).transpose(1, 2).contiguous() # Add contiguous()
        elif tensor.ndim == 4: # V: [B, N, T, Fv]
            n_series, seq_len, f_v = tensor.size(1), tensor.size(2), tensor.size(3)
            fv_head = f_v // self.n_head
            # Use reshape instead of view after permute for safety
            tensor = tensor.view(batch_size, n_series, seq_len, self.n_head, fv_head).permute(0, 3, 1, 2, 4).contiguous() # Add contiguous()
        else:
            raise ValueError(f"Unsupported tensor ndim for split: {tensor.ndim}")
        return tensor

    def concat(self, tensor):
        batch_size, head, n_series, seq_len, fv_head = tensor.size()
        f_v = head * fv_head
        # Use reshape instead of view after permute for safety
        tensor = tensor.permute(0, 2, 3, 1, 4).contiguous().view(batch_size, n_series, seq_len, f_v) # Add contiguous() before view
        return tensor

    def forward(self, q_emb, k_emb, x_input):
        """
        Args:
            q_emb (torch.Tensor): 查询嵌入 [batch_size, series_num, d_model]。
            k_emb (torch.Tensor): 键嵌入 [batch_size, series_num, d_model]。
            x_input (torch.Tensor): 原始输入时间序列 [batch_size, series_num, input_window, feature_dim]。
        Returns:
            torch.Tensor: 多头注意力的输出 [batch_size, series_num, input_window, feature_dim]。
        """
        batch_size = x_input.size(0)
        q, k = self.Wq(q_emb), self.Wk(k_emb)

        # --- V 处理：为每个时间序列使用单独的TCN ---
        if self.feature_dim > 1:
            x_tcn_input = x_input[:, :, :, 0]  # [B, N, T]
        else:
            x_tcn_input = x_input.squeeze(-1)  # [B, N, T]

        # 存储每个序列的TCN输出
        all_tcn_outputs = []
        
        for i in range(self.series_num):
            # 将所有除了目标序列之外的序列作为输入
            mask = torch.ones(self.series_num, dtype=torch.bool, device=self.device)
            mask[i] = False  # 排除目标序列
            
            # 提取除了目标序列外的所有序列
            other_series = x_tcn_input[:, mask, :]  # [B, N-1, T]
            
            # 将序列输入到当前TCN
            tcn_out = self.tcn_processors[i](other_series)  # [B, output_size, T]
            all_tcn_outputs.append(tcn_out)
        
        # 堆叠所有TCN输出
        stacked_outputs = torch.stack(all_tcn_outputs, dim=1)  # [B, N, output_size, T]
        
        # 调整维度顺序以便后续处理
        tcn_output = stacked_outputs.permute(0, 3, 1, 2)  # [B, T, N, C_tcn]
        
        # 投影到V空间
        v_projected = self.Wv_proj(tcn_output)  # [B, T, N, v_feature_dim]
        v = v_projected.permute(0, 2, 1, 3)  # -> [B, N, T, v_feature_dim]

        q, k = self.split(q), self.split(k)
        v = self.split(v)

        out, attn_weights = self.attention(q, k, v)

        out = self.concat(out)
        out = self.w_concat(out)

        return out

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
            x_embedding (torch.Tensor): 输入嵌入 [batch_size, series_num, d_model]。
            x (torch.Tensor): 原始输入或上一层输出 [batch_size, series_num, input_window, feature_dim]。
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
        n_layers (int): 编码器层数。
        ffn_hidden (int): FFN 隐藏层维度。
        drop_prob (float): Dropout 概率。
        tau (float): 注意力 softmax 温度。
        
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
        # x = [batch_size, input_window, series_num, feature_dim]
        x = x.permute(0, 2, 1, 3)  # [batch_size, series_num, input_window, feature_dim]
        encoder_out = self.encoder(x) # [batch_size, series_num, input_window, feature_dim]
        last_step_out = encoder_out[:, :, -1, :] # [batch_size, series_num, feature_dim]
        out = self.fc(last_step_out) # [batch_size, series_num, output_dim]
        out = out.unsqueeze(1) # [batch_size, 1, series_num, output_dim]
        if self.output_window > 1:
            out = out.repeat(1, self.output_window, 1, 1)
        return out

    # def get_GC(self, threshold=True, ignore_lag=True):
    #     GC_list_from_each_encoder_first_block = []
    #     for encoder_layer in self.encoder.layers:  # 遍历 EncoderLayer 实例
    #         tcn_processor = encoder_layer.attention.tcn_processor  # 这是一个 GrangerTCN 实例
            
    #         if tcn_processor.network_layers:  # 检查 TCN 是否有 TemporalBlock
    #             # 我们只关心 TCN 的第一个 TemporalBlock，因为它的输入直接对应 P 个原始序列
    #             first_temporal_block = tcn_processor.network_layers[0]
                
    #             # first_temporal_block.conv1.weight 的形状是:
    #             # [out_channels_C1, in_channels_P, kernel_size_K_tcn]
    #             # 其中 in_channels_P 是 self.series_num (原始序列数)
                
    #             current_norm = None
    #             if ignore_lag:
    #                 # 计算范数时，聚合 out_channels (dim 0) 和 kernel_size (dim 2)
    #                 # 得到每个输入序列 (in_channels_P) 的一个标量影响值
    #                 # 结果形状: [in_channels_P]
    #                 current_norm = torch.norm(first_temporal_block.conv1.weight, dim=(0, 2))
    #             else:
    #                 # 计算范数时，只聚合 out_channels (dim 0)
    #                 # 保留了 kernel_size 维度，表示每个输入序列在不同卷积核位置（滞后）的影响
    #                 # 结果形状: [in_channels_P, kernel_size_K_tcn]
    #                 current_norm = torch.norm(first_temporal_block.conv1.weight, dim=0)
                
    #             GC_list_from_each_encoder_first_block.append(current_norm)

    #     if not GC_list_from_each_encoder_first_block:
    #         # 如果没有 encoder layer 或 TCN block，返回一个空张量
    #         # 或者可以根据 series_num 和 tcn_kernel_size 返回特定形状的零张量
    #         # 例如: return torch.zeros(0, self.series_num).to(self.device) if ignore_lag else torch.zeros(0, self.series_num, self.config['model']['args']['tcn_kernel_size']).to(self.device)
    #         return torch.empty(0, device=self.device)


    #     # GC_list_from_each_encoder_first_block 中的所有张量现在都具有相同的形状:
    #     #   - 如果 ignore_lag=True: [series_num]
    #     #   - 如果 ignore_lag=False: [series_num, tcn_kernel_size]
        
    #     # 将来自每个 encoder layer 的 GC 张量堆叠起来
    #     # dim=0 表示将它们堆叠成一个新的第0维度 (num_encoder_layers)
    #     GC_stacked = torch.stack(GC_list_from_each_encoder_first_block, dim=0)
    #     # GC_stacked 的形状:
    #     #   - 如果 ignore_lag=True: [num_encoder_layers, series_num]
    #     #   - 如果 ignore_lag=False: [num_encoder_layers, series_num, tcn_kernel_size]

    #     if threshold:
    #         return (GC_stacked > 0).int()
    #     else:
    #         return GC_stacked
