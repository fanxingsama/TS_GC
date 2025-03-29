import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
from model.RRP import *
import math
from utils import prepare_device

# 时序嵌入
class Embedding(BaseModel):
    """
    参数：
        series_num (int): 输入中的时间序列数量。
        input_window (int): 输入时间序列窗口的长度。
        feature_dim (int): 时间序列中每个特征的维度。
        d_model (int): 嵌入向量的维度。在论文中表示为 D_QK。
        drop_prob (float): 用于正则化的 Dropout 概率。
        device (str): 计算设备（'cpu' 或 'cuda'）。
    """

    def __init__(self, series_num, input_window, feature_dim, d_model, drop_prob, device):
        super().__init__()
        self.series_num = series_num
        self.input_window = input_window
        self.feature_dim = feature_dim
        self.d_model = d_model
        self.drop_prob = drop_prob
        self.device = device
        self.feature_emb = Linear(in_features=self.input_window*self.feature_dim, out_features=self.d_model, bias=True) # 初始化线性层，将数据映射到高维空间
        # He初始化通常将权重初始化为服从均值为0、方差为2/n高斯分布或均匀分布，其中n是该层的输入单元数量（即特征数量或神经元数量）。
        self.feature_emb.weight.data.normal_(0, math.sqrt(2.0/(self.input_window*self.feature_dim+self.d_model))) # 对权重进行He初始化，选择适当的初始值，加速收敛。
        '''
        嵌入层使用归一化和DropOut的原因：
        归一化：嵌入层通常将输入数据映射到一个高维空间，这个过程可能会引入较大的尺度差异。通过 LayerNorm，可以确保嵌入向量的尺度一致，从而提高后续层的性能。
        DropOut：映射到高维空间之后，可能引入冗余特征，导致模型对这些特征的依赖程度过高
        '''
        self.norm = LayerNorm(self.d_model) # 初始化层归一化模块，把均值和方差调整到固定范围
        self.drop_out = Dropout(self.drop_prob) # 初始化 Dropout 层

    def forward(self, x):
        # [batch_size, series_num, input_window, feature_dim]
        '''
        # 将每个序列的多个时间步和特征展平为单个向量，从而适配feature_emb的输入
        # -1表示任意维度，具体的大小自动计算，确保重塑之后的张量元素总数和原始张量相同

        Linear 层的输入可以是任意维度的张量，但最后一个维度必须是 in_features。也就是说，输入形状可以是 [..., in_features]，其中 ... 表示任意数量的额外维度
        这里甚至可以写为x = x.reshape(-1, self.input_window*self.feature_dim)
        但是x = x.reshape(-1, self.series_num, self.input_window*self.feature_dim) 有更高的可读性，对数据结构的展示更好
        reshape只改变数据的维度结构，但是并不会改变变量所包含的特征信息

        同理，归一化操作是在 特征维度 上进行的，LayerNorm(self.d_model)要求最后一个特征大小必须是d_model
        '''
        x = x.reshape(-1, self.series_num, self.input_window*self.feature_dim) 
        embedding = self.feature_emb(x) 
        embedding = self.norm(embedding)
        embedding = self.drop_out(embedding)
        return embedding # 输出完整嵌入的向量
        # [batch_size, series_num, d_model]

# 多核因果卷积块
class CausalConv(BaseModel):
    """
        series_num (int): 输入中的时间序列数量。
        input_window (int): 输入时间序列窗口的长度。
        n_head (int): 注意力头的数量。在论文中表示为 h。
        device (str): 计算设备（'cpu' 或 'cuda'）。
        wgt (None): 核权重的占位符。
        grad (None): 梯度的占位符。
        rel (None): 相关性的占位符。
        K (nn.Parameter): 可学习的卷积核参数。
        mul (torch.einsum): 用于卷积的爱因斯坦求和。
        base (torch.Tensor): 用于缩放卷积结果的张量。
    """

    def __init__(self, series_num, input_window, n_head, device):
        super().__init__()
        self.series_num = series_num
        self.input_window = input_window
        self.n_head = n_head
        self.device = device

        self.wgt = None
        self.grad = None
        self.rel = None
        # nn.Parameter 表示该张量是一个可训练的参数
        self.K = nn.Parameter(torch.ones((self.n_head, self.series_num, self.series_num, self.input_window), dtype=torch.float))
        self.register_parameter("kernel", self.K) # 将参数 K 注册到模型中，命名为 "kernel"。这样可以在模型的参数列表中找到它。
        self.mul = einsum('hxyji,bxif->bhxyjf') # 使用爱因斯坦求和，实现张量计算
        # 创建了一个张量 base，其值是从 1 到 input_window 的整数序列，形状为 (1, 1, 1, 1, input_window, 1)，并将其移动到指定的设备计算。这个张量在后续的计算中用于修正计算结果。
        self.base = torch.tensor([i for i in range(1, self.input_window+1)]).reshape(1,1,1,1,-1,1).to(self.device)

    # 获取权重和保存权重
    def get_wgt(self):
        return self.wgt

    def save_wgt(self, wgt):
        self.wgt = wgt

    # 获取梯度和保存梯度
    def get_grad(self):
        return self.grad
    
    def save_grad(self, grad):
        self.grad = grad
    
    # 获取相关性并保存相关性
    def get_rel(self):
        return self.rel
    
    def save_rel(self, rel):
        self.rel = rel

    def forward(self, x):
        # [batch_size, series_num, input_window, hidden_dim]
        kernel = []
        '''
        对参数 K 进行循环移位操作，每次向右移动 i+1 个单位，并将结果存储在列表 kernel 中。
        通过循环移位生成多个版本的卷积核，每个卷积核都对输入时间序列进行不同程度的移位操作，以便考虑不同时间步之间的依赖关系。
        这样，模型可以学习到更长时间范围内的因果关系，而不仅仅是相邻时间步之间的关系。

        堆叠是为了将生成的多个移位版本的卷积核组织成一个统一的张量，方便后续的卷积操作。
        '''
        for i in range(self.input_window):
            shifted = torch.roll(self.K, i+1, dims=3) #  对张量进行移位操作，dim表示位移的维度，i+1是每次位移的步数，整数是向右
            kernel.append(shifted)
        kernel = torch.stack(kernel) # kernel 是一个列表，沿着一个新的维度将kernel堆叠起来。
        kernel = kernel.permute(1, 2, 3, 0, 4) # 对张量 kernel 的维度进行重新排列
        kernel = torch.tril(kernel, diagonal=0) # 将张量 kernel 的上三角部分置为零，仅保留其下三角部分（包括对角线），即未来的时间步不能影响过去的时间步。
        kernel.requires_grad_() # 加上梯度，训练的时候可以梯度更新

        self.save_wgt(kernel) # 保存权重
        kernel.register_hook(self.save_grad) # 注册梯度钩子，当模型进行反向传播时，会调用这个钩子函数，将梯度保存到 self.grad 中。

        x = self.mul([kernel ,x]) # einsum 用于张量计算。属于多核因果卷积
        x = x / self.base # 由于在因果卷积中可能会引入大量的padding，导致某些位置的计算结果被重复计算。通过除以 self.base，可以对结果进行归一化，修正由于填充导致的偏差。

        # 处理及时因果性
        for i in range(self.series_num):
            '''
            x[:,:,i,i,:,:]：表示第 i 个时间序列的对角线部分，即Si对Si的影响。
            roll(1, dims=2)：将第 2 维（时间维度）向右移动 1 步。这样，每个时间步的计算结果会依赖于前一个时间步的信息，而不是当前时间步的信息。
            x[:,:,i,i,0,:]：表示第 i 个时间序列的对角线部分的第一个时间步。
            torch.zeros_like(x[:,:,i,i,0,:])：生成一个与 x[:,:,i,i,0,:] 形状相同的零张量。
            *=：将第一个时间步的值置为零。为了确保第一个时间步没有依赖于任何未来信息，因为第一个时间步没有前一个时间步可以依赖。
            '''
            x[:,:,i,i,:,:] = x[:,:,i,i,:,:].roll(1, dims=2) 
            x[:,:,i,i,0,:] *= torch.zeros_like(x[:,:,i,i,0,:])

        return x
        # [batch_size, head, series_num(data source), series_num(data user), input_window, hidden_dim]
    
    # 计算卷积核 K 的 L1 范数的均值，用于正则化。
    def regularization(self):
        return torch.mean(torch.norm(self.K, dim=-1, p=1))

    # 相关性传播，解释模型的输出，
    def relprop(self, rel):
        for i in range(self.series_num):
            rel[:,:,i,i,:,:] = rel[:,:,i,i,:,:].roll(-1, dims=2)
        rel = rel * self.base
        rel_k, rel_x = self.mul.relprop(rel)
        self.save_rel(rel_k)
        return rel_x

# 多变量因果注意力
class MultiVariateCausalAttention(BaseModel):
    """
        series_num (int): 输入中的时间序列数量。
        input_window (int): 输入时间序列窗口的长度。
        feature_dim (int): 时间序列中每个特征的维度。
        d_model (int): 嵌入向量的维度。在论文中表示为 D_QK。
        n_head (int): 注意力头的数量。在论文中表示为 h。
        tau (float): softmax 的温度超参数。
        device (str): 计算设备（'cpu' 或 'cuda'）。
        wgt (None): 核权重的占位符。
        grad (None): 梯度的占位符。
        rel (None): 相关性的占位符。
        qk_mul (torch.einsum): 查询-键乘法的爱因斯坦求和。
        softmax (torch.nn.Softmax): softmax 函数。
        mask (nn.Parameter): 可学习的注意力掩码，用于稀疏性调整。
        hardmard_product (torch.einsum): 元素逐项乘法的爱因斯坦求和。
        mul (torch.einsum): 注意力矩阵-值乘法的爱因斯坦求和。
    """

    def __init__(self, series_num, input_window, feature_dim, d_model, n_head, tau, device):
        super().__init__()
        self.series_num = series_num
        self.input_window = input_window
        self.feature_dim = feature_dim
        self.d_model = d_model
        self.n_head = n_head
        self.d_tensor = self.d_model // self.n_head
        self.tau = tau
        self.device = device
        self.wgt = None
        self.grad = None
        self.rel = None
        self.qk_mul = einsum('bhid,bhdj->bhij') # 进行Q和K的点积运算
        self.softmax = Softmax(dim=-1)
        self.mask = nn.Parameter(torch.ones((self.n_head, self.series_num, self.series_num), dtype=torch.float)) # 一个可学习的注意力掩码
        self.register_parameter("mask", self.mask) # 注册名为mask的参数，添加到模型的参数列表中
        self.hardmard_product = einsum('hij,bhij->bhij') # 进行掩码与点积结果的逐元素乘法
        self.mul = einsum('bhij,bhjitf->bhitf') # 注意力矩阵与V的点积

    def get_wgt(self):
        return self.wgt

    def save_wgt(self, satt):
        self.wgt = satt

    def get_grad(self):
        return self.grad
    
    def save_grad(self, grad):
        self.grad = grad
    
    def get_rel(self):
        return self.rel
    
    def save_rel(self, rel):
        self.rel = rel

    # 执行注意力过程
    def forward(self, q, k, v):
        # q, k: [batch_size, head, series_num, d_tensor]
        # v: [batch_size, head, series_num(data source), series_num(data user), input_window, feature_dim]

        k_t = k.transpose(2, 3) # 把K的维度交换
        score = self.qk_mul([q,k_t])/math.sqrt(self.input_window*self.d_tensor) # 缩放点积
        A = self.hardmard_product([self.mask, score]) # 将mask与score进行逐元素乘法，使用掩码调整注意力权重
        A = self.softmax(A/self.tau) # 对A进行softmax操作，将A的值映射到[0, 1]范围
        A.requires_grad_() # 让A可以反向传播
        self.save_wgt(A) # 保存注意力权重
        A.register_hook(self.save_grad) # 注册梯度钩子，当模型进行反向传播时，会调用这个钩子函数，将梯度保存到 self.grad 中
        out = self.mul([A, v]) # 注意力权重矩阵A和V矩阵相乘
        return out
        # [batch_size, head, series_num, input_window, feature_dim]
    
    # 计算掩码的 L1 范数的均值，作为正则化项。这有助于稀疏化掩码，减少注意力权重的冗余。
    def regularization(self):
        return torch.mean(torch.norm(self.mask, dim=-1, p=1))
    
    # 定义相关性传播方法
    def relprop(self, rel):
        rel_A, rel_v = self.mul.relprop(rel)
        self.save_rel(rel_A)
        rel_score = self.softmax.relprop(rel_A)
        rel_mask, rel_score = self.hardmard_product.relprop(rel_score)
        rel_score *= math.sqrt(self.input_window * self.d_tensor)
        rel_q, rel_k = self.qk_mul.relprop(rel_score)
        rel_k = rel_k.transpose(2, 3)
        return rel_q, rel_k, rel_v

# 多头注意力
class MultiHeadAttention(BaseModel):
    """
        series_num (int): 输入中的时间序列数量。
        input_window (int): 输入时间序列窗口的长度。
        feature_dim (int): 时间序列中每个特征的维度。
        d_model (int): 嵌入向量的维度。在论文中表示为 D_QK。
        n_head (int): 注意力头的数量。在论文中表示为 h。
        tau (float): 用于 softmax 的温度超参数。
        device (str): 计算设备（'cpu' 或 'cuda'）。
        attention (MultiVariateCausalAttention): MultiVariateCausalAttention 类的实例。
        Wq (nn.Linear): 查询的线性投影层。
        Wk (nn.Linear): 键的线性投影层。
        Wv (CausalConv): 值的因果卷积层。
        w_concat (nn.Linear): 用于拼接注意力头输出的线性投影层。
    """

    def __init__(self, series_num, input_window, feature_dim, d_model, n_head, tau, device):
        super().__init__()
        self.series_num = series_num
        self.input_window = input_window
        self.feature_dim = feature_dim
        self.d_model = d_model
        self.n_head = n_head
        self.tau = tau
        self.device = device
        
        self.attention = MultiVariateCausalAttention(self.series_num, self.input_window, self.feature_dim, self.d_model, self.n_head, self.tau, self.device)
        self.Wq = Linear(in_features=self.d_model, out_features=self.d_model, bias=True)
        self.Wq.weight.data.normal_(0, math.sqrt(2.0/(self.d_model+self.d_model))) # He初始化
        self.Wk = Linear(in_features=self.d_model, out_features=self.d_model, bias=True)
        self.Wk.weight.data.normal_(0, math.sqrt(2.0/(self.d_model+self.d_model))) # He初始化
        self.Wv = CausalConv(self.series_num, self.input_window, self.n_head, self.device)
        self.w_concat = Linear(in_features=self.n_head * self.feature_dim, out_features=self.feature_dim, bias=False)
        self.w_concat.weight.data.normal_(0, math.sqrt(2.0/(self.d_model+self.d_model))) # He初始化

    def forward(self, q, k, v):
        # q, k: [batch_size, series_num, d_model]
        # v: [batch_size, head, series_num, input_window, feature_dim]

        # 1. dot product with weight matrices
        q, k, v = self.Wq(q), self.Wk(k), self.Wv(v)
        # q, k: [batch_size, series_num, d_model]
        # v: [batch_size, head, series_num(data source), series_num(data user), input_window, feature_dim]

        # 2. split tensor by number of heads
        q, k = self.split(q), self.split(k) # 将查询（Q）和键（K）张量按注意力头的数量进行拆分
        # q, k: [batch_size, head, series_num, d_tensor]
        
        # 3. do scale dot product to compute similarity
        out = self.attention(q, k, v)
        # out: [batch_size, head, series_num, input_window, feature_dim]
        
        # 4. concat and pass to linear layer
        out = out.reshape(-1, self.n_head, self.series_num * self.input_window, self.feature_dim)
        out = self.concat(out)
        out = out.reshape(-1, self.series_num, self.input_window, self.n_head * self.feature_dim)
        out = self.w_concat(out)
        return out
        # [batch_size, series_num, input_window, feature_dim]

    def split(self, tensor):
        batch_size, length, d_model = tensor.size()

        d_tensor = d_model // self.n_head
        tensor = tensor.view(batch_size, length, self.n_head, d_tensor).transpose(1, 2)

        return tensor

    def concat(self, tensor):
        batch_size, head, length, d_tensor = tensor.size()
        d_model = head * d_tensor

        tensor = tensor.permute(0, 2, 1, 3).contiguous().view(batch_size, length, d_model)
        return tensor

    def regularization(self):
        return self.attention.regularization() + self.Wv.regularization()

    def relprop(self, rel):
        rel = self.w_concat.relprop(rel)
        rel = rel.reshape(-1, self.series_num * self.input_window, self.n_head * self.feature_dim)
        rel = self.split(rel)
        rel = rel.reshape(-1, self.n_head, self.series_num, self.input_window, self.feature_dim)
        rel_q, rel_k, rel_v = self.attention.relprop(rel)
        rel_q, rel_k = self.concat(rel_q), self.concat(rel_k)
        rel_q, rel_k, rel_v = self.Wq.relprop(rel_q), self.Wk.relprop(rel_k), self.Wv.relprop(rel_v)
        return rel_q, rel_k, rel_v

class PositionwiseFeedForward(BaseModel):
    """
    This class implements the Positionwise Feed Forward Layer described in the paper.
    It is composed of two linear neural networks separated by a leaky ReLU activation function.

    参数：
        dim (int): 输入维度。
        hidden (int): 前馈层中间的隐藏维度。在论文中表示为 d_FFN。
        drop_prob (float): Dropout 概率（实际中未使用）。
        linear1 (nn.Linear): 第一个线性层，将输入转换为隐藏维度。
        linear2 (nn.Linear): 第二个线性层，将隐藏维度转换回输入维度。
        activation (nn.LeakyReLU): Leaky ReLU 激活函数。
        dropout (nn.Dropout): 具有指定概率的 Dropout 层（实际中未使用）。
    """

    def __init__(self, dim, hidden, drop_prob=0.1):
        super().__init__()
        self.linear1 = Linear(dim, hidden, bias=True)
        # He Initialization
        self.linear1.weight.data.normal_(0, math.sqrt(2.0/(dim+hidden)))
        self.linear2 = Linear(hidden, dim, bias=True)
        # He Initialization
        self.linear2.weight.data.normal_(0, math.sqrt(2.0/(hidden+dim)))
        self.activation = LeakyReLU()
        self.dropout = Dropout(drop_prob)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

    def relprop(self, rel):
        rel = self.linear2.relprop(rel)
        rel = self.dropout.relprop(rel)
        rel = self.activation.relprop(rel)
        rel = self.linear1.relprop(rel)
        return rel

class EncoderLayer(BaseModel):

    def __init__(self, series_num, input_window, feature_dim, d_model, n_head, ffn_hidden, drop_prob, tau, device):
        super().__init__()
        self.qk = Clone()
        self.attention = MultiHeadAttention(series_num, input_window, feature_dim, d_model, n_head, tau, device)
        self.norm1 = LayerNorm([input_window, feature_dim])
        self.dropout1 = Dropout(drop_prob)

        self.ffn = PositionwiseFeedForward(dim=feature_dim, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm2 = LayerNorm([input_window, feature_dim])
        self.dropout2 = Dropout(drop_prob)

    def forward(self, x_embedding, x):
        # 1. compute self attention
        # x_embedding: [batch_size, series_num, d_model]
        # x: [batch_size, series_num, input_window, feature_dim]
        q, k = self.qk(x_embedding, 2)
        x = self.attention(q=q, k=k, v=x)
        # x: [batch_size, series_num, input_window, feature_dim]
        
        # 2. add and norm
        x = self.dropout1(x)
        x = self.norm1(x)
        
        # 3. positionwise feed forward network
        x = self.ffn(x)

        # 4. add and norm
        x = self.dropout2(x)
        x = self.norm2(x)
        return x
        # x: [batch_size, series_num, input_window, feature_dim]

    def regularization(self):
        return self.attention.regularization()
    
    def relprop(self, rel):
        rel = self.norm2.relprop(rel)
        rel = self.dropout2.relprop(rel)
        rel = self.ffn.relprop(rel)
        rel = self.norm1.relprop(rel)
        rel = self.dropout1.relprop(rel)
        rel_q, rel_k, rel_v = self.attention.relprop(rel)
        rel_emb = self.qk.relprop((rel_q, rel_k))
        return rel_emb, rel

class Encoder(BaseModel):
    """
    This class implements an Encoder Layer of the Causality-Aware Transformer.

    参数：
        series_num (int): 输入中的时间序列数量。
        input_window (int): 输入时间序列窗口的长度。
        feature_dim (int): 时间序列中每个特征的维度。
        d_model (int): 嵌入向量的维度。在论文中表示为 D_QK。
        n_head (int): 注意力头的数量。在论文中表示为 h。
        ffn_hidden (int): 前馈层中的隐藏维度。在论文中表示为 d_FFN。
        drop_prob (float): Dropout 概率（实际中未使用）。
        tau (float): 用于注意力 softmax 的温度超参数。
        device (str): 计算设备（'cpu' 或 'cuda'）。
        qk (Clone): Clone 类的实例。
        attention (MultiHeadAttention): MultiHeadAttention 类的实例。
        norm1 (LayerNorm): 第一个注意力块后的层归一化。
        dropout1 (Dropout): 第一个注意力块后的 Dropout 层（实际中未使用）。
        ffn (PositionwiseFeedForward): PositionwiseFeedForward 类的实例。
        norm2 (LayerNorm): 前馈块后的层归一化。
        dropout2 (Dropout): 前馈块后的 Dropout 层（实际中未使用）。
    """

    def __init__(self, series_num, input_window, feature_dim, d_model, n_head, n_layers, ffn_hidden, drop_prob, tau, device):
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
                                                  ffn_hidden=ffn_hidden,
                                                  drop_prob=drop_prob,
                                                  tau=tau,
                                                  device=device)
                                     for _ in range(n_layers)])

    def forward(self, x):
        # x: [batch_size, series_num, input_window, feature_dim]
        embedding = self.emb(x)
        for layer in self.layers:
            x = layer(embedding, x)
        return x
        # x: [batch_size, series_num, input_window, feature_dim]

    def regularization(self):
        loss = 0
        for layer in self.layers:
            loss += layer.regularization()
        return loss/len(self.layers)

    def relprop(self, rel):
        for layer in self.layers:
            emb_rel, rel = layer.relprop(rel)
        return rel

class PredictModel(BaseModel):
    """
    This class implements the PredictModel, a causality-aware transformer-based deep learning model
    for making predictions on time series data.

    参数：
        config (dict): 包含数据加载器和架构参数的配置字典。
        d_model (int): 嵌入向量的维度。在论文中表示为 D_QK。
        n_head (int): 注意力头的数量。在论文中表示为 h。
        n_layers (int): 编码器层数。
        ffn_hidden (int): 前馈层中的隐藏维度。在论文中表示为 d_FFN。
        drop_prob (float): Dropout 概率（实际中未使用）。
        tau (float): 用于注意力 softmax 的温度超参数。
        data_feature (dict): 数据加载器参数。
        model_config (dict): 架构参数。
        input_window (int): 输入时间序列窗口的长度。
        output_window (int): 输出时间序列窗口的长度。
        series_num (int): 输入中的时间序列数量。
        feature_dim (int): 时间序列中每个特征的维度。
        output_dim (int): 模型输出的维度。
        device (torch.device): 计算设备。
        encoder (Encoder): Encoder 类的实例。
        fc (nn.Linear): 用于预测输出的线性层。
    """

    def __init__(self, config, d_model, n_head, n_layers, ffn_hidden, drop_prob, tau):
        super().__init__()
        self.data_feature = config['data_loader']['args']
        self.model_config = config['arch']['args']

        self.input_window = self.data_feature.get('time_step')
        self.output_window = self.data_feature.get('output_window')
        self.series_num = self.data_feature.get('series_num')
        self.feature_dim = self.data_feature.get('feature_dim')
        self.output_dim = self.data_feature.get('output_dim')

        self.d_model = d_model
        self.n_head = n_head
        self.n_layers = n_layers
        self.ffn_hidden = ffn_hidden
        self.drop_prob = drop_prob
        self.tau = tau

        self.device, device_ids = prepare_device(config['n_gpu'])

        self.encoder = Encoder(series_num=self.series_num,
                               input_window=self.input_window,
                               feature_dim=self.feature_dim,
                               d_model=self.d_model, 
                               n_head=self.n_head,
                               n_layers=self.n_layers,
                               ffn_hidden=self.ffn_hidden,
                               drop_prob=self.drop_prob,
                               tau=self.tau,
                               device=self.device)
        
        self.fc = Linear(in_features=self.feature_dim, out_features=self.output_dim, bias=True)
        # He Initialization
        self.fc.weight.data.normal_(0, math.sqrt(2.0/(self.d_model+self.output_dim)))
    def forward(self, x):
        # x = [batch_size, input_window, series_num, feature_dim]
        x = x.permute(0, 2, 1, 3)  # [batch_size, series_num, input_window, feature_dim]
        out = self.encoder(x) # [batch_size, series_num, input_window, feature_dim]
        out = self.fc(out) # [batch_size, series_num, input_window, output_dim]
        out = out.permute(0, 2, 1, 3)
        out = out[:,-self.output_window:,...]
        return out
    
    def regularization(self):
        return self.encoder.regularization()
    
    def relprop(self, rel):
        pad = torch.zeros((rel.shape[0],self.input_window-self.output_window,rel.shape[2],rel.shape[3])).to(self.device)
        rel = torch.cat((pad,rel),1)
        rel = rel.permute(0, 2, 1, 3)
        rel = self.fc.relprop(rel)
        rel = self.encoder.relprop(rel)
        rel = rel.permute(0, 2, 1, 3)
        return rel
