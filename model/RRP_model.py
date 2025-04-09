import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.util import safe_divide


# 在模块的前向传播中记录输入和输出，没有输入和输出无法计算相关性分数。
def forward_hook(self, input, output):
    if type(input[0]) in (list, tuple): # 检查输入是否是列表或元组。如果是，说明输入有多个张量。
        self.X = []
        for i in input[0]:
            x = i.detach() # 将张量从计算图中分离，以避免影响原始计算图
            x.requires_grad = True # 将分离后的张量设置为 requires_grad=True，以便在反向传播时可以计算梯度。
            self.X.append(x) # 保存张量到X里
    else:
        self.X = input[0].detach()
        self.X.requires_grad = True

    self.Y = output # 得到输出


# 定义了一个基类，用于实现 LRP 的基本功能。
class RelProp(nn.Module):
    def __init__(self):
        super(RelProp, self).__init__()
        self.register_forward_hook(forward_hook) # 保存输入和输出

    def gradprop(self, Z, X, S):
        C = torch.autograd.grad(Z, X, S, retain_graph=True) # 计算当前层的输出Z对输入的X的梯度
        return C

    def relprop(self, R):
        return R

# 相关性分数计算公式
class RegRelProp(RelProp):
    def relprop(self, R):
        """
        输入：R (torch.Tensor): 上一层的相关性评分。
        输出：R (torch.Tensor): 当前层的相关性评分。
        """
        Z = F.linear(self.X, self.weight) # 当前层的输出值
        S = safe_divide(R, Z) # 前一层相关性分数除以当前层的输出，得到归一化的相关性分数 S
        R = self.X * torch.autograd.grad(Z, self.X, S)[0] # 计算当前层的输出值Z对当前层输入X的梯度，并将其与归一化的相关性分数S相乘，之后乘当前层的输入
        return R

class RRP:
    def __init__(self, model):
        self.model = model
        self.model.eval() # 模型设置为评估模式

    # 收集并计算模型中每个层的因果关系分数，包括注意力矩阵和卷积核的因果分数。
    def generate_RRP(self, batch_size, input, interpreted_series):
        inputs = torch.split(input, batch_size)
        relAs, relKs = [], [] 
        for data in inputs: # 多批次数据叠加
            relA, relK = self._generate_RRP(data, interpreted_series)
            relAs.append(relA)
            relKs.append(relK)
        relA = torch.stack(relAs).mean(0)
        relK = torch.stack(relKs).mean(0)
        return relA, relK # 返回注意矩阵和卷积核的因果分数
        
    # 为单个批次的输入数据生成因果关系分数。
    def _generate_RRP(self, input, interpreted_series):
        """
        input (torch.Tensor):输入数据张量[total_batch， input_window, series_num, feature_dim]
        interpreted_series (int)：被解释时间序列的索引的序号。
        """
        output = self.model(input) # 得到模型输出
        
        one_hot = torch.zeros_like(output, dtype=torch.float).to(output.device) # 创建一个与输出形状相同的 one-hot 张量，仅在序列序号所对应的位置设置为 1。
        one_hot[:,:,interpreted_series,:] = 1
        one_hot_vector = one_hot.clone() # 克隆 one-hot 张量，并设置其需要计算梯度。
        one_hot.requires_grad_(True)
        one_hot = torch.sum(one_hot * output) # 计算 one-hot 张量与模型输出的点积。
        
        self.model.zero_grad()
        one_hot.backward(retain_graph=True)
        self.model.relprop(one_hot_vector)  # 调用模型的 relprop 方法，将 one-hot 张量的梯度传播回输入层，计算每个输入特征对输出的贡献

        # 收集因果关系分数
        relAs=[] # 注意力矩阵因果关系分数
        relKs=[] # 卷积核因果关系分数
        for layer in self.model.encoder.layers: # 遍历模型的编码器层，收集每个层的因果关系分数。
            # 梯度调制
            relA = layer.attention.attention.get_rel() * torch.abs(layer.attention.attention.get_grad())
            relK = layer.attention.Wv.get_rel() * torch.abs(layer.attention.Wv.get_grad())

            relA = relA.clamp(min=0)        # 只考虑正向因果分数
            relK = relK.clamp(min=0)        
            relAs.append(relA.mean((0,1)))  # mean for sample and head
            relKs.append(relK.mean(0))      # mean for head
        # 将所有层的分数堆叠起来，并计算它们的乘积，得到最终的因果关系分数。
        relA = torch.stack(relAs).prod(0)   
        relK = torch.stack(relKs).prod(0)  
        return relA, relK


# 克隆模块，用于在 LRP 中处理多个输入，将输入复制多次
class Clone(RelProp):
    def forward(self, input, num):
        self.__setattr__('num', num) # 将复制次数存储为模块的属性，以便在反向传播时使用。
        outputs = []
        for _ in range(num):
            outputs.append(input) # 将输入张量 input 复制 num 次，放入outputs中

        return outputs

    def relprop(self, R):
        Z = []
        for _ in range(self.num):
            Z.append(self.X) # 将输入张量 self.X 复制 num 次，放入Z中
        S = [safe_divide(r, z) for r, z in zip(R, Z)]
        C = self.gradprop(Z, self.X, S)[0]
        R = self.X * C

        return R
    

