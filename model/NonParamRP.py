import torch
import torch.nn as nn
import torch.nn.functional as F

# 安全地执行除法操作，避免除以零。
def safe_divide(a, b):
    den = b.clamp(min=1e-9) + b.clamp(max=1e-9)
    den = den + den.eq(0).type(den.type()) * 1e-9
    return a / den * b.ne(0).type(b.type())

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


class RegRelProp(RelProp):
    # 相关性分数计算公式
    def relprop(self, R):
        """
        输入：R (torch.Tensor): 上一层的相关性评分。
        输出：R (torch.Tensor): 当前层的相关性评分。
        """
        Z = F.linear(self.X, self.weight) # 当前层的输出值
        S = safe_divide(R, Z) # 前一层相关性分数除以当前层的输出，得到归一化的相关性分数 S
        R = self.X * torch.autograd.grad(Z, self.X, S)[0] # 计算当前层的输出值Z对当前层输入X的梯度，并将其与归一化的相关性分数S相乘，之后乘当前层的输入
        return R


class Linear(nn.Linear, RegRelProp):
    pass


class LeakyReLU(nn.LeakyReLU, RelProp):
    pass


class Softmax(nn.Softmax, RelProp):
    pass


class LayerNorm(nn.LayerNorm, RelProp):
    pass


class Dropout(nn.Dropout, RelProp):
    pass


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
    

# 执行爱因斯坦求和
class einsum(RelProp):
    def __init__(self, equation):
        super().__init__()
        self.equation = equation
    def forward(self, *operands):
        return torch.einsum(self.equation, *operands)
    
    def relprop(self, R):
        Z = self.forward(self.X)
        S = safe_divide(R, Z)
        C = self.gradprop(Z, self.X, S)

        if torch.is_tensor(self.X) == False:
            outputs = []
            outputs.append(self.X[0] * C[0])
            outputs.append(self.X[1] * C[1])
        else:
            outputs = self.X * (C[0])
        return outputs