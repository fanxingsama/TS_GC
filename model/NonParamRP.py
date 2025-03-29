import torch
import torch.nn as nn
import torch.nn.functional as F

'''
LRP模型：将模型的输出（如预测结果）分解为输入特征的贡献，从而帮助理解模型的决策过程。

'''

# 安全地执行除法操作，避免除以零。
def safe_divide(a, b):
    den = b.clamp(min=1e-9) + b.clamp(max=1e-9)
    den = den + den.eq(0).type(den.type()) * 1e-9
    return a / den * b.ne(0).type(b.type())

# 在模块的前向传播中记录输入和输出。
def forward_hook(self, input, output):
    if type(input[0]) in (list, tuple):
        self.X = []
        for i in input[0]:
            x = i.detach()
            x.requires_grad = True
            self.X.append(x)
    else:
        self.X = input[0].detach()
        self.X.requires_grad = True

    self.Y = output


# 定义了一个基类，用于实现 LRP 的基本功能。
class RelProp(nn.Module):
    def __init__(self):
        super(RelProp, self).__init__()
        self.register_forward_hook(forward_hook)

    def gradprop(self, Z, X, S):
        C = torch.autograd.grad(Z, X, S, retain_graph=True)
        return C

    def relprop(self, R):
        return R
    
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
        self.__setattr__('num', num)
        outputs = []
        for _ in range(num):
            outputs.append(input)

        return outputs

    def relprop(self, R):
        Z = []
        for _ in range(self.num):
            Z.append(self.X)
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
    
# class IndexSelect(RelProp):
#     def forward(self, inputs, dim, indices):
#         self.__setattr__('dim', dim)
#         self.__setattr__('indices', indices)

#         return torch.index_select(inputs, dim, indices)

#     def relprop(self, R, alpha):
#         Z = self.forward(self.X, self.dim, self.indices)
#         S = safe_divide(R, Z)
#         C = self.gradprop(Z, self.X, S)

#         if torch.is_tensor(self.X) == False:
#             outputs = []
#             outputs.append(self.X[0] * C[0])
#             outputs.append(self.X[1] * C[1])
#         else:
#             outputs = self.X * (C[0])
#         return outputs

# class Clone(RelProp):
#     def forward(self, input, num):
#         self.__setattr__('num', num)
#         outputs = []
#         for _ in range(num):
#             outputs.append(input)

#         return outputs

#     def relprop(self, R, alpha):
#         Z = []
#         for _ in range(self.num):
#             Z.append(self.X)
#         S = [safe_divide(r, z) for r, z in zip(R, Z)]
#         C = self.gradprop(Z, self.X, S)[0]

#         R = self.X * C

#         return R

# class Cat(RelProp):
#     def forward(self, inputs, dim):
#         self.__setattr__('dim', dim)
#         return torch.cat(inputs, dim)

#     def relprop(self, R, alpha):
#         Z = self.forward(self.X, self.dim)
#         S = safe_divide(R, Z)
#         C = self.gradprop(Z, self.X, S)

#         outputs = []
#         for x, c in zip(self.X, C):
#             outputs.append(x * c)

#         return outputs


# class Sequential(nn.Sequential):
#     def relprop(self, R, alpha):
#         for m in reversed(self._modules.values()):
#             R = m.relprop(R, alpha)
#         return R

# class BatchNorm2d(nn.BatchNorm2d, RelProp):
#     def relprop(self, R, alpha):
#         X = self.X
#         beta = 1 - alpha
#         weight = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3) / (
#             (self.running_var.unsqueeze(0).unsqueeze(2).unsqueeze(3).pow(2) + self.eps).pow(0.5))
#         Z = X * weight + 1e-9
#         S = R / Z
#         Ca = S * weight
#         R = self.X * (Ca)
#         return R

# class Add(RelPropSimple):
#     def forward(self, inputs):
#         return torch.add(*inputs)

#     def relprop(self, R, alpha):
#         Z = self.forward(self.X)
#         S = safe_divide(R, Z)
#         C = self.gradprop(Z, self.X, S)

#         a = self.X[0] * C[0]
#         b = self.X[1] * C[1]

#         a_sum = a.sum()
#         b_sum = b.sum()

#         a_fact = safe_divide(a_sum.abs(), a_sum.abs() + b_sum.abs()) * R.sum()
#         b_fact = safe_divide(b_sum.abs(), a_sum.abs() + b_sum.abs()) * R.sum()

#         a = a * safe_divide(a_fact, a.sum())
#         b = b * safe_divide(b_fact, b.sum())

#         outputs = [a, b]

#         return outputs