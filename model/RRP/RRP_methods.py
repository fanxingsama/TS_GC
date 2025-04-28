import torch
import torch.nn as nn
from model.RRP_model import RegRelProp, RelProp
from util import safe_divide


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