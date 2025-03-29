import torch
import torch.nn as nn
import torch.nn.functional as F
from model.NonParamRP import *

class RegRelProp(RelProp):
    # 对线性层进行回归相关传播。
    def relprop(self, R):
        """
        Args:
            R (torch.Tensor): Relevance scores from the previous layer.

        Returns:
            R (torch.Tensor): Relevance scores for the current layer.
        """
        Z = F.linear(self.X, self.weight)
        S = safe_divide(R, Z)
        R = self.X * torch.autograd.grad(Z, self.X, S)[0]
        return R

class Linear(nn.Linear, RegRelProp):
    """
    这个类扩展了nn。线性类纳入回归相关传播（RRP）算法。
    它通过线性层执行回归相关性传播。
    """
    pass
