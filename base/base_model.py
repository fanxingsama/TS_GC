import torch.nn as nn
import numpy as np
from abc import abstractmethod

class BaseModel(nn.Module):
    @abstractmethod # 这个装饰器表示这个方法必须在子类中实现，否则会抛出异常。
    def forward(self, *inputs):
        raise NotImplementedError 

    # 打印模型的可训练参数数量。通过print触发
    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)
