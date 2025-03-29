import torch.nn as nn
import numpy as np
from abc import abstractmethod

# BaseModel 是一个抽象基类，它确保所有子类都必须实现 forward 方法。这有助于保持代码的规范性和一致性。
class BaseModel(nn.Module):
    @abstractmethod # 这个装饰器表示这个方法必须在子类中实现，否则会抛出异常。
    def forward(self, *inputs):
        raise NotImplementedError # 子类必须实现这个方法，否则会抛出错误。

    # 打印模型的可训练参数数量。通过print触发
    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)
