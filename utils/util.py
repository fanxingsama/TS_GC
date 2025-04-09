import json
import torch
import pandas as pd
from pathlib import Path
from itertools import repeat
from collections import OrderedDict

# 安全地执行除法操作，避免除以零。
def safe_divide(a, b):
    den = b.clamp(min=1e-9) + b.clamp(max=1e-9)
    den = den + den.eq(0).type(den.type()) * 1e-9
    return a / den * b.ne(0).type(b.type())

# 确保指定目录存在
def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)

# 读取json文件，返回内容
def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)

# 把内容写入json文件
def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)

# 创建一个无限循环的数据加载器。
def inf_loop(data_loader):
    for loader in repeat(data_loader):
        yield from loader

# 配置GPU
def prepare_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device

# 跟踪和计算各种指标的平均值
class MetricTracker:
    def __init__(self, *keys):
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    # 重置所有指标的总和、计数和平均值为 0。
    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    # 更新指定指标的值和计数
    def update(self, key, value, n=1):
        self._data.loc[key, 'total'] += value * n
        self._data.loc[key, 'counts'] += n
        self._data.loc[key, 'average'] = self._data.loc[key, 'total'] / self._data.loc[key, 'counts']

    # 获取指定指标的平均值。
    def avg(self, key):
        return self._data.average[key]

    # 获取所有指标的平均值。
    def result(self):
        return dict(self._data.average)
