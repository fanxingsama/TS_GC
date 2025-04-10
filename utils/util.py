import json
import os
import argparse
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
    # 检查文件是否存在
    if not fname.exists():
        # 文件不存在，创建并写入内容
        with fname.open('w') as handle:
            json.dump(content, handle, indent=4, sort_keys=False)
    else:
        # 文件已存在，不进行操作
        print(f"文件 {fname} 已存在，不进行写入操作。")

# 根据给定的名称，初始化一个对象。
def init_obj_by_config(config_json, name, module, *args, **kwargs):
    module_name = config_json[name]['type']
    module_args = dict(config_json[name]['args'])
    assert all([k not in module_args for k in kwargs]), '不允许覆盖配置文件中给定的参数'
    module_args.update(kwargs)
    '''
    getattr是为了从module中获取module_name对应的对象
    (*args, **module_args)是来给getattr(module, module_name)传参的
    过程就是先得到对象，然后用参数给对象实例化
    '''
    # ，getattr(module, module_name)相当于实例化了一个对象
    return getattr(module, module_name)(*args, **module_args)

def from_args(args):
        # 如果args是字典，则将其转换为argparse.Namespace对象
        if isinstance(args, dict):
            args = argparse.Namespace(**args)
        elif not isinstance(args, tuple): # 如果args不是字典也不是元组，则使用argparse解析命令行参数
            args = args.parse_args()

        if args.device is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = args.device

        config = read_json(Path(args.config))
        if hasattr(args, 'name') and args.name is not None:
            config['name'] = args.name
        if hasattr(args, 'data_dir') and args.data_dir is not None: # 多个文件的情况下，可能需要读多个csv，所以需要修改数据路径
            config['data_loader']['args']['data_dir'] = args.data_dir

        return config
# 创建一个无限循环的数据加载器。
def inf_loop(data_loader):
    for loader in repeat(data_loader):
        yield from loader

