import os
import logging
import torch
import argparse
from pathlib import Path
from functools import reduce, partial
from operator import getitem
from datetime import datetime
from logger import setup_logging
from utils import read_json, write_json


# 用于解析配置 JSON 文件的类。
class ConfigParser:
    def __init__(self, config, resume=None, modification=None, run_id=None):
        """
        config：包含配置和训练超参数的字典。例如，config.json 文件的内容。
        resume：字符串，正在加载的检查点的路径。
        modification：字典，键值对为 keychain:value，指定要从配置字典中替换的位置值。
        run_id：训练过程的唯一标识符。用于保存检查点和训练日志，默认使用时间戳。
        """
        self._config = _update_config(config, modification) # 根据modification更新config
        self.resume = resume
        save_dir = Path(self.config['trainer']['save_dir']) # 根据配置文件中的 save_dir 和实验名称 name，创建保存模型和日志的目录。

        exper_name = self.config['name']
        if run_id is None: # 使用时间戳作为默认的运行id
            run_id = datetime.now().strftime(r'%m%d_%H%M%S')
        self._save_dir = save_dir / 'models' / exper_name / run_id
        self._log_dir = save_dir / 'log' / exper_name / run_id

        # 创建保存检查点和日志的目录。
        exist_ok = run_id == ''
        self.save_dir.mkdir(parents=True, exist_ok=exist_ok)
        self.log_dir.mkdir(parents=True, exist_ok=exist_ok)

        # 将更新后的配置文件保存到检查点目录中。
        write_json(self.config, self.save_dir / 'config.json')

        # 配置日志模块
        setup_logging(self.log_dir)
        self.log_levels = {
            0: logging.WARNING,
            1: logging.INFO,
            2: logging.DEBUG
        }

    # 从命令行参数中解析配置文件，并返回一个ConfigParser对象。
    @classmethod #  允许在不直接调用类构造函数的情况下创建类的实例，即不需要实例化ConfigParser，直接用ConfigParser.from_args()来解析配置文件
    def from_args(cls, args, options='', run_id=None):
        # cls在这里表示ConfigParser类本身
        # run_id：唯一标识符，用于保存检查点和日志
        for opt in options:
            args.add_argument(*opt.flags, default=None, type=opt.type)

        
        # 如果args是字典，则将其转换为argparse.Namespace对象
        if isinstance(args, dict):
            args = argparse.Namespace(**args)
        elif not isinstance(args, tuple): # 如果args不是字典也不是元组，则使用argparse解析命令行参数
            args = args.parse_args()

        if args.device is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = args.device
        
        # 如果指定了检查点参数，则从指定的检查点中加载配置文件。
        if args.resume is not None:
            resume = Path(args.resume)
            cfg_fname = resume.parent / 'config.json'
        else:
            msg_no_cfg = "Configuration file need to be specified. Add '-c config.json', for example."
            assert args.config is not None, msg_no_cfg
            resume = None
            cfg_fname = Path(args.config)
        
        # 从配置文件中读取配置信息，并更新为默认值
        config = read_json(cfg_fname)
        if hasattr(args, 'name') and args.name is not None:
            config['name'] = args.name
        if hasattr(args, 'data_dir') and args.data_dir is not None:
            config['data_loader']['args']['data_dir'] = args.data_dir
        
        # 如果指定了配置文件和检查点，则更新配置信息,用于微调
        if args.config and resume:
            config.update(read_json(args.config))

        # 
        modification = {opt.target : getattr(args, _get_opt_name(opt.flags)) for opt in options}
        return cls(config, resume, modification, run_id)

    # 根据给定的名称，初始化一个对象。
    def init_obj(self, name, module, *args, **kwargs):
        """
        `object = config.init_obj('name', module, a, b=1)`
        is equivalent to
        `object = module.name(a, b=1)`
        """
        module_name = self[name]['type']
        module_args = dict(self[name]['args'])
        assert all([k not in module_args for k in kwargs]), 'Overwriting kwargs given in config file is not allowed'
        module_args.update(kwargs)
        return getattr(module, module_name)(*args, **module_args)

    # 根据给定的名称，初始化一个函数。
    def init_ftn(self, name, module, *args, **kwargs):
        """
        `function = config.init_ftn('name', module, a, b=1)`
        is equivalent to
        `function = lambda *args, **kwargs: module.name(a, *args, b=1, **kwargs)`.
        """
        module_name = self[name]['type']
        module_args = dict(self[name]['args'])
        assert all([k not in module_args for k in kwargs]), 'Overwriting kwargs given in config file is not allowed'
        module_args.update(kwargs)
        return partial(getattr(module, module_name), *args, **module_args)

    # 获取config字典中给定名称的条目。
    def __getitem__(self, name):
        return self.config[name]

    # 获取日志记录器。
    def get_logger(self, name, verbosity=2):
        msg_verbosity = 'verbosity option {} is invalid. Valid options are {}.'.format(verbosity, self.log_levels.keys())
        assert verbosity in self.log_levels, msg_verbosity
        logger = logging.getLogger(name)
        logger.setLevel(self.log_levels[verbosity])
        return logger

    # 采用property装饰器，让_config、_save_dir、_log_dir等属性都能够被直接取用，可以直接采用属性的方式获取
    # 比如con = ConfigParser(xx),之后con.config就是获取的_config
    @property
    def config(self):
        return self._config

    @property
    def save_dir(self):
        return self._save_dir

    @property
    def log_dir(self):
        return self._log_dir

# 根据 modification 字典更新config中的值
def _update_config(config, modification):
    if modification is None:
        return config

    for k, v in modification.items():
        if v is not None:
            _set_by_path(config, k, v)
    return config

# 根据flags获取opt的名称
def _get_opt_name(flags):
    for flg in flags:
        if flg.startswith('--'):
            return flg.replace('--', '')
    return flags[0].replace('--', '')

# 根据keys设置tree中的值
def _set_by_path(tree, keys, value):
    keys = keys.split(';')
    _get_by_path(tree, keys[:-1])[keys[-1]] = value

# 根据keys获取tree中的值
def _get_by_path(tree, keys):
    return reduce(getitem, keys, tree)
