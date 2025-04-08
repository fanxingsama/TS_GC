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


# 用于解析args和配置项 JSON 文件的类。
class args_config_analyse:
    def __init__(self, config, resume=None, run_id=None):
        """
        config：包含配置和训练超参数的字典。例如，config.json 文件的内容。
        resume：字符串，正在加载的检查点的路径。
        modification：字典，键值对为 keychain:value，指定要从配置字典中替换的位置值。
        run_id：训练过程的唯一标识符。用于保存检查点和训练日志，默认使用时间戳。
        """

        self._config = config
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

    # 从命令行参数中解析配置文件，并返回一个args_config_analyse对象。
    @classmethod
    def from_args(aca, args, options='', run_id=None):
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

        config = read_json(Path(args.config))
        if hasattr(args, 'name') and args.name is not None:
            config['name'] = args.name
        if hasattr(args, 'data_dir') and args.data_dir is not None: # 多个文件的情况下，可能需要读多个csv，所以需要修改数据路径
            config['data_loader']['args']['data_dir'] = args.data_dir

        # 返回一个args_config_analyse对象，重新调用了__init__方法
        return aca(config, None, run_id)

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