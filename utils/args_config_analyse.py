import os
import argparse
from pathlib import Path
from functools import  partial
from datetime import datetime
from logger import setup_logging
from utils import read_json, write_json


# 用于解析args和配置项 JSON 文件的类。
class args_config_analyse:
    def __init__(self, config, run_id=None):
        """
        config：包含配置和训练超参数的字典。例如，config.json 文件的内容。
        run_id：训练过程的唯一标识符。用于保存检查点和训练日志，默认使用时间戳。
        """

        self._config = config
        if run_id is None:
            run_id = datetime.now().strftime(r'%m%d_%H%M%S')
        # 指定保存的log的地方和格式
        self._save_dir =  'saved' / run_id

        exist_ok = run_id == '' # 如果 run_id 是空字符串，exist_ok 被设置为 True。如果不是空字符串，则 exist_ok 被设置为 False
        
        save_dir = Path(self.config['trainer']['save_dir'])
        self.save_dir.mkdir(parents=True, exist_ok=exist_ok) # 创建目录

        write_json(self.config, self.save_dir / 'model_config.json') # 保存模型的超参数配置
        # setup_logging(self.log_dir) # 保存模型的运行log

    # 从命令行参数中解析配置文件，并返回一个args_config_analyse对象。
    @classmethod
    def from_args(aca, args, run_id=None):
        # 如果args是字典，则将其转换为argparse.Namespace对象
        if isinstance(args, dict):
            args = argparse.Namespace(**args)
        elif not isinstance(args, tuple): # 如果args不是字典也不是元组，则使用argparse解析命令行参数
            args = args.parse_args()

        if args.device is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = args.device

        config = read_json(Path(args.config))
        if hasattr(args, 'data_dir') and args.data_dir is not None: # 多个文件的情况下，可能需要读多个csv，所以需要修改数据路径
            config['data_loader']['args']['data_dir'] = args.data_dir

        # 返回一个args_config_analyse对象，重新调用了__init__方法
        return aca(config, run_id)

    # 根据给定的名称，初始化一个对象。
    def init_obj_by_config(self, name, module, *args, **kwargs):
        module_name = self[name]['type']
        module_args = dict(self[name]['args'])
        assert all([k not in module_args for k in kwargs]), '不允许覆盖配置文件中给定的参数'
        module_args.update(kwargs)
        '''
        getattr是为了从module中获取module_name对应的对象
        (*args, **module_args)是来给getattr(module, module_name)传参的
        过程就是先得到对象，然后用参数给对象实例化
        '''
        # ，getattr(module, module_name)相当于实例化了一个对象
        return getattr(module, module_name)(*args, **module_args)

    # 获取config字典中给定名称的条目。
    def __getitem__(self, name):
        return self.config[name]


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