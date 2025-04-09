# logger/logger.py
import logging
import logging.config
from pathlib import Path
from utils import read_json
import json
from collections import OrderedDict

def setup_logging(save_dir, log_config='logger/logger_config.json', default_level=logging.INFO):
    log_config = Path(log_config) # 加载日志配置文件
    if log_config.is_file():
        with open(log_config, 'r', encoding='utf-8') as file:
            config = json.load(file, object_hook=OrderedDict)
            for _, handler in config['handlers'].items():
                if 'filename' in handler:
                    handler['filename'] = str(save_dir / handler['filename'])
            logging.config.dictConfig(config)
    else: # 如果配置文件不存在
        print("Warning: logging configuration file is not found in {}.".format(log_config))
        logging.basicConfig(level=default_level)

log_levels = {
            0: logging.WARNING,
            1: logging.INFO,
            2: logging.DEBUG
        }
# 获取日志记录器。
def get_logger(name, verbosity=2):
    '''
    name:日志名称
    verbosity:日志级别
    '''
    msg_verbosity = 'verbosity option {} is invalid. Valid options are {}.'.format(verbosity, log_levels.keys())
    assert verbosity in log_levels, msg_verbosity
    logger = logging.getLogger(name) # 获取日志记录器
    logger.setLevel(log_levels[verbosity]) # 设置日志级别
    return logger