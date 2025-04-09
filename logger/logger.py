import logging
import logging.config
from pathlib import Path
from utils import read_json

'''
加载一个日志配置文件（默认为 logger/logger_config.json），并根据提供的保存目录路径（save_dir）动态修改日志文件的存储路径，实现日志记录功能。
'''
def setup_logging(save_dir, log_config='logger/logger_config.json', default_level=logging.INFO):
    log_config = Path(log_config) # 加载日志配置文件
    if log_config.is_file():
        config = read_json(log_config) # 读取日志配置文件
        for _, handler in config['handlers'].items(): 
            # 遍历文件handlers部分，检查是否包含filename属性，如果有，路径修改为save_dir与原路径的结合
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