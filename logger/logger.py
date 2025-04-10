# logger/logger.py
import logging
import logging.config
from pathlib import Path
from utils import read_json
import json
from collections import OrderedDict

# 设置日志的基本配置，包括创建文件等
def setup_logging(save_dir, log_config='logger/logger_config.json', default_level=logging.INFO):
    log_config = Path(log_config) # 加载日志配置文件
    if log_config.is_file():
        with open(log_config, 'r', encoding='utf-8') as file:
            config = json.load(file, object_hook=OrderedDict)
            for _, handler in config['handlers'].items():
                if 'filename' in handler: # 如果存在 filename 键，则更新其值为 save_dir 与 filename 的组合路径。
                    handler['filename'] = str(save_dir / handler['filename']) 
            logging.config.dictConfig(config)
    else: # 如果配置文件不存在
        print("Warning: logging configuration file is not found in {}.".format(log_config))
        logging.basicConfig(level=default_level)

# 获取日志记录器，用来写入log
def get_logger():
    logger = logging.getLogger('train') # 获取日志记录器
    logger.setLevel(logging.INFO) # 设置日志级别
    return logger