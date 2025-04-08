import logging
import logging.config
from pathlib import Path
from utils import read_json

'''
它通过加载一个日志配置文件（默认为 logger/logger_config.json），并根据提供的保存目录路径（save_dir）
动态修改日志文件的存储路径，从而实现灵活的日志记录功能。
'''
def setup_logging(save_dir, log_config='logger/logger_config.json', default_level=logging.INFO):
    log_config = Path(log_config)
    if log_config.is_file():
        config = read_json(log_config)
        for _, handler in config['handlers'].items():
            if 'filename' in handler:
                handler['filename'] = str(save_dir / handler['filename'])

        logging.config.dictConfig(config)
    else:
        print("Warning: logging configuration file is not found in {}.".format(log_config))
        logging.basicConfig(level=default_level)
