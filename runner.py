from datetime import datetime
from pathlib import Path
import train
import interpret as interpret
from utils.util import  read_json


def main(config, run_id):
    # Run training
    train.main(config, run_id)
    
    # Reset or reload config before interpret
    config = read_json(Path(config_path))
    
    # Run interpret with clean config
    interpret.main(config, run_id)


# 单独运行这个文件
if __name__=="__main__":
    run_id = datetime.now().strftime(r'%m%d_%H%M%S') # 获得当前时间
    config_path = 'config/config_demo.json'
    config = read_json(Path(config_path))
    main(config, run_id)