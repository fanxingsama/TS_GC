import argparse
from parse_config import ConfigParser
from datetime import datetime
import train
import interpret
import torch
import os
import numpy as np
import pandas as pd
from pathlib import Path
from utils import read_json

# 构造不同任务下的数据集和真实标签路径
def construct_demo():
    task_list = {}
    for i in [15]:
        task_list[f'fMRI{i}'] = {
            'dataset': f"data/fMRI/timeseries{i}.csv",
            'groundtruth': f"data/fMRI/sim{i}_gt_processed.csv"
        }
    return task_list

def construct_fMRI():
    task_list = {}
    for i in range(1,29):
        task_list[f'fMRI{i}'] = {
            'dataset': f"data/fMRI/timeseries{i}.csv",
            'groundtruth': f"data/fMRI/sim{i}_gt_processed.csv"
        }
    return task_list

tasks={
    'demo': construct_demo,
    'fMRI': construct_fMRI,
}    

def runtask(label, args, dataset, ground_truth, task_name):
    SEED = 123
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(SEED)

    # 构建一个字典，里面包含项目运行所需的基本参数
    args_dict = {'name':f'Batch Runner/{label}/{task_name}',
                 'config': args.config,
                 'resume': None,
                 'device': args.device,
                 'data_dir': dataset}
    config = ConfigParser.from_args(args=args_dict, run_id='model') # 解析命令行参数
    
    train.main(config) # 训练模型
    torch.cuda.empty_cache() # 清理显存
    # 加载模型并进行解释
    model, config, data_loader = interpret.load_model(f'saved/models/Batch Runner/{label}/{task_name}/model', args, f'Batch Runner/{label}/{task_name}', 'casuality')
    interpret.main(model, config, data_loader, ground_truth) 
    torch.cuda.empty_cache()

def main(args):
    task_list = tasks[args.task]()
    label = datetime.now().strftime(r'%m%d_%H%M%S') # 获得当前时间
    for task_name, task_msg in task_list.items(): # 对每一个task使用runtask方法
        runtask(label, args, task_msg['dataset'], task_msg['groundtruth'], task_name)

    # 获取文件保存路径
    configJSON = read_json(args.config)
    save_dir = Path(configJSON['trainer']['save_dir'])

    # 汇总结果
    results = [] # 存储每个任务的评估结果
    for task_name in task_list:
        fname = f'{save_dir}/log/Batch Runner/{label}/{task_name}/casuality/info.log' # 查找每个任务的日志文件
        if os.path.exists(fname):
            with open(fname, 'r') as f: # 读取文件内容
                lines = f.readlines()
                result={ # 将结果保存到字典中
                    "Precision'": float(lines[-8].split(':')[-1][:-1]),
                    "Recall'": float(lines[-7].split(':')[-1][:-1]),
                    "F1'": float(lines[-6].split(':')[-1][:-1]),
                    "Precision": float(lines[-4].split(':')[-1][:-1]),
                    "Recall": float(lines[-3].split(':')[-1][:-1]),
                    "F1": float(lines[-2].split(':')[-1][:-1]),
                    "PoD": float(lines[-1].split(':')[-1][:-2])/100
                }
                results.append(result)
    df = pd.DataFrame(results, index=[i for i in range(1,len(results)+1)])
    summary_dir = save_dir / 'log' / 'Batch Runner' / label / 'summary.csv'
    df.to_csv(summary_dir) # 保存结果到csv文件中
    print("===================Summary===================")
    print('\t'+ df.to_string().replace('\n', '\n\t'))

if __name__=="__main__":
    args = argparse.ArgumentParser(description='CausalityInterpret')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-d', '--device', default="0", type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-t', '--task', default='fMRI', type=str,
                      help='task (default: fMRI)')
    args = args.parse_args()
    
    label = main(args)
