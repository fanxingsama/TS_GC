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

# python runner.py -c config/config_fMRI.json

def construct_demo():
    task_list = {}
    for i in [15]:
        task_list[f'fMRI{i}'] = {
            'dataset': f"../fMRI/timeseries{i}.csv",
            'groundtruth': f"../fMRI/sim{i}_gt_processed.csv"
        }
    return task_list

def construct_fMRI():
    task_list = {}
    for i in range(1,29):
        task_list[f'fMRI{i}'] = {
            'dataset': f"../fMRI/timeseries{i}.csv",
            'groundtruth': f"../fMRI/sim{i}_gt_processed.csv"
        }
    return task_list

# def construct_basic_diamond():
#     task_list = {}
#     for i in range(10):
#         task_list[f'diamond{i}'] = {
#             'dataset': f"data/basic/diamond/data_{i}.csv",
#             'groundtruth': f"data/basic/diamond/groundtruth.csv"
#         }
#     return task_list

# def construct_basic_mediator():
#     task_list = {}
#     for i in range(10):
#         task_list[f'mediator{i}'] = {
#             'dataset': f"data/basic/mediator/data_{i}.csv",
#             'groundtruth': f"data/basic/mediator/groundtruth.csv"
#         }
#     return task_list

# def construct_basic_v():
#     task_list = {}
#     for i in range(10):
#         task_list[f'v{i}'] = {
#             'dataset': f"data/basic/v/data_{i}.csv",
#             'groundtruth': f"data/basic/v/groundtruth.csv"
#         }
#     return task_list

# def construct_basic_fork():
#     task_list = {}
#     for i in range(10):
#         task_list[f'fork{i}'] = {
#             'dataset': f"data/basic/fork/data_{i}.csv",
#             'groundtruth': f"data/basic/fork/groundtruth.csv"
#         }
#     return task_list

# def construct_lorenz():
#     task_list = {}
#     for i in range(10):
#         task_list[f'lorenz{i}'] = {
#             'dataset': f"data/lorenz96/timeseries{i}.csv",
#             'groundtruth': f"data/lorenz96/groundtruth.csv"
#         }
#     return task_list

tasks={
    'demo': construct_demo,
    'fMRI': construct_fMRI,
    # 'diamond': construct_basic_diamond,
    # 'mediator': construct_basic_mediator,
    # 'v': construct_basic_v,
    # 'fork': construct_basic_fork,
    # 'lorenz':construct_lorenz
}    

def runtask(label, args, dataset, ground_truth, task_name):
    # fix random seeds for reproducibility
    SEED = 123
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(SEED)
    args_dict = {'name':f'Batch Runner/{label}/{task_name}',
                 'config': args.config,
                 'resume': None,
                 'device': args.device,
                 'data_dir': dataset}
    config = ConfigParser.from_args(args=args_dict, run_id='model')
    train.main(config)
    torch.cuda.empty_cache()
    model, config, data_loader = interpret.load_model(f'saved/models/Batch Runner/{label}/{task_name}/model', args, f'Batch Runner/{label}/{task_name}', 'casuality')
    interpret.main(model, config, data_loader, ground_truth)
    torch.cuda.empty_cache()

def main(args):
    import re
    import os
    
    # 1. 读取配置文件中的数据集路径
    configJSON = read_json(args.config)
    data_dir = configJSON['data_loader']['args']['data_dir']
    dataset_name = os.path.basename(data_dir).replace('.csv', '')
    
    # 2. 尝试推导 groundtruth 的路径 (仅为了能够输出准确率，如果找不到则为 None)
    match = re.search(r'timeseries(\d+)', dataset_name)
    if match:
        idx = match.group(1)
        gt_path = data_dir.replace(f'timeseries{idx}.csv', f'sim{idx}_gt_processed.csv')
    else:
        gt_path = None
        
    # 确认 gt_path 是否真的存在
    if gt_path and not os.path.exists(gt_path):
        gt_path = None

    label = datetime.now().strftime(r'%m%d_%H%M%S')
    
    print(f"===========================================================")
    print(f"🚀 开始单任务运行模式")
    print(f"📂 数据集: {dataset_name} ({data_dir})")
    print(f"🎯 GroundTruth: {gt_path if gt_path else '未找到'}")
    print(f"===========================================================\n")
    
    # 3. 直接运行这一个任务
    runtask(label, args, data_dir, gt_path, dataset_name)

    # 4. 运行结束后，尝试提取并打印评估结果摘要
    save_dir = Path(configJSON['trainer']['save_dir'])
    fname = f'{save_dir}/log/Batch Runner/{label}/{dataset_name}/casuality/info.log'
    
    if os.path.exists(fname):
        with open(fname, 'r') as f:
            lines = f.readlines()
            try:
                result={
                    "Precision'": float(lines[-8].split(':')[-1][:-1]),
                    "Recall'": float(lines[-7].split(':')[-1][:-1]),
                    "F1'": float(lines[-6].split(':')[-1][:-1]),
                    "Precision": float(lines[-4].split(':')[-1][:-1]),
                    "Recall": float(lines[-3].split(':')[-1][:-1]),
                    "F1": float(lines[-2].split(':')[-1][:-1]),
                    "PoD": float(lines[-1].split(':')[-1][:-2])/100
                }
                df = pd.DataFrame([result], index=[1])
                print("\n===================Summary===================")
                print('\t'+ df.to_string().replace('\n', '\n\t'))
            except Exception:
                pass # 如果没有 groundtruth，日志里就没有这些指标，直接跳过即可
            
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
