from pathlib import Path
import torch
import numpy as np
import data_loader.data_loaders as module_data
import model.model as module_arch
from model.k_means import K_means_analyze
import argparse
import os
import pandas as pd
from copy import deepcopy
from utils.args_config_analyse import args_config_analyse
from evaluator.evaluator import evaluate, getextendeddelays, evaluatedelay
from logger.logger import get_logger, setup_logging
from model.RRP_model import RRP
from utils.util import init_obj_by_config, read_json, init_obj_by_config

SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


# 加载预训练的模型以及其配置
def load_model(config, model_path):
    data_loader = init_obj_by_config(config, 'data_loader', module_data) # 初始化数据加载器
    config['data_loader']['args']['series_num']=data_loader.series_num
    config['data_loader']['args']['time_step']=data_loader.time_step
    config['data_loader']['args']['output_window']=data_loader.output_window
    
    model = init_obj_by_config('arch', module_arch, config) # 根据config的配置项内容，初始化模型
    model = model.cuda() # 把模型放到设备上
    checkpoint = torch.load(model_path, weights_only=False) # 加载训练好的模型
    model.load_state_dict(checkpoint['state_dict']) # 让模型加载参数
    return model, data_loader


def main(model, config, data_loader, gt_path, log_path):
    setup_logging(log_path)
    interpret_logger = get_logger()
    print("===================开始运行模型结果评估===================")
    RRP_interpreter = RRP(model) # 创建RRP解释器
    print("已有的真实关系文件路径:"+ (gt_path if gt_path else "None"))
    
    columns = list(data_loader.df_data.columns) # 获取时间序列名称
    series_num = data_loader.series_num # 获取时间序列数量
    data = [timeslice[0] for timeslice in data_loader.dataset] # 获取时间序列数据
    label = [timeslice[1] for timeslice in data_loader.dataset] # 获取时间序列标签
    data = torch.tensor(np.array(data), dtype=torch.float).cuda()
    label = torch.tensor(np.array(label), dtype=torch.float).cuda()

    relA=[]
    relK=[]
    
    # 得到每一个时间序列的注意力相关性分数和卷积核相关性分数
    for interpreted_series in range(series_num): 
        rel_a, rel_k = RRP_interpreter.generate_RRP(data_loader.batch_size, data, interpreted_series) 
        relA.append(rel_a.detach().cpu().numpy()[interpreted_series]) # 将相关性分数从 GPU 移动到 CPU，并转换为 NumPy 数组。
        relk_align = deepcopy(rel_k.detach().cpu().numpy()[:,interpreted_series,-1,:])
        # relK[i][i][-1]是零向量，因为time_step数据不能用来预测未来本身的time_step
        relk_align[interpreted_series,:] = rel_k.detach().cpu().numpy()[interpreted_series,interpreted_series,-2,:]
        relK.append(relk_align)

    # 从配置文件得到的聚类参数
    m = config['explainer']['m']
    n = config['explainer']['n']
    assert m<n, "选择的前m个集群的数量必须小于n个集群的总数"
    ans = K_means_analyze(relA, relK, m, n, config['data_loader']['args']['time_step']) # 用K-means对相关性分数进行聚类

    # 打印因果关系结果
    interpret_logger.info("===================因果关系结果===================")
    for e in ans:
        interpret_logger.info(f"{columns[e[0]]} causes {columns[e[1]]} with a delay of {e[2]} time steps.")

    # 保存因果图结构到CSV文件
    causal_data = []
    for e in ans:
        causal_data.append({
            "source": columns[e[0]],  # 头结点
            "target": columns[e[1]],  # 尾结点
            "delay": e[2]             # 延迟时间
        })
    
    # 创建DataFrame
    causal_df = pd.DataFrame(causal_data)
    
    # 确定保存路径与原模型在同一目录
    csv_path = os.path.join(log_path, "causal_structure.csv")
    causal_df.to_csv(csv_path, index=False)

    # 评估因果关系
    allcauses={i:[] for i in range(len(columns))} # 时间序列因变量列表
    alldelays={} # 因果关系延迟列表
    for causal in ans:
        allcauses[causal[1]].append(causal[0])
        alldelays[(causal[1],causal[0])]=causal[2]

    if gt_path: # 如果提供了真实因果关系图，则进行评估
        interpret_logger.info("===================对比真实因果图之后因果关系的评估===================")
        FP, TP, FPdirect, TPdirect, FN, FPs, FPsdirect, TPs, TPsdirect, FNs, F1, F1direct = evaluate(interpret_logger, gt_path, allcauses, columns) # 得到模型的各项指标
        extendeddelays, readgt, extendedreadgt = getextendeddelays(gt_path, columns) # 获得因果时间延迟
        percentagecorrect = evaluatedelay(extendeddelays, alldelays, TPs, 1)*100 # 得到模型发现的时间延迟与真实延迟的匹配程度
        interpret_logger.info(f"正确发现的延迟百分比: {percentagecorrect}%")


# 如果这个脚本单独运行
if __name__ == '__main__':
    run_id = '0410_152831'
    log_path = Path('saved')/ run_id / 'model_interpret'
    model_path = Path('saved')/ run_id / 'model/model_best.pth'
    model_config_path = Path('config/config_FMRI.json')
    gt_path = None
    # gt_path = 'data/fMRI/sim1_gt_processed.csv'
    config = read_json(model_config_path) 
    model, data_loader = load_model(config, model_path)
    main(model, config, data_loader, gt_path, log_path) # 开始评估