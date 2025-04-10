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
from utils import prepare_device
from logger.logger import get_logger
from model.RRP_model import RRP

SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


# 加载预训练的模型以及其配置
def load_model(path, args, name='Causality Detecting', run_id=None):
    config_path = path + '/config.json'
    checkpoint_path = path + '/model_best.pth'
    args_dict = {'name': name,
                 'config': config_path,
                 'device': args.device}
    config = args_config_analyse.from_args(args=args_dict, run_id=run_id) # 解析配置文件

    data_loader = config.init_obj('data_loader', module_data) # 根据配置项，初始化一个数据加载器对象
    config['data_loader']['args']['series_num']=data_loader.series_num
    config['data_loader']['args']['time_step']=data_loader.time_step
    config['data_loader']['args']['output_window']=data_loader.output_window
    
    model = config.init_obj('arch', module_arch, config) # 根据配置文件初始化模型

    model = model.cuda() # 把模型放到设备上
    checkpoint = torch.load(checkpoint_path, weights_only=False) # 加载训练好的模型
    model.load_state_dict(checkpoint['state_dict']) # 让模型加载参数
    return model, config, data_loader


def main(model, config, data_loader, gt, model_path):
    logger = get_logger('train')
    logger.info("===================开始运行模型结果评估===================")
    attribution_generator = RRP(model) # 创建RRP解释器
    logger.info("ground_truth:"+ (gt if gt else "None"))
    
    device = prepare_device() # 获取所使用的设备 
    columns = list(data_loader.df_data.columns) # 获取时间序列名称
    series_num = data_loader.series_num # 获取时间序列数量
    data = [timeslice[0] for timeslice in data_loader.dataset] # 获取时间序列数据
    label = [timeslice[1] for timeslice in data_loader.dataset] # 获取时间序列标签
    data = torch.tensor(np.array(data), dtype=torch.float).to(device)
    label = torch.tensor(np.array(label), dtype=torch.float).to(device)

    relA=[]
    relK=[]
    
    for interpreted_series in range(series_num): # 针对每一个时间序列进行解释
        rel_a, rel_k = attribution_generator.generate_RRP(data_loader.batch_size, data, interpreted_series) # 得到相关性分数
        # 将生成的相关性分数从 GPU 移动到 CPU，并转换为 NumPy 数组。
        relA.append(rel_a.detach().cpu().numpy()[interpreted_series])
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
    logger.info("===================因果关系结果===================")
    for e in ans:
        logger.info(f"{columns[e[0]]} causes {columns[e[1]]} with a delay of {e[2]} time steps.")

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
    csv_path = os.path.join(os.path.dirname(model_path), "causal_structure.csv")
    causal_df.to_csv(csv_path, index=False)
    logger.info(f"因果图结构已保存到: {csv_path}")

    # 评估因果关系
    allcauses={i:[] for i in range(len(columns))} # 时间序列因变量列表
    alldelays={} # 因果关系延迟列表
    for causal in ans:
        allcauses[causal[1]].append(causal[0])
        alldelays[(causal[1],causal[0])]=causal[2]

    if gt: # 如果提供了真实因果关系图，则进行评估
        logger.info("===================对比真实因果图之后因果关系的评估===================")
        FP, TP, FPdirect, TPdirect, FN, FPs, FPsdirect, TPs, TPsdirect, FNs, F1, F1direct = evaluate(logger, gt, allcauses, columns) # 得到模型的各项指标
        extendeddelays, readgt, extendedreadgt = getextendeddelays(gt, columns) # 获得因果时间延迟
        percentagecorrect = evaluatedelay(extendeddelays, alldelays, TPs, 1)*100 # 得到模型发现的时间延迟与真实延迟的匹配程度
        logger.info(f"正确发现的延迟百分比: {percentagecorrect}%")


# 如果这个脚本单独运行
if __name__ == '__main__':
    args = argparse.ArgumentParser(description='CausalityInterpret')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args = args.parse_args() 
    # 加载模型
    def render(args):
        model_path = 'saved/models/0410_152831/FMRI15/model'
        # model_path = 'saved/models/0410_104535'
        # gt_path = 'data/fMRI/sim1_gt_processed.csv'
        gt_path = None
        return load_model(model_path, args), gt_path, model_path
    (model, config, data_loader), gt, model_path = render(args)
    main(model, config, data_loader, gt, model_path) # 开始评估