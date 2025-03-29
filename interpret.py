import torch
import numpy as np
import data_loader.data_loaders as module_data
import model.model as module_arch
import model.loss as module_loss
import argparse
from copy import deepcopy
from parse_config import ConfigParser
from explainer.explainer import RRP
from evaluator.evaluator import evaluate, getextendeddelays, evaluatedelay
from utils import prepare_device
from sklearn.cluster import KMeans

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
                 'resume': None,
                 'device': args.device}
    config = ConfigParser.from_args(args=args_dict, run_id=run_id) # 解析配置文件

    logger = config.get_logger('train') # 获取日志记录器

    data_loader = config.init_obj('data_loader', module_data) # 根据配置项，初始化一个数据加载器对象
    valid_data_loader = data_loader.split_validation() # 分离出验证集（train同款）
    config['data_loader']['args']['series_num']=data_loader.series_num
    config['data_loader']['args']['time_step']=data_loader.time_step
    config['data_loader']['args']['output_window']=data_loader.output_window
    
    model = config.init_obj('arch', module_arch, config) # 根据配置文件初始化模型

    device, device_ids = prepare_device(config['n_gpu']) # 获取所使用的设备
    model = model.to(device) # 把模型放到设备上
    if len(device_ids) > 1: # 多GPU情况
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    checkpoint = torch.load(checkpoint_path) # 加载预训练好的模型
    model.load_state_dict(checkpoint['state_dict']) # 让模型加载参数
    return model, config, data_loader
    
# 使用K-means进行因果关系分析并构建因果图
def analyze(relA, relK, m, n, time_step):
    """
    
    Args:
        relA (List[torch.Tensor]): 每个时间序列的注意力矩阵的相关性分数。
        relK (List[torch.Tensor]): 每个时间序列的因果卷积核的相关性分数。
        m (int): 考虑的顶部聚类数量。
        n (int): 总聚类数量。
        time_step (int): 输入的时间步数。

    Returns:
        ans (List[Tuple[int, int, int]]): 表示因果图边的元组列表（原因、效果、滞后）
    """
    estimator = KMeans(n_clusters=n) # 搭建K-means模型
    ans = []
    # find causes of series i
    for i,relAi in enumerate(relA):
        if relAi.sum()==0.0: # all the weights to series i are zero
            continue
        data=np.array(relAi)
        estimator.fit(data.reshape(-1,1))
        cluster_labels = estimator.labels_
        cluster_centers = estimator.cluster_centers_
        cluster_centers = cluster_centers.reshape(-1)
        largest_m_clusters = np.argsort(cluster_centers)[-m:]
        for j in range(len(relAi)):
            if cluster_labels[j] in largest_m_clusters:
                relKij = relK[i][j]
                indices = np.argsort(-1 * relKij)
                ans.append((j,i,time_step-1-indices[0]))
    return ans

# 评估因果关系
def eval(logger, gt, allcauses, alldelays, columns):
    '''
    logger：日志记录器。
    gt：真实因果关系图。
    allcauses：检测到的因果关系。
    alldelays：检测到的延迟。
    columns：时间序列的名称。
    '''
    # 使用evaluator里的函数来对模型进行评估
    FP, TP, FPdirect, TPdirect, FN, FPs, FPsdirect, TPs, TPsdirect, FNs, F1, F1direct = evaluate(logger, gt, allcauses, columns) # 得到模型的各项指标
    extendeddelays, readgt, extendedreadgt = getextendeddelays(gt, columns) # 获得因果时间延迟
    percentagecorrect = evaluatedelay(extendeddelays, alldelays, TPs, 1)*100 # 得到模型发现的时间延迟与真实延迟的匹配程度
    logger.info(f"Percentage of delays that are correctly discovered: {percentagecorrect}%")

def main(model, config, data_loader, gt, bigdata=False):
    logger = config.get_logger('train')
    logger.info("===================Running===================")
    attribution_generator = RRP(model) # 创建RRP解释器
    logger.info("ground_truth:"+ (gt if gt else "None"))
    
    device, device_ids = prepare_device(config['n_gpu']) # 获取所使用的设备 
    columns = list(data_loader.df_data.columns) # 获取时间序列名称
    series_num = data_loader.series_num # 获取时间序列数量
    data = [timeslice[0] for timeslice in data_loader.dataset] # 获取时间序列数据
    label = [timeslice[1] for timeslice in data_loader.dataset] # 获取时间序列标签
    data = torch.tensor(np.array(data), dtype=torch.float).to(device)
    label = torch.tensor(np.array(label), dtype=torch.float).to(device)
    if bigdata: # 如果是bigdata，则减少一下数据量
        data = data.mean(0).unsqueeze(0)
        label = label.mean(0).unsqueeze(0)

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
    ans = analyze(relA, relK, m, n, config['data_loader']['args']['time_step']) # 用K-means对相关性分数进行聚类

    # 打印因果关系结果
    logger.info("===================Results===================")
    for e in ans:
        logger.info(f"{columns[e[0]]} causes {columns[e[1]]} with a delay of {e[2]} time steps.")

    # 评估因果关系
    allcauses={i:[] for i in range(len(columns))} # 时间序列因变量列表
    alldelays={} # 因果关系延迟列表
    for causal in ans:
        allcauses[causal[1]].append(causal[0])
        alldelays[(causal[1],causal[0])]=causal[2]
    if gt: # 如果提供了真实因果关系图，则进行评估
        logger.info("===================Evaluation===================")
        eval(logger, gt, allcauses, alldelays, columns)

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='CausalityInterpret')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    
    args = args.parse_args()
    # 加载模型
    def render(args):
        return load_model('saved/models/Causality Discovery/0714_134931', args), 'data/fMRI/sim1_gt_processed.csv', False
    (model, config, data_loader), gt, bigdata = render(args)
    main(model, config, data_loader, gt, bigdata) # 开始评估
