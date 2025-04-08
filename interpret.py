import torch
import numpy as np
import data_loader.data_loaders as module_data
import model.model as module_arch
import argparse
from copy import deepcopy
from utils.args_config_analyse import args_config_analyse
from evaluator.evaluator import evaluate, getextendeddelays, evaluatedelay
from utils import prepare_device
from sklearn.cluster import KMeans

SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

class RRP:
    def __init__(self, model):
        self.model = model
        self.model.eval() # 模型设置为评估模式

    # 收集并计算模型中每个层的因果关系分数，包括注意力矩阵和卷积核的因果分数。
    def generate_RRP(self, batch_size, input, interpreted_series):
        inputs = torch.split(input, batch_size)
        relAs, relKs = [], [] 
        for data in inputs: # 多批次数据叠加
            relA, relK = self._generate_RRP(data, interpreted_series)
            relAs.append(relA)
            relKs.append(relK)
        relA = torch.stack(relAs).mean(0)
        relK = torch.stack(relKs).mean(0)
        return relA, relK # 返回注意矩阵和卷积核的因果分数
        
    # 为单个批次的输入数据生成因果关系分数。
    def _generate_RRP(self, input, interpreted_series):
        """
        input (torch.Tensor):输入数据张量[total_batch， input_window, series_num, feature_dim]
        interpreted_series (int)：被解释时间序列的索引的序号。
        """
        output = self.model(input) # 得到模型输出
        
        one_hot = torch.zeros_like(output, dtype=torch.float).to(output.device) # 创建一个与输出形状相同的 one-hot 张量，仅在序列序号所对应的位置设置为 1。
        one_hot[:,:,interpreted_series,:] = 1
        one_hot_vector = one_hot.clone() # 克隆 one-hot 张量，并设置其需要计算梯度。
        one_hot.requires_grad_(True)
        one_hot = torch.sum(one_hot * output) # 计算 one-hot 张量与模型输出的点积。
        
        self.model.zero_grad()
        one_hot.backward(retain_graph=True)
        self.model.relprop(one_hot_vector)  # 调用模型的 relprop 方法，将 one-hot 张量的梯度传播回输入层，计算每个输入特征对输出的贡献

        # 收集因果关系分数
        relAs=[] # 注意力矩阵因果关系分数
        relKs=[] # 卷积核因果关系分数
        for layer in self.model.encoder.layers: # 遍历模型的编码器层，收集每个层的因果关系分数。
            # 梯度调制
            relA = layer.attention.attention.get_rel() * torch.abs(layer.attention.attention.get_grad())
            relK = layer.attention.Wv.get_rel() * torch.abs(layer.attention.Wv.get_grad())

            # w/o interpretation
            # relA = layer.attention.attention.get_wgt()
            # relK = layer.attention.Wv.get_wgt()

            relA = relA.clamp(min=0)        # 只考虑正向因果分数
            relK = relK.clamp(min=0)        
            relAs.append(relA.mean((0,1)))  # mean for sample and head
            relKs.append(relK.mean(0))      # mean for head
        # 将所有层的分数堆叠起来，并计算它们的乘积，得到最终的因果关系分数。
        relA = torch.stack(relAs).prod(0)   
        relK = torch.stack(relKs).prod(0)  
        return relA, relK


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

    device = prepare_device() # 获取所使用的设备
    model = model.to(device) # 把模型放到设备上
    checkpoint = torch.load(checkpoint_path, weights_only=False) # 加载预训练好的模型
    model.load_state_dict(checkpoint['state_dict']) # 让模型加载参数
    return model, config, data_loader
    
# 使用K-means进行因果关系分析并构建因果图
def analyze(relA, relK, m, n, time_step):
    """
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
    logger.info(f"正确发现的延迟百分比: {percentagecorrect}%")


def main(model, config, data_loader, gt):
    logger = config.get_logger('train')
    logger.info("===================开始运行interpret===================")
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
    ans = analyze(relA, relK, m, n, config['data_loader']['args']['time_step']) # 用K-means对相关性分数进行聚类

    # 打印因果关系结果
    logger.info("===================因果关系结果===================")
    for e in ans:
        logger.info(f"{columns[e[0]]} causes {columns[e[1]]} with a delay of {e[2]} time steps.")

    # 评估因果关系
    allcauses={i:[] for i in range(len(columns))} # 时间序列因变量列表
    alldelays={} # 因果关系延迟列表
    for causal in ans:
        allcauses[causal[1]].append(causal[0])
        alldelays[(causal[1],causal[0])]=causal[2]

    if gt: # 如果提供了真实因果关系图，则进行评估
        logger.info("===================对比真实因果图之后因果关系的评估===================")
        eval(logger, gt, allcauses, alldelays, columns)


# 如果这个脚本单独运行
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
