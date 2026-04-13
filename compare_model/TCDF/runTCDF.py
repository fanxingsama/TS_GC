import TCDF
import argparse
import torch
import pandas as pd
import numpy as np
import networkx as nx
import pylab
import copy
import matplotlib.pyplot as plt
import os
import sys

# os.chdir(os.path.dirname(sys.argv[0])) #uncomment this line to run in VSCode

def check_positive(value):
    """Checks if argument is positive integer (larger than zero)."""
    ivalue = int(value)
    if ivalue <= 0:
         raise argparse.ArgumentTypeError("%s should be positive" % value)
    return ivalue

def check_zero_or_positive(value):
    """Checks if argument is positive integer (larger than or equal to zero)."""
    ivalue = int(value)
    if ivalue < 0:
         raise argparse.ArgumentTypeError("%s should be positive" % value)
    return ivalue

class StoreDictKeyPair(argparse.Action):
    """Creates dictionary containing datasets as keys and ground truth files as values."""
    def __call__(self, parser, namespace, values, option_string=None):
        my_dict = {}
        for kv in values.split(","):
            k,v = kv.split("=")
            my_dict[k] = v
        setattr(namespace, self.dest, my_dict)

def getextendeddelays(gtfile, columns):
    """Collects the total delay of indirect causal relationships. 
    Modified to support variable names and optional delay column."""
    
    # 确保列名列表是字符串类型，以便匹配
    columns_list = [str(c) for c in columns]
    # 读取GT文件，不设表头
    gtdata = pd.read_csv(gtfile, header=None)

    readgt = dict()
    for k in range(len(columns_list)):
        readgt[k] = []
    
    num_cols = gtdata.shape[1]
    pairdelays = dict()
    gtnrrelations = 0
    
    for i in range(len(gtdata)):
        # 获取原始输入：第0列为因，第1列为果
        cau_raw = gtdata.iloc[i, 0]
        eff_raw = gtdata.iloc[i, 1]
        # 如果有第3列则读取延迟，否则默认为0
        delay = gtdata.iloc[i, 2] if num_cols >= 3 else 0
        
        # 辅助函数：将变量名或字符串索引转换为整数索引
        def to_index(val):
            # 如果是整数类型直接返回
            if isinstance(val, (int, np.integer)):
                return int(val)
            # 如果是字符串变量名（如 'x0'），从 columns 列表中查找其索引
            s_val = str(val).strip()
            if s_val in columns_list:
                return columns_list.index(s_val)
            # 如果字符串本身就是数字（如 '0'），转换成 int
            if s_val.isdigit():
                return int(s_val)
            raise ValueError(f"Ground Truth 中的值 '{val}' 在数据列名中未找到，也不是有效的索引。")

        try:
            cause_idx = to_index(cau_raw)
            effect_idx = to_index(eff_raw)
            
            # 填充数据结构（内部逻辑仍使用索引以保证计算效率）
            readgt[effect_idx].append(cause_idx)
            pairdelays[(effect_idx, cause_idx)] = delay
            gtnrrelations += 1
        except ValueError as e:
            print(f"警告: 跳过 GT 文件第 {i+1} 行: {e}")
            continue

    # --- 以下为原函数的拓扑路径搜索逻辑，保持不变 ---
    g = nx.DiGraph()
    g.add_nodes_from(readgt.keys())
    for e in readgt:
        cs = readgt[e]
        for c in cs:
            g.add_edge(c, e)

    extendedreadgt = copy.deepcopy(readgt)
    
    for c1 in range(len(columns)):
        for c2 in range(len(columns)):
            paths = list(nx.all_simple_paths(g, c1, c2, cutoff=2))
            if len(paths)>0:
                for path in paths:
                    for p in path[:-1]:
                        if p not in extendedreadgt[path[-1]]:
                            extendedreadgt[path[-1]].append(p)
                            
    extendedgtdelays = dict()
    for effect in extendedreadgt:
        causes = extendedreadgt[effect]
        for cause in causes:
            if (effect, cause) in pairdelays:
                delay = pairdelays[(effect, cause)]
                extendedgtdelays[(effect, cause)]=[delay]
            else:
                paths = list(nx.all_simple_paths(g, cause, effect, cutoff=2))
                extendedgtdelays[(effect, cause)]=[]
                for p in paths:
                    d=0
                    for j in range(len(p)-1):
                        d+=pairdelays[(p[j+1], p[j])]
                    extendedgtdelays[(effect, cause)].append(d)

    return extendedgtdelays, readgt, extendedreadgt

def evaluate(gtfile, validatedcauses, columns):
    """Evaluates the results of TCDF by comparing it to the ground truth graph, and calculating precision, recall and F1-score. F1'-score, precision' and recall' include indirect causal relationships."""
    extendedgtdelays, readgt, extendedreadgt = getextendeddelays(gtfile, columns)
    FP=0
    FPdirect=0
    TPdirect=0
    TP=0
    FN=0
    FPs = []
    FPsdirect = []
    TPsdirect = []
    TPs = []
    FNs = []
    for key in readgt:
        for v in validatedcauses[key]:
            if v not in extendedreadgt[key]:
                FP+=1
                FPs.append((key,v))
            else:
                TP+=1
                TPs.append((key,v))
            if v not in readgt[key]:
                FPdirect+=1
                FPsdirect.append((key,v))
            else:
                TPdirect+=1
                TPsdirect.append((key,v))
        for v in readgt[key]:
            if v not in validatedcauses[key]:
                FN+=1
                FNs.append((key, v))
          
    print("Total False Positives': ", FP)
    print("Total True Positives': ", TP)
    print("Total False Negatives: ", FN)
    print("Total Direct False Positives: ", FPdirect)
    print("Total Direct True Positives: ", TPdirect)
    print("TPs': ", TPs)
    print("FPs': ", FPs)
    print("TPs direct: ", TPsdirect)
    print("FPs direct: ", FPsdirect)
    print("FNs: ", FNs)
    precision = recall = 0.

    if float(TP+FP)>0:
        precision = TP / float(TP+FP)
    print("Precision': ", precision)
    if float(TP + FN)>0:
        recall = TP / float(TP + FN)
    print("Recall': ", recall)
    if (precision + recall) > 0:
        F1 = 2 * (precision * recall) / (precision + recall)
    else:
        F1 = 0.
    print("F1' score: ", F1,"(includes direct and indirect causal relationships)")

    precision = recall = 0.
    if float(TPdirect+FPdirect)>0:
        precision = TPdirect / float(TPdirect+FPdirect)
    print("Precision: ", precision)
    if float(TPdirect + FN)>0:
        recall = TPdirect / float(TPdirect + FN)
    print("Recall: ", recall)
    if (precision + recall) > 0:
        F1direct = 2 * (precision * recall) / (precision + recall)
    else:
        F1direct = 0.
    print("F1 score: ", F1direct,"(includes only direct causal relationships)")
    return FP, TP, FPdirect, TPdirect, FN, FPs, FPsdirect, TPs, TPsdirect, FNs, F1, F1direct

def evaluatedelay(extendedgtdelays, alldelays, TPs, receptivefield):
    """Evaluates the delay discovery of TCDF by comparing the discovered time delays with the ground truth."""
    zeros = 0
    total = 0.
    for i in range(len(TPs)):
        tp=TPs[i]
        discovereddelay = alldelays[tp]
        gtdelays = extendedgtdelays[tp]
        for d in gtdelays:
            if d <= receptivefield:
                total+=1.
                error = d - discovereddelay
                if error == 0:
                    zeros+=1
                
            else:
                next
           
    if zeros==0:
        return 0.
    else:
        return zeros/float(total)


def runTCDF(datafile):
    """Loops through all variables in a dataset and return the discovered causes, time delays, losses, attention scores and variable names."""
    df_data = pd.read_csv(datafile)

    allcauses = dict()
    alldelays = dict()
    allreallosses=dict()
    allscores=dict()

    columns = list(df_data)
    for c in columns:
        idx = df_data.columns.get_loc(c)
        causes, causeswithdelay, realloss, scores = TCDF.findcauses(c, cuda=cuda, epochs=nrepochs, 
        kernel_size=kernel_size, layers=levels, log_interval=loginterval, 
        lr=learningrate, optimizername=optimizername,
        seed=seed, dilation_c=dilation_c, significance=significance, file=datafile)

        allscores[idx]=scores
        allcauses[idx]=causes
        alldelays.update(causeswithdelay)
        allreallosses[idx]=realloss

    return allcauses, alldelays, allreallosses, allscores, columns

# def plotgraph(stringdatafile,alldelays,columns):
#     """Plots a temporal causal graph showing all discovered causal relationships annotated with the time delay between cause and effect."""
#     G = nx.DiGraph()
#     for c in columns:
#         G.add_node(c)
#     for pair in alldelays:
#         p1,p2 = pair
#         nodepair = (columns[p2], columns[p1])

#         G.add_edges_from([nodepair],weight=alldelays[pair])
    
#     edge_labels=dict([((u,v,),d['weight'])
#                     for u,v,d in G.edges(data=True)])
    
#     pos=nx.circular_layout(G)
#     nx.draw_networkx_edge_labels(G,pos,edge_labels=edge_labels)
#     nx.draw(G,pos, node_color = 'white', edge_color='black',node_size=1000,with_labels = True)
#     ax = plt.gca()
#     ax.collections[0].set_edgecolor("#000000") 

#     pylab.show()

def main(datafiles, evaluation):
    if evaluation:
        totalF1direct = [] #contains F1-scores of all datasets
        totalF1 = [] #contains F1'-scores of all datasets

        receptivefield=1
        for l in range(0, levels):
            receptivefield+=(kernel_size-1) * dilation_c**(l)

    for datafile in datafiles.keys(): 
        stringdatafile = str(datafile)
        if '/' in stringdatafile:
            stringdatafile = str(datafile).rsplit('/', 1)[1]
        
        print("\n Dataset: ", stringdatafile)

        # run TCDF
        allcauses, alldelays, allreallosses, allscores, columns = runTCDF(datafile) #results of TCDF containing indices of causes and effects

        print("\n===================Results for", stringdatafile,"==================================")
        
        print("===================================修改开始===============================================")
        # for pair in alldelays:
        #     print(columns[pair[1]], "causes", columns[pair[0]],"with a delay of",alldelays[pair],"time steps.")
        
        import csv
        # 去掉原本的 .csv 后缀，提取出纯数据集名
        dataset_name = stringdatafile.replace('.csv', '')
        # 按照“TCDF_数据集名_causal.csv”的格式拼接文件名
        csv_filename = f"TCDF_{dataset_name}_causal.csv"

        with open(csv_filename, mode='w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            for pair in alldelays:
                # 保持原本的终端打印输出
                print(columns[pair[1]], "causes", columns[pair[0]],"with a delay of",alldelays[pair],"time steps.")
                # 将因果关系写入CSV文件：第一列是因 (columns[pair[1]])，第二列是果 (columns[pair[0]])
                writer.writerow([columns[pair[1]], columns[pair[0]], 1])

        print(f"因果关系已成功保存至: {csv_filename}")

        print("===================================修改结束===============================================")

        if evaluation:
            # evaluate TCDF by comparing discovered causes with ground truth
            print("\n===================Evaluation for", stringdatafile,"===============================")
            FP, TP, FPdirect, TPdirect, FN, FPs, FPsdirect, TPs, TPsdirect, FNs, F1, F1direct = evaluate(datafiles[datafile], allcauses, columns)
            totalF1.append(F1)
            totalF1direct.append(F1direct)

            # evaluate delay discovery
            extendeddelays, readgt, extendedreadgt = getextendeddelays(datafiles[datafile], columns)
            percentagecorrect = evaluatedelay(extendeddelays, alldelays, TPs, receptivefield)*100
            print("Percentage of delays that are correctly discovered: ", percentagecorrect,"%")
            
        print("==================================================================================")
        
        # if args.plot:
        #     plotgraph(stringdatafile, alldelays, columns)

    # In case of multiple datasets, calculate average F1-score over all datasets and standard deviation
    if len(datafiles.keys())>1 and evaluation:  
        print("\nOverall Evaluation: \n")      
        print("F1' scores: ")
        for f in totalF1:
            print(f)
        print("Average F1': ", np.mean(totalF1))
        print("Standard Deviation F1': ", np.std(totalF1),"\n")
        print("F1 scores: ")
        for f in totalF1direct:
            print(f)
        print("Average F1: ", np.mean(totalF1direct))
        print("Standard Deviation F1: ", np.std(totalF1direct))




parser = argparse.ArgumentParser(description='TCDF: Temporal Causal Discovery Framework')

parser.add_argument('--cuda', action="store_true", default=False, help='Use CUDA (GPU) (default: False)')
parser.add_argument('--epochs', type=check_positive, default=1000, help='Number of epochs (default: 1000)')
parser.add_argument('--kernel_size', type=check_positive, default=4, help='Size of kernel, i.e. window size. Maximum delay to be found is kernel size - 1. Recommended to be equal to dilation coeffient (default: 4)')
parser.add_argument('--hidden_layers', type=check_zero_or_positive, default=0, help='Number of hidden layers in the depthwise convolution (default: 0)') 
parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate (default: 0.01)')
parser.add_argument('--optimizer', type=str, default='Adam', choices=['Adam', 'RMSprop'], help='Optimizer to use (default: Adam)')
parser.add_argument('--log_interval', type=check_positive, default=500, help='Epoch interval to report loss (default: 500)')
parser.add_argument('--seed', type=check_positive, default=1111, help='Random seed (default: 1111)')
parser.add_argument('--dilation_coefficient', type=check_positive, default=4, help='Dilation coefficient, recommended to be equal to kernel size (default: 4)')
parser.add_argument('--significance', type=float, default=0.8, help="Significance number stating when an increase in loss is significant enough to label a potential cause as true (validated) cause. See paper for more details (default: 0.8)")
parser.add_argument('--plot', action="store_true", default=False, help='Show causal graph (default: False)')
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('--ground_truth',action=StoreDictKeyPair, help='Provide dataset(s) and the ground truth(s) to evaluate the results of TCDF. Argument format: DataFile1=GroundtruthFile1,Key2=Value2,... with a key for each dataset containing multivariate time series (required file format: csv, a column with header for each time series) and a value for the corresponding ground truth (required file format: csv, no header, index of cause in first column, index of effect in second column, time delay between cause and effect in third column)')
group.add_argument('--data', nargs='+', help='(Path to) one or more datasets to analyse by TCDF containing multiple time series. Required file format: csv with a column (incl. header) for each time series')

args = parser.parse_args()

print("Arguments:", args)

if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, you should probably run with --cuda to speed up training.")
if args.kernel_size != args.dilation_coefficient:
    print("WARNING: The dilation coefficient is not equal to the kernel size. Multiple paths can lead to the same delays. Set kernel_size equal to dilation_c to have exaxtly one path for each delay.")

kernel_size = args.kernel_size
levels = args.hidden_layers+1
nrepochs = args.epochs
learningrate = args.learning_rate
optimizername = args.optimizer
dilation_c = args.dilation_coefficient
loginterval = args.log_interval
seed=args.seed
cuda=args.cuda
significance=args.significance

if args.ground_truth is not None:
    datafiles = args.ground_truth
    main(datafiles, evaluation=True)

else:
    datafiles = dict()
    for dataset in args.data:
        datafiles[dataset]=""
    main(datafiles, evaluation=False)


# python runTCDF.py --ground_truth ../fMRI/timeseries6.csv=../fMRI/sim6_gt_processed.csv  --epochs 500 --log_interval 250 --plot
# python runTCDF.py --ground_truth time_series_nolinear.csv=causal_nolinear.csv  --epochs 500 --log_interval 250 --plot
# python runTCDF.py --ground_truth ./SNR/time_series_nolinear_snr20.csv=./SNR/causal_nolinear.csv  --epochs 500 --log_interval 250 --plot