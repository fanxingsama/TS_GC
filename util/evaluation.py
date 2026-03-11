import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score

def load_edges(file_path):
    """
    读取无表头的文件，第一列为因，第二列为果
    支持逗号(csv)或制表符/空格分隔(txt)
    返回边集合 (set of tuples) 以及所有节点的集合
    """
    try:
        # 尝试以逗号分隔读取
        df = pd.read_csv(file_path, header=None)
        if df.shape[1] < 2:
            # 如果解析失败，尝试以空格/制表符分隔读取
            df = pd.read_csv(file_path, header=None, delim_whitespace=True)
    except Exception as e:
        print(f"读取文件 {file_path} 失败: {e}")
        return set(), set()
    
    edges = set(tuple(x) for x in df.iloc[:, :2].values)
    
    # 提取所有出现过的节点
    nodes = set(df.iloc[:, 0].unique()).union(set(df.iloc[:, 1].unique()))
    return edges, nodes

def calculate_shd(true_edges, pred_edges, nodes):
    """
    计算 SHD (Structural Hamming Distance)
    包含了缺失边 (Missing)、多余边 (Extra) 以及反向边 (Reversed) 的情况。
    反向边计为 1 次错误（而不是缺失+多余的 2 次错误）。
    """
    shd = 0
    visited_pairs = set()
    
    for i in nodes:
        for j in nodes:
            # 针对时间序列自环 (Self-loops) 的情况
            if i == j:
                if ((i, i) in true_edges) != ((i, i) in pred_edges):
                    shd += 1
                continue
            
            # 无向节点对，避免重复遍历
            pair = frozenset([i, j])
            if pair in visited_pairs:
                continue
            visited_pairs.add(pair)
            
            t_ij = (i, j) in true_edges
            t_ji = (j, i) in true_edges
            p_ij = (i, j) in pred_edges
            p_ji = (j, i) in pred_edges
            
            # 如果两个图在 i 和 j 之间的连接完全一致
            if t_ij == p_ij and t_ji == p_ji:
                continue
                
            # 判断是否为单纯的反向边 (Reversal)
            # 真实为 i->j，预测为 j->i，且没有双向边
            if (t_ij and not t_ji) and (not p_ij and p_ji):
                shd += 1
            # 真实为 j->i，预测为 i->j，且没有双向边
            elif (not t_ij and t_ji) and (p_ij and not p_ji):
                shd += 1
            else:
                # 其他情况：单纯的加边或减边
                shd += abs(t_ij - p_ij) + abs(t_ji - p_ji)
                
    return shd

def evaluate_causal_discovery(pred_file, true_file):
    # 1. 加载数据
    pred_edges, pred_nodes = load_edges(pred_file)
    true_edges, true_nodes = load_edges(true_file)
    
    # 2. 获取所有的节点集合（构建完整的邻接矩阵参考框架）
    all_nodes = list(pred_nodes.union(true_nodes))
    if not all_nodes:
        print("未检测到任何节点，请检查文件格式。")
        return
    
    print(f"检测到 {len(all_nodes)} 个唯一节点，总计 {len(all_nodes)**2} 个可能的有向边(含自环)。")
    print(f"真实边数量: {len(true_edges)}")
    print(f"预测边数量: {len(pred_edges)}")
    
    # 3. 构建 Flattened Arrays 用于计算 F1 和 AUROC
    y_true = []
    y_pred = []
    
    for i in all_nodes:
        for j in all_nodes:
            y_true.append(1 if (i, j) in true_edges else 0)
            y_pred.append(1 if (i, j) in pred_edges else 0)
            
    # 4. 计算指标
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall    = recall_score(y_true, y_pred, zero_division=0)
    
    # 如果 y_true 只有 0 或只有 1，AUROC 将无法计算
    try:
        auroc = roc_auc_score(y_true, y_pred)
    except ValueError:
        auroc = float('nan')
        print("警告: 真实数据中仅包含单一类别（全是边或全无边），无法计算 AUROC。")
        
    shd = calculate_shd(true_edges, pred_edges, all_nodes)
    
    # 5. 输出结果
    print("-" * 30)
    print("=== 评估指标 ===")
    print(f"AUROC : {auroc:.4f}" if not np.isnan(auroc) else "AUROC : NaN")
    print(f"F1    : {f1:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"SHD   : {shd}")
    print("-" * 30)

if __name__ == "__main__":
    
    # 请将以下路径替换为你自己的文件路径
    TRUE_FILE = "./compare_model_matrix/linear/causal_linear.csv"
    # PRED_FILE = "../compare_model/CausalFormer/csv/CausalFormer_timeseries6.csv" # CausalFormer预测矩阵
    PRED_FILE = "./compare_model_matrix/linear/GVAR.csv" # TCDF预测矩阵
    # PRED_CSV = "../saved/2026-03-02_02_15-19-34/GC_matrix_constrained.csv"  # TS-GC预测矩阵
    
    evaluate_causal_discovery(PRED_FILE, TRUE_FILE)