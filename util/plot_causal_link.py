from matplotlib import rcParams
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import networkx as nx
import os

rcParams['font.family'] = 'SimHei'
rcParams['axes.unicode_minus'] = False

def save_causal_links(csv_path, img_save_path=None, series_names=None):
    """
    绘制美化的格兰杰因果关系图 (力导向布局 + 权重映射)
    """
    # 1. 初始化图
    G = nx.DiGraph()
    
    # 添加所有节点
    if series_names:
        G.add_nodes_from(series_names)
        
    # 2. 读取数据
    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
        try:
            df = pd.read_csv(csv_path, header=None)
            for _, row in df.iterrows():
                source, target, weight = row[0], row[1], float(row[2])
                G.add_edge(source, target, weight=weight)
        except Exception as e:
            print(f"读取CSV出错: {e}")
            return
    
    nodes = list(G.nodes())
    if not nodes:
        print("无节点可绘制")
        return

    # 3. 计算布局 (核心改变：使用力导向图 spring_layout)
    # k值决定节点间距，k越大节点越分散
    # seed保证每次运行图的形状一样
    pos = nx.spring_layout(G, k=0.8, iterations=50, seed=42) 

    plt.figure(figsize=(12, 10))
    
    # 4. 节点样式设计
    # 计算每个节点的度 (入度+出度)，用于决定节点大小和颜色
    d = dict(G.degree)
    node_sizes = [v * 100 + 300 for v in d.values()] # 基础大小300，连接越多越大
    node_colors = [v for v in d.values()] # 颜色映射值

    # 绘制节点
    # cmap='YlGnBu' 是颜色条，可以选择 'coolwarm', 'viridis' 等
    nodes_draw = nx.draw_networkx_nodes(G, pos, 
                                        node_size=node_sizes, 
                                        node_color=node_colors, 
                                        cmap=plt.cm.YlGnBu, 
                                        alpha=0.9,
                                        edgecolors='grey')
    
    # 5. 边样式设计
    edges = G.edges(data=True)
    if edges:
        weights = [data['weight'] for _, _, data in edges]
        # 归一化权重用于设置边的粗细 (1到4之间)
        max_w = max(weights) if weights else 1
        width = [(w / max_w) * 3 + 0.5 for w in weights]
        # 归一化权重用于设置透明度
        edge_colors = [plt.cm.Greys(w/max_w * 0.5 + 0.3) for w in weights] # 越强越黑

        # 绘制边 (使用曲线 connectionstyle='arc3, rad=0.1')
        nx.draw_networkx_edges(G, pos, 
                               width=width, 
                               edge_color=edge_colors,
                               arrowsize=20, 
                               arrowstyle='-|>',
                               connectionstyle="arc3,rad=0.15", # 弧度
                               node_size=node_sizes) # 让箭头不插进节点里

    # 6. 标签样式
    nx.draw_networkx_labels(G, pos, 
                            font_size=9, 
                            font_family='SimHei', 
                            font_weight='bold')

    plt.title('格兰杰因果关系图 (力导向布局)', fontsize=15, pad=20)
    plt.axis('off')
    
    # 添加一个颜色条说明节点重要性
    # cbar = plt.colorbar(nodes_draw, shrink=0.8)
    # cbar.set_label('节点连接数 (Degree)', rotation=270, labelpad=15)

    plt.tight_layout()
    
    if img_save_path:
        os.makedirs(os.path.dirname(img_save_path), exist_ok=True)
        plt.savefig(img_save_path, dpi=300, bbox_inches='tight')
        print(f"美化因果图已保存至: {img_save_path}")
    else:
        plt.show()
    
    
    plt.close()
    

csv_path = "data/GC_predict.csv"
series_names = ['X1', 'X2', 'X3', 'X4', 'X5', 'X6']

save_causal_links(csv_path = csv_path,  img_save_path = None , series_names = series_names)