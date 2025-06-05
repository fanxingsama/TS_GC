from matplotlib import rcParams
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import FancyArrowPatch

rcParams['font.family'] = 'SimHei'
rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


def save_causal_links(csv_path, img_save_path=None):
    df = pd.read_csv(csv_path, header=None, names=['source', 'target', 'weight'])

    # 解析数据
    edges = []
    for _, row in df.iterrows():
        edges.append((int(row['source']), int(row['target']), float(row['weight'])))

    # 创建有向图
    G = nx.DiGraph()

    # 添加边和权重
    for source, target, weight in edges:
        G.add_edge(source, target, weight=weight)

    # 获取所有节点
    nodes = list(G.nodes())
    nodes.sort()  # 确保节点顺序一致

    # 创建环形布局
    pos = {}
    n_nodes = len(nodes)
    for i, node in enumerate(nodes):
        angle = 2 * np.pi * i / n_nodes - np.pi/2  # 从顶部开始
        pos[node] = (np.cos(angle), np.sin(angle))

    # 设置图形大小和样式
    plt.figure(figsize=(4, 4))
    ax = plt.gca()
    ax.set_aspect('equal')

    # 绘制边（有向箭头）
    for source, target, data in G.edges(data=True):
        weight = data['weight']
        
        # 获取起点和终点坐标
        x1, y1 = pos[source]
        x2, y2 = pos[target]
        
        # 计算箭头的起点和终点（考虑节点大小）
        node_radius = 0.08
        
        # 计算方向向量
        dx = x2 - x1
        dy = y2 - y1
        length = np.sqrt(dx**2 + dy**2)
        
        # 标准化方向向量
        if length > 0:
            dx_norm = dx / length
            dy_norm = dy / length
            
            # 调整起点和终点
            start_x = x1 + node_radius * dx_norm
            start_y = y1 + node_radius * dy_norm
            end_x = x2 - node_radius * dx_norm
            end_y = y2 - node_radius * dy_norm
            
            # 创建箭头
            arrow = FancyArrowPatch(
                (start_x, start_y), (end_x, end_y),
                arrowstyle='->', 
                mutation_scale=15,
                color='#666666',
                linewidth=0.8,
                alpha=0.7
            )
            ax.add_patch(arrow)

    # 绘制节点
    node_colors = '#4285f4'  # 蓝色
    node_size = 350

    nx.draw_networkx_nodes(G, pos, 
                        node_color=node_colors,
                        node_size=node_size,
                        alpha=0.9,
                        edgecolors='white',
                        linewidths=2)

    # 绘制节点标签
    labels = {node: f'X{node}' for node in nodes}
    nx.draw_networkx_labels(G, pos, labels,
                        font_size=6,
                        font_color='white',
                        font_weight='bold')

    # 设置图形属性
    plt.title('格兰杰因果图', 
            fontsize=12, fontweight='bold', pad=10)

    # 移除坐标轴
    plt.axis('off')

    # 设置图形边界
    plt.xlim(-1.3, 1.3)
    plt.ylim(-1.3, 1.3)

    # 调整布局
    plt.tight_layout()
    plt.savefig(img_save_path, dpi=300, bbox_inches='tight')