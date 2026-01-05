import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

def hierarchy_pos(G, root=None, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5):
    """
    自定义函数：为树状结构生成分层布局位置 (Hierarchical Layout)
    :param G: 图
    :param root: 根节点
    :param width: 布局总宽度
    :param vert_gap: 层级间的垂直间距
    :param vert_loc: 当前层的垂直坐标
    :param xcenter: 当前层的水平中心
    :return: pos 字典 {节点: (x, y)}
    """
    if not nx.is_tree(G):
        # 如果不是严格的树（比如有孤立点），回退到 shell 布局或 spring 布局
        return nx.shell_layout(G)
        
    pos = {root: (xcenter, vert_loc)}
    children = list(G.neighbors(root))
    if not children:
        return pos
    
    dx = width / len(children) 
    nextx = xcenter - width/2 - dx/2
    
    for child in children:
        nextx += dx
        pos.update(hierarchy_pos(G, root=child, width=dx, vert_gap=vert_gap, 
                                 vert_loc=vert_loc-vert_gap, xcenter=nextx))
    return pos

def analyze_root_cause_visualized(file_path):
    # 1. 读取数据
    try:
        df = pd.read_csv(file_path, header=None, names=['Cause', 'Effect', 'Weight'])
    except FileNotFoundError:
        print(f"找不到文件: {file_path}")
        return

    # 2. 构建图
    G = nx.DiGraph()
    for _, row in df.iterrows():
        G.add_edge(str(row['Cause']), str(row['Effect']), weight=float(row['Weight']))

    # 3. 计算最大生成树 (MST)
    mst = nx.maximum_spanning_arborescence(G, attr='weight', default=None)
    
    # 4. 寻找根节点
    root_candidates = [n for n, d in mst.in_degree() if d == 0]
    if not root_candidates:
        print("未找到根节点，无法使用树状布局。")
        return
    
    root = root_candidates[0]
    print(f"诊断结果：根原因变量为 {root}")

    # 5. 可视化设置 (关键修改部分)
    plt.figure(figsize=(14, 10)) # 增大画布

    # --- 使用自定义的树状布局 ---
    # 这会将根节点放在最上面，子节点依次往下排，彻底解决遮挡问题
    try:
        pos = hierarchy_pos(mst, root=root)
    except Exception as e:
        print("树状布局生成失败，尝试使用弹簧布局扩大间距...")
        pos = nx.spring_layout(mst, seed=42, k=2.0) # k值越大，节点越分散

    # 绘制边 (先画线，且带有弧度，避免直线穿过节点)
    # connectionstyle='arc3,rad=0.1' 可以让线稍微弯曲一点，避免重合
    nx.draw_networkx_edges(mst, pos, edge_color='gray', arrowstyle='-|>', arrowsize=20, 
                           alpha=0.6, connectionstyle='arc3,rad=0.05')

    # 绘制节点 (增大节点，设为浅色以便看清文字)
    nx.draw_networkx_nodes(mst, pos, node_size=1500, node_color='#E8F6F3', edgecolors='#1ABC9C')

    # 绘制节点标签 (确保字体居中)
    nx.draw_networkx_labels(mst, pos, font_size=11, font_weight='bold', font_color='#2C3E50')

    # --- 绘制权重标签 (解决“挡住内容”的核心) ---
    edge_labels = nx.get_edge_attributes(mst, 'weight')
    edge_labels = {k: f"{v:.2f}" for k, v in edge_labels.items()}
    
    nx.draw_networkx_edge_labels(
        mst, 
        pos, 
        edge_labels=edge_labels, 
        font_color='red',
        font_size=9,
        label_pos=0.5,  # 标签在线的中间
        rotate=False,   # 不旋转文字，方便阅读
        # 【关键】：给文字加一个白色背景框，这样即使线穿过，文字也清晰可见
        bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=0.5)
    )

    plt.title(f"Root Cause Diagnosis Tree (Root: {root})", fontsize=15)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 请确保这里是你的CSV文件路径
    file_path = 'GC_matrix.csv' 
    analyze_root_cause_visualized(file_path)