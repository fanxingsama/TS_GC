import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import rcParams
import numpy as np
import networkx as nx

rcParams['font.family'] = 'SimHei'
rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 读取CSV文件
def load_causal_data(file_path):
    try:
        df = pd.read_csv(file_path)
        
        if len(df.columns) != 3:
            print(f"警告: CSV文件应该有3列 (头结点, 尾结点, 延迟), 但发现{len(df.columns)}列")
            return None
            
        # 为列命名（如果需要）
        if list(df.columns) != ['source', 'target', 'delay']:
            df.columns = ['source', 'target', 'delay']
            
        return df
    except Exception as e:
        print(f"读取CSV文件时出错: {e}")
        return None

# 绘制因果图
def draw_causal_graph(data):
    # 获取所有唯一节点
    nodes = sorted(list(set(data['source'].tolist() + data['target'].tolist())))
    
    # 创建一个有向图
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    
    # 添加边和权重（延迟）
    for _, row in data.iterrows():
        G.add_edge(row['source'], row['target'], delay=row['delay'])
    
    # 使用networkx的spring布局算法
    pos = nx.spring_layout(G, seed=42)
    
    # 创建画布
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 绘制节点
    for node in nodes:
        x, y = pos[node]
        circle = patches.Circle((x, y), 0.1, facecolor='lightblue', edgecolor='black')
        ax.add_patch(circle)
        ax.text(x, y, node, ha='center', va='center', size=12, fontweight='bold')
    
    # 检查并处理自因果关系
    self_causals = data[data['source'] == data['target']]
    
    # 绘制常规边和延迟标签
    for source, target, edge_data in G.edges(data=True):
        # 跳过自因果关系，稍后单独处理
        if source == target:
            continue
            
        # 获取源节点和目标节点的位置
        x_source, y_source = pos[source]
        x_target, y_target = pos[target]
        
        # 计算箭头方向
        dx = x_target - x_source
        dy = y_target - y_source
        
        # 避免箭头指向节点中心，而是指向节点边缘
        offset = 0.11  # 略大于节点半径
        length = np.sqrt(dx**2 + dy**2)
        x_target = x_source + dx * (1 - offset/length)
        y_target = y_source + dy * (1 - offset/length)
        
        # 绘制箭头
        ax.annotate('', 
                   xy=(x_target, y_target), 
                   xytext=(x_source, y_source),
                   arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8))
        
        # 在边上添加延迟标签
        middle_x = (x_source + x_target) / 2
        middle_y = (y_source + y_target) / 2
        # 在箭头中间稍微偏移的位置放置延迟标签
        offset_x = 0.03 * (-dy/length if length > 0 else 0)
        offset_y = 0.03 * (dx/length if length > 0 else 0)
        ax.text(middle_x + offset_x, middle_y + offset_y, 
                f"τ={edge_data['delay']}", 
                ha='center', va='center', 
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    
    # 单独处理自因果关系
    for _, row in self_causals.iterrows():
        node = row['source']  # source和target相同
        x, y = pos[node]
        
        # 创建一个自循环箭头
        radius = 0.15
        theta = np.linspace(0, 2*np.pi, 100)
        
        # 创建一个圆弧，从节点右侧开始，顺时针旋转270度
        arc_start = -np.pi/4
        arc_end = np.pi*3/4
        arc_theta = np.linspace(arc_start, arc_end, 100)
        
        # 绘制自循环弧线
        arc_x = x + radius * np.cos(arc_theta)
        arc_y = y + radius * np.sin(arc_theta)
        ax.plot(arc_x, arc_y, color='black', linewidth=1.5)
        
        # 添加箭头
        arrow_pos = arc_end - 0.2  # 箭头位置在弧线靠近结束位置
        dx = -np.sin(arrow_pos) * 0.02  # 箭头方向的x分量
        dy = np.cos(arrow_pos) * 0.02   # 箭头方向的y分量
        ax.arrow(x + radius * np.cos(arrow_pos), 
                 y + radius * np.sin(arrow_pos),
                 dx, dy, 
                 head_width=0.03, 
                 head_length=0.03, 
                 fc='black', 
                 ec='black')
        
        # 添加延迟标签，位于弧线上方
        label_theta = -np.pi/8  # 标签位置在弧线顶部偏右
        label_x = x + (radius + 0.05) * np.cos(label_theta)
        label_y = y + (radius + 0.05) * np.sin(label_theta)
        
        ax.text(label_x, label_y, 
                f"τ={row['delay']}", 
                ha='center', va='center', 
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    
    # 设置图形的显示范围
    margin = 0.3  # 增加边距以确保自循环完全可见
    x_values = [pos[node][0] for node in nodes]
    y_values = [pos[node][1] for node in nodes]
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)
    
    ax.set_xlim(x_min - margin, x_max + margin)
    ax.set_ylim(y_min - margin, y_max + margin)
    
    # 隐藏坐标轴
    ax.axis('off')
    plt.title('因果网络图', fontsize=16)
    
    return fig, ax

# 主函数
def main():
    # file_path = '../data/fMRI/sim15_gt_processed.csv' # 文件路径
    file_path = '../saved/models/0410_152831/FMRI15/causal_structure.csv'
    
    # 加载数据
    data = load_causal_data(file_path)
    
    if data is not None:
        print("加载的因果关系数据:")
        print(data)
        
        # 检查是否有自因果关系
        self_causals = data[data['source'] == data['target']]
        if not self_causals.empty:
            print(f"发现 {len(self_causals)} 个自因果关系:")
            print(self_causals)
        
        # 绘制图形
        fig, ax = draw_causal_graph(data)
        
        # 显示图形
        plt.tight_layout()
        plt.show()
        
        # 询问是否保存图像
        save = input("是否保存图像? (y/n): ")
        if save.lower() == 'y':
            output_path = input("请输入保存路径 (例如: causal_graph.png): ")
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"图像已保存到 {output_path}")
    else:
        print("无法处理因果数据，请检查CSV文件格式。")

if __name__ == "__main__":
    main()