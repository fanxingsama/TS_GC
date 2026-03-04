import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

def plot_causal_graph_v2(csv_filename, output_image='causal_graph_v2.png'):
    # 1. 读取 CSV 文件 (不指定表头)
    # 无论用户是否有第三列，这里先全部读入
    try:
        df = pd.read_csv(csv_filename, header=None)
    except Exception as e:
        print(f"文件读取失败: {e}")
        return

    # 2. 自动检测列数并切片
    num_cols = df.shape[1]
    if num_cols >= 2:
        # 核心逻辑：强制只取前两列，忽略第三列
        df = df.iloc[:, :2] 
        df.columns = ['Cause', 'Effect']
        if num_cols > 2:
            print(f"提示：检测到 {num_cols} 列数据，已自动忽略第三列及后续冗余信息。")
    else:
        print("错误：CSV 文件列数不足 2 列，无法构建因果链。")
        return

    # 3. 构建有向图对象
    G = nx.DiGraph()
    for _, row in df.iterrows():
        # 排除空行并确保节点名为字符串格式
        if pd.notna(row['Cause']) and pd.notna(row['Effect']):
            G.add_edge(str(row['Cause']), str(row['Effect']))

    # 4. 配置绘图样式与布局
    # k 值控制节点间距，seed 保证每次生成的图位置一致
    pos = nx.spring_layout(G, k=2.5, seed=42)
    plt.rcParams['figure.figsize'] = (14, 10) # 针对 37 条边的大图，建议使用大尺寸
    
    # 绘制节点：淡蓝色背景 + 深蓝色边框
    nx.draw_networkx_nodes(G, pos, node_size=3000, node_color='skyblue', 
                           alpha=0.9, edgecolors='navy')

    # 绘制边：带有 0.1 的弧度，防止双向因果线重合，灰色半透明箭头
    nx.draw_networkx_edges(G, pos, width=1.5, alpha=0.5, edge_color='gray', 
                           arrowsize=20, arrowstyle='->', 
                           connectionstyle='arc3,rad=0.1')

    # 绘制标签：节点名居中显示
    nx.draw_networkx_labels(G, pos, font_size=9, font_weight='bold')

    plt.title("Causal Network with Auto-column Detection", fontsize=16)
    plt.axis('off') # 隐藏不必要的坐标轴
    
    # 5. 保存并输出
    plt.tight_layout()
    plt.show()
    # plt.savefig(output_image, dpi=300)
    # print(f"成功生成因果链图，已保存为: {output_image}")


# 执行绘图
if __name__ == "__main__":
    # 请确保文件名与你保存的 CSV 文件名一致
    plot_causal_graph_v2('../data/161.csv')