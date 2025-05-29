import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm # 用于中文字体
import io # 用于处理字符串IO作为文件

# 尝试查找可用的中文字体
def get_chinese_font():
    try:
        # 优先尝试常见的中文黑体
        font = fm.FontProperties(fname='/usr/share/fonts/truetype/wqy/wqy-microhei.ttc') # Linux常见路径
        if font.get_name() != 'WenQuanYi Micro Hei': # 检查字体是否真的加载成功
             raise FileNotFoundError
        return font
    except FileNotFoundError:
        pass # 如果找不到，尝试其他
    try:
        font = fm.FontProperties(family='SimHei') # Windows常见
        if font.get_name() != 'SimHei':
             raise Exception
        return font
    except Exception:
        pass
    try:
        font = fm.FontProperties(family='Arial Unicode MS') # Mac/Windows
        if font.get_name() != 'Arial Unicode MS':
            raise Exception
        return font
    except Exception:
        print("警告：未找到推荐的中文字体 (文泉驿微米黑, SimHei, Arial Unicode MS)。标签可能无法正确显示中文。")
        print("您可以尝试安装这些字体或修改代码以使用您系统上已有的中文字体。")
        return None # 返回None，让matplotlib使用默认字体

# 获取中文字体属性
chinese_font = get_chinese_font()

def plot_causal_graph_from_csv(csv_path_or_data,
                               output_image_path=None,
                               series_names=None,
                               show_weights=True,
                               weight_threshold=0.0,
                               figsize=(10, 8),
                               layout_type='spring',
                               node_size=2000,
                               node_color='skyblue',
                               font_size=10,
                               arrow_size=20,
                               edge_color='gray',
                               weighted_edge_width=False,
                               self_loops_allowed=False):
    """
    从CSV文件或数据字符串绘制因果图。

    Args:
        csv_path_or_data (str or pd.DataFrame): CSV文件路径或包含CSV数据的字符串或预加载的DataFrame。
                                                格式: cause_idx,effect_idx[,weight] (无表头)
        output_image_path (str, optional): 图片保存路径。如果为None，则显示图片。
        series_names (list or dict, optional): 序列的名称。
                                              如果是list，索引对应节点ID。
                                              如果是dict，键是节点ID，值是名称。
                                              默认为None，使用节点ID作为标签。
        show_weights (bool, optional): 是否在边上显示权重。默认为True。
        weight_threshold (float, optional): 仅显示权重高于此阈值的边。默认为0.0。
        figsize (tuple, optional): 图像的尺寸。默认为(10, 8)。
        layout_type (str, optional): networkx的布局类型。
                                     可选: 'spring', 'circular', 'kamada_kawai', 'random', 'shell', 'spectral'.
                                     默认为'spring'。
        node_size (int, optional): 节点大小。默认为2000。
        node_color (str, optional): 节点颜色。默认为'skyblue'。
        font_size (int, optional): 标签字体大小。默认为10。
        arrow_size (int, optional): 箭头大小。默认为20。
        edge_color (str, optional): 边的颜色。默认为'gray'。
        weighted_edge_width (bool, optional): 是否根据权重调整边的宽度。默认为False。
        self_loops_allowed (bool, optional): 是否允许并绘制自环。默认为False。
    """
    try:
        if isinstance(csv_path_or_data, str):
            if '\n' in csv_path_or_data: # 假设是数据字符串
                data = io.StringIO(csv_path_or_data)
                df = pd.read_csv(data, header=None)
            else: # 假设是文件路径
                df = pd.read_csv(csv_path_or_data, header=None)
        elif isinstance(csv_path_or_data, pd.DataFrame):
            df = csv_path_or_data
        else:
            raise ValueError("csv_path_or_data 必须是文件路径、CSV字符串或pandas DataFrame。")

    except FileNotFoundError:
        print(f"错误：CSV文件未找到于 '{csv_path_or_data}'")
        return
    except pd.errors.EmptyDataError:
        print(f"错误：CSV文件 '{csv_path_or_data}' 为空。")
        return
    except Exception as e:
        print(f"读取CSV时发生错误: {e}")
        return

    # 确定列的数量
    num_cols = df.shape[1]
    if num_cols < 2:
        print("错误：CSV文件必须至少有两列（原因，结果）。")
        return

    has_weights_column = num_cols >= 3

    # 创建有向图
    G = nx.DiGraph()

    # 收集所有节点以确保即使没有边的节点也被添加
    all_nodes = set(df.iloc[:, 0].unique()) | set(df.iloc[:, 1].unique())
    min_node = 0
    if all_nodes: # 确保all_nodes不为空
        min_node = min(all_nodes) if min(all_nodes) == 0 else 0 # 确保节点从0开始或包含0
        max_node = max(all_nodes)
        for i in range(min_node, int(max_node) + 1): # 转换为int以防是float
            G.add_node(i)
    else: # 如果CSV为空但series_names可能被提供
        if series_names:
            if isinstance(series_names, list):
                 for i in range(len(series_names)): G.add_node(i)
            elif isinstance(series_names, dict):
                 for node_id in series_names.keys(): G.add_node(node_id)


    # 添加边和权重
    edges_to_add = []
    edge_weights = {}
    edge_widths = []

    for _, row in df.iterrows():
        cause = int(row.iloc[0])
        effect = int(row.iloc[1])

        if not self_loops_allowed and cause == effect:
            continue

        weight = 1.0 # 默认权重
        if has_weights_column:
            try:
                weight = float(row.iloc[2])
            except ValueError:
                print(f"警告：在行 {_} 中找到无效的权重值，将使用默认权重1.0。")
                weight = 1.0

        if weight > weight_threshold:
            edges_to_add.append((cause, effect))
            edge_weights[(cause, effect)] = f"{weight:.2f}" # 格式化权重用于显示
            if weighted_edge_width:
                edge_widths.append(weight) # 收集原始权重用于边宽

    G.add_edges_from(edges_to_add)

    if not G.nodes():
        print("图中没有节点可供绘制。")
        return

    plt.figure(figsize=figsize)

    # 节点标签
    labels = {}
    if series_names:
        if isinstance(series_names, list):
            for i, name in enumerate(series_names):
                if i in G.nodes(): # 只为存在的节点添加标签
                    labels[i] = str(name)
        elif isinstance(series_names, dict):
            for node_id, name in series_names.items():
                if node_id in G.nodes():
                    labels[node_id] = str(name)
        else:
            print("警告：series_names 格式无效。将使用节点ID作为标签。")
            labels = {node: str(node) for node in G.nodes()}
    else:
        labels = {node: str(node) for node in G.nodes()}


    # 选择布局
    if layout_type == 'circular':
        pos = nx.circular_layout(G)
    elif layout_type == 'kamada_kawai':
        pos = nx.kamada_kawai_layout(G)
    elif layout_type == 'random':
        pos = nx.random_layout(G, seed=42)
    elif layout_type == 'shell':
        pos = nx.shell_layout(G)
    elif layout_type == 'spectral':
        pos = nx.spectral_layout(G)
    else: # 默认为 spring_layout
        pos = nx.spring_layout(G, k=0.8, iterations=50, seed=42) # k值调整节点间距

    # 绘制图
    nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color=node_color, alpha=0.9)

    # 处理边宽
    current_edge_widths = 1.5 # 默认边宽
    if weighted_edge_width and edge_widths:
        # 标准化权重以获得合理的边宽范围
        min_w, max_w = min(edge_widths), max(edge_widths)
        if max_w > min_w:
            current_edge_widths = [(1 + 4 * (w - min_w) / (max_w - min_w) if max_w > min_w else 1) for w in edge_widths]
        else: # 所有权重相同
            current_edge_widths = [2.5] * len(edge_widths)
    elif weighted_edge_width and not edge_widths: # 如果启用了权重边宽但没有符合条件的边
        current_edge_widths = []


    nx.draw_networkx_edges(G, pos,
                           edgelist=list(G.edges()), # 确保edgelist与width顺序一致
                           width=current_edge_widths if weighted_edge_width and current_edge_widths else 1.5,
                           edge_color=edge_color,
                           arrows=True,
                           arrowstyle='-|>',
                           arrowsize=arrow_size,
                           connectionstyle='arc3,rad=0.1') # 给边一点弧度以避免重叠

    # 绘制节点标签
    nx.draw_networkx_labels(G, pos, labels, font_size=font_size, font_family=chinese_font.get_name() if chinese_font else None)

    # 绘制边权重
    if show_weights and has_weights_column:
        # 仅为实际存在的边绘制权重
        valid_edge_weights = {edge: weight_str for edge, weight_str in edge_weights.items() if edge in G.edges()}
        nx.draw_networkx_edge_labels(G, pos,
                                     edge_labels=valid_edge_weights,
                                     font_size=font_size - 2,
                                     font_color='darkred',
                                     font_family=chinese_font.get_name() if chinese_font else None)

    plt.title("因果关系图", fontsize=15, fontproperties=chinese_font)
    plt.axis('off') # 关闭坐标轴
    plt.tight_layout()

    if output_image_path:
        plt.savefig(output_image_path, dpi=300, bbox_inches='tight')
        print(f"因果图已保存至: {output_image_path}")
    else:
        plt.show()
    plt.close()

# --- 演示样例 ---
if __name__ == '__main__':
    # 1. 使用包含权重的CSV数据字符串进行演示
    sample_csv_data_with_weights = """0,1,0.95
0,2,0.88
1,3,0.76
2,3,0.92
3,4,0.85
1,4,0.5
0,0,0.99 
"""
    print("--- 演示1: 带权重的因果图 (包含自环，如果允许) ---")
    plot_causal_graph_from_csv(
        sample_csv_data_with_weights,
        series_names=["变量A", "变量B", "变量C", "变量D", "变量E"],
        show_weights=True,
        output_image_path="causal_graph_demo1.png",
        layout_type='kamada_kawai',
        weighted_edge_width=False,
        self_loops_allowed=True # 允许自环
    )

    # 2. 使用不带权重的CSV数据字符串，并使用不同布局
#     sample_csv_data_no_weights = """0,1
# 1,2
# 2,0
# 3,1
# """
#     print("\n--- 演示2: 不带权重的因果图，圆形布局 ---")
#     plot_causal_graph_from_csv(
#         sample_csv_data_no_weights,
#         series_names={0: "X₀", 1: "X₁", 2: "X₂", 3: "X₃"}, # 使用字典定义名称
#         show_weights=False, # 因为CSV中没有权重列，这里设为False或True效果类似（显示默认权重1.0）
#         output_image_path="causal_graph_demo2.png",
#         layout_type='circular',
#         node_color='lightgreen'
#     )

#     # 3. 演示过滤权重
#     print("\n--- 演示3: 带权重并使用阈值过滤的因果图 ---")
#     plot_causal_graph_from_csv(
#         sample_csv_data_with_weights, # 复用带权重的数据
#         series_names=["S0", "S1", "S2", "S3", "S4"],
#         show_weights=True,
#         weight_threshold=0.90, # 只显示权重 > 0.90 的边
#         output_image_path="causal_graph_demo3.png",
#         layout_type='spring',
#         self_loops_allowed=False # 不允许自环
#     )

#     # 4. 演示一个更复杂的例子，节点名称较长
#     complex_csv_data = """0,1,0.8
# 1,2,0.9
# 2,3,0.7
# 3,0,0.6
# 0,4,0.5
# 1,4,0.95
# 4,2,0.88
# """
#     print("\n--- 演示4: 节点名称较长，自定义颜色和大小 ---")
#     plot_causal_graph_from_csv(
#         complex_csv_data,
#         series_names={
#             0: "国内生产总值(GDP)",
#             1: "居民消费价格指数(CPI)",
#             2: "失业率",
#             3: "工业增加值",
#             4: "社会消费品零售总额"
#         },
#         show_weights=True,
#         output_image_path="causal_graph_demo4.png",
#         layout_type='spring',
#         node_size=4000,
#         node_color='#FFD700', # 金色
#         font_size=8,
#         arrow_size=25,
#         edge_color='purple',
#         weighted_edge_width=True,
#         figsize=(14,12)
#     )

#     # 5. 演示当CSV文件路径不存在时的情况 (需要手动创建一个名为 "non_existent_file.csv" 的文件来测试成功路径)
#     print("\n--- 演示5: CSV文件路径不存在 ---")
#     plot_causal_graph_from_csv("non_existent_file.csv", output_image_path="causal_graph_demo5.png")

#     # 6. 演示处理空的CSV数据
#     empty_csv_data = ""
#     print("\n--- 演示6: 空的CSV数据 ---")
#     plot_causal_graph_from_csv(empty_csv_data, output_image_path="causal_graph_demo6.png", series_names=["A","B"])

#     # 7. 演示只有节点没有边的情况
#     nodes_only_csv = """0
# 1
# 2""" # 这种格式不符合cause,effect，但可以用来测试只有节点的情况
#     # 改为提供一个空的DataFrame，但有series_names
#     df_nodes_only = pd.DataFrame(columns=[0,1]) #空的边
#     print("\n--- 演示7: 只有节点没有边 (通过series_names定义节点) ---")
#     plot_causal_graph_from_csv(df_nodes_only, series_names=["Alpha", "Beta", "Gamma"], output_image_path="causal_graph_demo7.png")