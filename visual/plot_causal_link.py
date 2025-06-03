import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm # 用于中文字体
import io # 用于处理字符串IO作为文件
import numpy as np # 导入numpy用于数学运算

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
                               dataset_csv_path=None,
                               show_weights=True,
                               weight_threshold=0.0,
                               figsize=(10, 10), # 调整默认大小为方形，更适合环形
                               node_size=2500,
                               node_color='skyblue',
                               font_size=10,
                               arrow_size=20,
                               edge_color='gray',
                               weighted_edge_width=False,
                               self_loops_allowed=False,
                               label_offset_ratio=0.1): # 新增：标签偏移比例
    """
    从CSV文件或数据字符串绘制环形有向因果图。

    Args:
        csv_path_or_data (str or pd.DataFrame): 定义因果链接的CSV文件路径、数据字符串或DataFrame。
                                                格式: cause_idx,effect_idx[,weight] (无表头)
        output_image_path (str, optional): 图片保存路径。如果为None，则显示图片。
        dataset_csv_path (str, optional): 包含原始时间序列数据的CSV文件路径。
                                          将使用此文件的列标题作为图中节点的名称。
        show_weights (bool, optional): 是否在边上显示权重。默认为True。
        weight_threshold (float, optional): 仅显示权重高于此阈值的边。默认为0.0。
        figsize (tuple, optional): 图像的尺寸。默认为(10, 10)。
        node_size (int, optional): 节点大小。默认为2500。
        node_color (str, optional): 节点颜色。默认为'skyblue'。
        font_size (int, optional): 标签字体大小。默认为10。
        arrow_size (int, optional): 箭头大小。默认为20。
        edge_color (str, optional): 边的颜色。默认为'gray'。
        weighted_edge_width (bool, optional): 是否根据权重调整边的宽度。默认为False。
        self_loops_allowed (bool, optional): 是否允许并绘制自环。默认为False。
        label_offset_ratio (float, optional): 节点标签相对于节点中心的偏移比例，用于避免标签与节点重叠。
    """
    series_names_from_dataset = None
    if dataset_csv_path:
        try:
            df_dataset = pd.read_csv(dataset_csv_path)
            series_names_from_dataset = df_dataset.columns.tolist()
            print(f"从 '{dataset_csv_path}' 加载的系列名称: {series_names_from_dataset}")
        except FileNotFoundError:
            print(f"警告：数据集CSV文件 '{dataset_csv_path}' 未找到。将使用节点ID作为标签。")
        except Exception as e:
            print(f"警告：读取数据集CSV '{dataset_csv_path}' 时发生错误: {e}。将使用节点ID作为标签。")


        if isinstance(csv_path_or_data, str):
            if '\n' in csv_path_or_data:
                data = io.StringIO(csv_path_or_data)
                df_causal = pd.read_csv(data, header=None)
            else:
                df_causal = pd.read_csv(csv_path_or_data, header=None)
        elif isinstance(csv_path_or_data, pd.DataFrame):
            df_causal = csv_path_or_data
        else:
            raise ValueError("csv_path_or_data 必须是文件路径、CSV字符串或pandas DataFrame。")



    num_cols = df_causal.shape[1]
    if num_cols < 2:
        print("错误：因果关系CSV文件必须至少有两列（原因，结果）。")
        return
    has_weights_column = num_cols >= 3

    G = nx.DiGraph()

    all_nodes_in_causal_data = set()
    if not df_causal.empty:
        # 确保在转换为int之前处理潜在的非数字值或NaN
        causal_col0 = pd.to_numeric(df_causal.iloc[:, 0], errors='coerce')
        causal_col1 = pd.to_numeric(df_causal.iloc[:, 1], errors='coerce')
        all_nodes_in_causal_data.update(causal_col0.dropna().unique())
        all_nodes_in_causal_data.update(causal_col1.dropna().unique())


    max_node_id_from_names = -1
    if series_names_from_dataset:
        max_node_id_from_names = len(series_names_from_dataset) - 1
        for i in range(len(series_names_from_dataset)):
            G.add_node(i)

    if all_nodes_in_causal_data:
        int_nodes_in_causal_data = {int(n) for n in all_nodes_in_causal_data}
        if int_nodes_in_causal_data:
             max_node_id_from_causal = max(int_nodes_in_causal_data)
             overall_max_node_id = max(max_node_id_from_names, max_node_id_from_causal)
             for i in range(overall_max_node_id + 1):
                 if i not in G:
                     G.add_node(i)

    edges_to_add = []
    edge_weights = {}
    edge_widths_values = []

    for idx, row in df_causal.iterrows():
        try:
            cause = int(row.iloc[0])
            effect = int(row.iloc[1])
        except ValueError:
            print(f"警告：在行 {idx} 中找到无效的节点ID，跳过此行。 ({row.iloc[0]}, {row.iloc[1]})")
            continue

        if not self_loops_allowed and cause == effect:
            continue

        weight = 1.0
        if has_weights_column:
            try:
                weight = float(row.iloc[2])
            except ValueError:
                print(f"警告：在行 {idx} (cause:{cause}, effect:{effect}) 中找到无效的权重值，将使用默认权重1.0。")
                weight = 1.0

        if weight > weight_threshold:
            if cause not in G: G.add_node(cause)
            if effect not in G: G.add_node(effect)
            edges_to_add.append((cause, effect))
            edge_weights[(cause, effect)] = f"{weight:.2f}"
            if weighted_edge_width:
                edge_widths_values.append(weight)

    G.add_edges_from(edges_to_add)

    if not G.nodes():
        print("图中没有节点可供绘制。")
        return

    plt.figure(figsize=figsize)
    ax = plt.gca() # 获取当前轴

    # --- 环形布局和标签偏移逻辑 ---
    pos = nx.circular_layout(G)
    
    labels = {}
    label_pos = {} # 用于存储标签的偏移位置

    if series_names_from_dataset:
        for i, name in enumerate(series_names_from_dataset):
            if i in G.nodes():
                labels[i] = str(name)
        for node in G.nodes():
            if node not in labels: # 处理超出 series_names 范围的节点
                labels[node] = str(node)
    else:
        labels = {node: str(node) for node in G.nodes()}

    # 计算标签的偏移位置
    for node, (x, y) in pos.items():
        angle = np.arctan2(y, x) # 计算节点相对于中心点的角度
        # 根据角度将标签向外偏移
        label_x = x + label_offset_ratio * np.cos(angle)
        label_y = y + label_offset_ratio * np.sin(angle)
        label_pos[node] = (label_x, label_y)
    # --- 结束环形布局和标签偏移逻辑 ---

    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_size, node_color=node_color, alpha=0.9)

    current_edge_widths_to_draw = 1.5
    actual_edges_in_graph = list(G.edges())

    if weighted_edge_width and edge_widths_values and actual_edges_in_graph:
        weights_for_existing_edges = []
        temp_edge_weights_map = {edge: float(weight_str) for edge, weight_str in edge_weights.items()}
        for edge in actual_edges_in_graph:
            weights_for_existing_edges.append(temp_edge_weights_map.get(edge, 1.0))

        if weights_for_existing_edges:
            min_w, max_w = min(weights_for_existing_edges), max(weights_for_existing_edges)
            if max_w > min_w:
                current_edge_widths_to_draw = [(1 + 4 * (w - min_w) / (max_w - min_w)) for w in weights_for_existing_edges]
            else:
                current_edge_widths_to_draw = [2.5] * len(weights_for_existing_edges)
        else:
             current_edge_widths_to_draw = [1.5] * len(actual_edges_in_graph)

    nx.draw_networkx_edges(G, pos, ax=ax,
                           edgelist=actual_edges_in_graph,
                           width=current_edge_widths_to_draw,
                           edge_color=edge_color,
                           arrows=True,
                           arrowstyle='-|>', # 或者 '->', 'fancy'
                           arrowsize=arrow_size,
                           connectionstyle='arc3,rad=0.15') # 调整弧度

    # 使用计算好的 label_pos 绘制标签
    nx.draw_networkx_labels(G, label_pos, labels, ax=ax, font_size=font_size,
                            font_family=chinese_font.get_name() if chinese_font else None,
                            horizontalalignment='center', verticalalignment='center')


    if show_weights and has_weights_column:
        valid_edge_weights = {edge: weight_str for edge, weight_str in edge_weights.items() if edge in G.edges()}
        nx.draw_networkx_edge_labels(G, pos, ax=ax,
                                     edge_labels=valid_edge_weights,
                                     font_size=font_size - 2,
                                     font_color='darkred',
                                     font_family=chinese_font.get_name() if chinese_font else None,
                                     bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.1')) # 给权重标签加背景


    plt.title("因果关系环形图", fontsize=15, fontproperties=chinese_font)
    ax.set_aspect('equal') # 保持环形是圆的
    plt.axis('off')
    plt.tight_layout()

    if output_image_path:
        plt.savefig(output_image_path, dpi=300, bbox_inches='tight')
        print(f"因果图已保存至: {output_image_path}")
    else:
        plt.show()
    plt.close()

# --- 演示样例 ---
if __name__ == '__main__':
    sample_dataset_content = {
        "变量A": [100, 102, 105, 103, 106, 107, 108],
        "变量B": [2.0, 2.1, 2.0, 2.2, 2.3, 2.0, 2.1],
        "变量C": [5.0, 4.9, 5.1, 5.0, 4.8, 4.9, 5.0],
        "变量D": [80, 82, 85, 83, 86, 88, 87],
        "变量E": [200, 205, 210, 208, 215, 212, 218],
        "变量F": [30, 33, 31, 35, 32, 34, 36],
        "变量G": [10, 11, 12, 10, 13, 11, 12]
    }
    df_sample_dataset = pd.DataFrame(sample_dataset_content)
    sample_dataset_csv_path = "sample_dataset_for_circular_plot.csv"
    df_sample_dataset.to_csv(sample_dataset_csv_path, index=False)
    print(f"示例数据集已创建: {sample_dataset_csv_path}")

    # 1. 环形图演示
    sample_causal_circular_data = """0,1,0.9
1,2,0.8
2,3,0.7
3,0,0.6
0,4,0.5
1,5,0.95
2,6,0.88
4,0,0.4 
5,1,0.3
6,2,0.8
0,6,0.75
"""
    print("\n--- 演示 环形因果图 ---")
    plot_causal_graph_from_csv(
        sample_causal_circular_data,
        dataset_csv_path=sample_dataset_csv_path,
        show_weights=True,
        output_image_path="causal_circular_graph_demo.png",
        node_size=3000,
        font_size=9,
        arrow_size=25,
        weighted_edge_width=True,
        self_loops_allowed=False, # 在环形图中自环可能不美观
        figsize=(12,12), # 增大图像尺寸
        label_offset_ratio=0.15 # 调整标签偏移
    )

    # 2. 节点较少的环形图
    sample_causal_simple_circular_data = """0,1,0.9
1,2,0.8
2,0,0.7
0,3,0.5
3,1,0.6
"""
    simple_dataset_content = { "X0":[1], "X1":[1], "X2":[1], "X3":[1]}
    df_simple_dataset = pd.DataFrame(simple_dataset_content)
    simple_dataset_csv_path = "simple_dataset_for_circular_plot.csv"
    df_simple_dataset.to_csv(simple_dataset_csv_path, index=False)
    print(f"简单示例数据集已创建: {simple_dataset_csv_path}")


   

    # 清理示例数据集文件
    try:
        import os
        os.remove(sample_dataset_csv_path)
        print(f"示例数据集已删除: {sample_dataset_csv_path}")
        os.remove(simple_dataset_csv_path)
        print(f"简单示例数据集已删除: {simple_dataset_csv_path}")
    except OSError as e:
        print(f"删除示例数据集时出错: {e}")
