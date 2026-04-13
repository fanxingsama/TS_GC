import json
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib import rcParams

rcParams['axes.unicode_minus'] = False


def hierarchy_pos(G, root, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5):
    """为树状结构生成分层布局"""
    pos = {root: (xcenter, vert_loc)}
    children = list(G.neighbors(root))
    if not children:
        return pos
    dx = width / len(children)
    nextx = xcenter - width / 2 - dx / 2
    for child in children:
        nextx += dx
        pos.update(hierarchy_pos(G, root=child, width=dx, vert_gap=vert_gap,
                                 vert_loc=vert_loc - vert_gap, xcenter=nextx))
    return pos


def analyze_root_cause_and_save(json_path, output_img_path):
    if not json_path.exists():
        print(f"错误: 找不到文件 {json_path}")
        return

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    variables = data['variables']
    root_id = data['root']

    # ---- 构建异常传播树 (用于生成层次布局) ----
    tree = nx.DiGraph()
    for var_id in variables:
        tree.add_node(var_id)
    anomaly_edge_set = set()
    for edge in data['anomaly_edges']:
        s, t = edge['source'], edge['target']
        tree.add_edge(s, t)
        anomaly_edge_set.add((s, t))

    # ---- 构建完整因果图 (异常边 + 额外因果边) ----
    full_graph = tree.copy()
    causal_edge_set = set()
    for edge in data.get('causal_edges', []):
        s, t = edge['source'], edge['target']
        full_graph.add_edge(s, t)
        causal_edge_set.add((s, t))

    # ---- 布局：基于异常树做层次布局 ----
    try:
        pos = hierarchy_pos(tree, root=root_id, width=2.4, vert_gap=0.30)
    except Exception:
        pos = nx.spring_layout(full_graph, seed=42, k=2.0)

    # ---- 节点分类 ----
    anomaly_nodes = set()
    for s, t in anomaly_edge_set:
        anomaly_nodes.add(s)
        anomaly_nodes.add(t)
    non_anomaly_nodes = set(variables.keys()) - anomaly_nodes

    # ---- 绘图 ----
    fig, ax = plt.subplots(figsize=(10, 9))
    node_size = 1400

    # 1) 先画额外因果边 (浅灰、细线、弧形，底层)
    if causal_edge_set:
        nx.draw_networkx_edges(
            full_graph, pos,
            edgelist=list(causal_edge_set),
            node_size=node_size,
            edge_color='#cccccc',
            arrowstyle='-|>',
            arrowsize=14,
            width=1.2,
            alpha=0.55,
            connectionstyle='arc3,rad=0.15',
            ax=ax,
        )

    # 2) 再画异常传播边 (粗黑箭头，高亮)
    nx.draw_networkx_edges(
        full_graph, pos,
        edgelist=list(anomaly_edge_set),
        node_size=node_size,
        edge_color='#222222',
        arrowstyle='-|>',
        arrowsize=22,
        width=2.8,
        alpha=0.9,
        connectionstyle='arc3,rad=0.0',
        ax=ax,
    )

    # 3) 绘制异常路径节点 (鲜红)
    if anomaly_nodes:
        nx.draw_networkx_nodes(
            full_graph, pos,
            nodelist=list(anomaly_nodes),
            node_size=node_size,
            node_color='#e74c3c',
            edgecolors='#a93226',
            linewidths=1.8,
            ax=ax,
        )

    # 4) 绘制非异常路径节点 (浅粉)
    if non_anomaly_nodes:
        nx.draw_networkx_nodes(
            full_graph, pos,
            nodelist=list(non_anomaly_nodes),
            node_size=node_size,
            node_color='#f5b7b1',
            edgecolors='#cd6155',
            linewidths=1.5,
            ax=ax,
        )

    # 5) 标签 (白色字体)
    nx.draw_networkx_labels(
        full_graph, pos,
        font_size=12, font_weight='bold',
        font_color='white',
        ax=ax,
    )

    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(str(output_img_path), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"成功: 因果推理路径图已保存至 {output_img_path}")


def main():
    script_dir = Path(__file__).parent
    json_path = script_dir / '1.json'
    output_img_path = '异常传播推理路径图.pdf'
    analyze_root_cause_and_save(json_path, output_img_path)


if __name__ == "__main__":
    main()