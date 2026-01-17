import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
import os
from matplotlib import rcParams
from util.util import get_latest_run_id


# --- 配置字体 (保持和 test_TS_GC 一致) ---
rcParams['font.family'] = 'SimHei'
rcParams['axes.unicode_minus'] = False

def hierarchy_pos(G, root=None, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5):
    """
    自定义函数：为树状结构生成分层布局位置 (Hierarchical Layout)
    """
    if not nx.is_tree(G):
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

# 构建最大生成树
def analyze_root_cause_and_save(model_folder_path):
    # 1. 构建路径
    input_csv_path = model_folder_path / "GC_matrix.csv"
    output_img_path = model_folder_path / "root_cause_max_tree.png"

    print(f"正在处理文件: {input_csv_path}")

    # 2. 读取数据
    if not input_csv_path.exists():
        print(f"错误: 找不到文件 {input_csv_path}")
        return

    try:
        # 假设 constrained csv 没有表头，格式为 source, target, weight
        df = pd.read_csv(input_csv_path, header=None, names=['Cause', 'Effect', 'Weight'])
    except Exception as e:
        print(f"读取 CSV 失败: {e}")
        return

    if df.empty:
        print("警告: CSV 文件为空，无法构建生成树。")
        return

    # 3. 构建图
    G = nx.DiGraph()
    for _, row in df.iterrows():
        G.add_edge(str(row['Cause']), str(row['Effect']), weight=float(row['Weight']))

    if G.number_of_nodes() == 0:
        print("警告: 图中没有节点。")
        return

    # 4. 计算最大生成树 (MST)
    try:
        mst = nx.maximum_spanning_arborescence(G, attr='weight', default=None)
    except Exception as e:
        print(f"构建最大生成树失败 (可能是图不连通): {e}")
        # 如果无法构建完美树，退化为使用原始图（或者你可以选择报错）
        mst = G 
    
    # 5. 寻找根节点
    root_candidates = [n for n, d in mst.in_degree() if d == 0]
    if not root_candidates:
        print("未找到明确的根节点，尝试使用度数最大的节点作为布局根节点。")
        # 备选方案：入度为0的节点，或者出度最大的节点
        if mst.nodes:
            root = max(mst.nodes, key=mst.out_degree)
        else:
            return
    else:
        root = root_candidates[0]
        
    print(f"诊断结果：根原因变量为 {root}")

    # 6. 可视化设置
    plt.figure(figsize=(14, 10))

    # --- 布局选择 ---
    try:
        pos = hierarchy_pos(mst, root=root)
    except Exception:
        pos = nx.spring_layout(mst, seed=42, k=2.0)

    # 绘制节点 (先定义节点大小，后面要用到)
    node_size = 1500 

    # 绘制节点
    nx.draw_networkx_nodes(mst, pos, node_size=node_size, node_color='#E8F6F3', edgecolors='#1ABC9C')

    # 绘制边 (关键修改：加入 node_size 和 arrows=True)
    nx.draw_networkx_edges(
        mst, 
        pos, 
        node_size=node_size,  # <--- 【关键】告诉画边的函数节点有多大，它会自动缩短边
        edge_color='gray', 
        arrowstyle='-|>',     # 箭头样式
        arrowsize=20,         # 箭头大小
        arrows=True,          # 显式开启箭头
        alpha=0.6, 
        connectionstyle='arc3,rad=0.05'
    )

    # 绘制节点标签
    nx.draw_networkx_labels(mst, pos, font_size=11, font_weight='bold', font_color='#2C3E50')

    # --- 绘制权重标签 ---
    edge_labels = nx.get_edge_attributes(mst, 'weight')
    # 格式化权重保留2位小数
    edge_labels = {k: f"{v:.2f}" for k, v in edge_labels.items()}
    
    nx.draw_networkx_edge_labels(
        mst, 
        pos, 
        edge_labels=edge_labels, 
        font_color='red',
        font_size=9,
        label_pos=0.5,
        rotate=False,
        bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=0.5)
    )

    plt.title(f"根因诊断树 (Root: {root})", fontsize=15)
    plt.axis('off')
    
    # 7. 保存图片而不是显示
    plt.tight_layout()
    plt.savefig(output_img_path, dpi=300, bbox_inches='tight')
    plt.close() # 关闭画布释放内存
    
    print(f"成功: 最大生成树图片已保存至 {output_img_path}")

def main():
    # 1. 获取 ID (逻辑同 test_TS_GC.py)
    # 你可以手动指定，也可以自动获取最新
    run_id = 'T2_high' 
    # run_id = get_latest_run_id()
    
    if not run_id:
        print("错误: 无法获取 run_id")
        return

    # 2. 构建基础路径
    base_path = Path('saved') / run_id
    
    if not base_path.exists():
        print(f"错误: 文件夹不存在 {base_path}")
        return

    # 3. 执行分析
    analyze_root_cause_and_save(base_path)

if __name__ == "__main__":
    main()