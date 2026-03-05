import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_custom_matrix(file_path):
    # 1. 读取数据
    df = pd.read_csv(file_path, header=None, names=['row', 'col', 'val'])

    # 2. 透视表转换为矩阵
    matrix = df.pivot(index='row', columns='col', values='val')
    matrix = matrix.fillna(0)

    # 3. 计算尺寸：根据矩阵大小动态调整画布，防止太挤
    fig_size = max(8, len(matrix.columns) * 0.5)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    
    # 4. 绘图
    # aspect='equal' 保证每个像素点（格子）是正方形
    im = ax.imshow(matrix.values, cmap="Blues", aspect='equal')
    
    # 5. 解决标签重叠：旋转45度，并设置对齐方式为右对齐
    plt.xticks(np.arange(len(matrix.columns)), matrix.columns, 
               rotation=45, ha='right', rotation_mode='anchor')
    plt.yticks(np.arange(len(matrix.index)), matrix.index)
    
    # 6. 强制整体绘图区域为正方形
    ax.set_box_aspect(1) 
    
    # 7. 设置标题和轴标签
    plt.title("Causal Relationship Matrix", fontsize=14, pad=20)
    plt.xlabel("Effect (Outcome)", fontsize=12)
    plt.ylabel("Cause (Input)", fontsize=12)
    
    # 反转Y轴
    ax.invert_yaxis()
    
    # 8. 细节优化：添加网格线让格子更明显
    ax.set_xticks(np.arange(len(matrix.columns)+1)-0.5, minor=True)
    ax.set_yticks(np.arange(len(matrix.index)+1)-0.5, minor=True)
    ax.grid(which="minor", color="lightgray", linestyle='-', linewidth=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    # 自动调整布局，防止标签超出画布边界
    plt.tight_layout()
    plt.show()

# 调用函数
# plot_custom_matrix('../data/virtual/fMRI/sim3_gt_processed.csv')
plot_custom_matrix('../data/virtual/causal_relations.csv')
# plot_custom_matrix('../data/161.csv')