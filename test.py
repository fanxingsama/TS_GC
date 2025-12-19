import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def create_granger_causality_matrix():
    """
    创建格兰杰因果矩阵可视化
    """
    # 示例数据：5x5格兰杰因果矩阵，按照图像的强度分布
    granger_matrix = np.array([
        [0.7, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.7, 0.8, 0.0, 1.0],
        [0.0, 0.0, 0.0, 0.9, 1.0],
        [0.0, 0.8, 0.0, 0.8, 0.0],
        [0.0, 0.0, 0.0, 0.6, 0.0]
        # [0.7, 0.5, 0.2, 0.4, 0.2],
        # [0.2, 0.7, 0.8, 0.2, 1.0],
        # [0.5, 0.2, 0.4, 0.9, 1.0],
        # [0.3, 0.8, 0.2, 0.8, 0.2],
        # [0.4, 0.5, 0.3, 0.6, 0.4]
    ])
    
    # 创建自定义色彩映射 - 从浅蓝到深蓝
    colors = ['#FFFFFF', '#E8F4FD', '#B8D4E3', '#7FB3D3', '#4682B4', '#1E3A8A']
    n_bins = 100
    cmap = LinearSegmentedColormap.from_list('blue_gradient', colors, N=n_bins)
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 绘制热力图，移除边框和刻度
    im = ax.imshow(granger_matrix, cmap=cmap, aspect='equal', vmin=0, vmax=1)
    
    # 移除刻度和标签
    ax.set_xticks([])
    ax.set_yticks([])
    
    # 移除边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    # 不显示数值标注
    
    # 不添加颜色条、标题和标签
    
    # 不添加网格线
    
    plt.tight_layout()
    return fig, granger_matrix


# 执行示例
if __name__ == "__main__":
    # 创建示例矩阵图
    fig, matrix = create_granger_causality_matrix()
    plt.show()
    
    
    print(f"\n生成的矩阵:\n{matrix}")
