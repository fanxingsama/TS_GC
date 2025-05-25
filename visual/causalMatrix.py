import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 从单个CSV文件绘制因果矩阵图
def visualize_single_causality_csv(csv_path, png_path=False, show=True):
    try:
        df = pd.read_csv(csv_path)
        # 确保CSV文件包含所需的列
        assert 'source' in df.columns and 'target' in df.columns, "CSV文件必须包含 'source' 和 'target' 列"
        
        # 检查是否有强度列
        has_strength = False
        if 'Strength' in df.columns:
            has_strength = True
            strength_column = 'Strength'
        elif len(df.columns) > 2:
            # 如果有第三列但不叫Strength，假设它是强度列
            has_strength = True
            strength_column = df.columns[2]
    except Exception as e:
        print(f"读取CSV文件出错: {e}")
        return
    
    # 获取所有唯一的节点
    all_nodes = sorted(list(set(df['source'].unique()) | set(df['target'].unique())))
    n = len(all_nodes)
    
    # 创建节点索引映射
    node_to_idx = {node: i for i, node in enumerate(all_nodes)}
    
    # 初始化因果矩阵 (行=受影响的，列=影响源)
    causality_matrix = np.zeros((n, n))
    
    # 如果有强度列，创建强度矩阵
    if has_strength:
        strength_matrix = np.zeros((n, n))
        strength_matrix.fill(np.nan)  # 用NaN填充以表示无关系
    
    # 填充因果矩阵
    for _, row in df.iterrows():
        source_idx = node_to_idx[row['source']]
        target_idx = node_to_idx[row['target']]
        causality_matrix[target_idx, source_idx] = 1  # target受source影响
        
        # 如果有强度信息，填充强度矩阵
        if has_strength:
            strength_matrix[target_idx, source_idx] = row[strength_column]
    
    # 创建图形并设置大小
    plt.figure(figsize=(8, 7))
    
    # 可视化因果矩阵 - 使用固定的深蓝色，不显示颜色条
    plt.imshow(causality_matrix, cmap='Blues', interpolation='nearest', vmin=0, vmax=1)
    
    # 添加标题和轴标签
    plt.title("Causality Matrix")
    plt.xlabel('Causal series (Source)')
    plt.ylabel('Affected series (Target)')
    
    # 添加节点标签
    plt.xticks(np.arange(n), all_nodes, rotation=45, ha="right")
    plt.yticks(np.arange(n), all_nodes)
    
    # 添加网格线以便更好地区分单元格
    plt.grid(False)
    
    # 添加网格线
    for i in range(n + 1):
        plt.axhline(y=i - 0.5, color='white', linewidth=0.5)
        plt.axvline(x=i - 0.5, color='white', linewidth=0.5)
    
    # 如果有强度列，在每个有因果关系的单元格中添加强度值
    if has_strength:
        for i in range(n):
            for j in range(n):
                if causality_matrix[i, j] > 0:
                    # 在深蓝色背景上使用白色文字
                    plt.text(j, i, f"{strength_matrix[i, j]:.2f}", 
                             ha="center", va="center", color="white", fontweight='bold')
    
    plt.tight_layout()
    if show:
        plt.show()
    if png_path:
        plt.savefig(png_path)
    
    if has_strength:
        return causality_matrix, strength_matrix, all_nodes
    else:
        return causality_matrix, all_nodes

# 比较两个因果关系CSV文件，并标出不一致的地方
def compare_causality_csvs(true_csv_path, estimated_csv_path, png_path=False, show=True):
    try:
        true_df = pd.read_csv(true_csv_path)
        estimated_df = pd.read_csv(estimated_csv_path)
        
        # 确保CSV文件包含所需的列
        assert 'source' in true_df.columns and 'target' in true_df.columns, "真实CSV必须包含 'source' 和 'target' 列"
        assert 'source' in estimated_df.columns and 'target' in estimated_df.columns, "估计CSV必须包含 'source' 和 'target' 列"
        
        # 检查是否有强度列
        has_strength_true = False
        has_strength_est = False
        
        if 'Strength' in true_df.columns:
            has_strength_true = True
            strength_column_true = 'Strength'
        elif len(true_df.columns) > 2:
            has_strength_true = True
            strength_column_true = true_df.columns[2]
            
        if 'Strength' in estimated_df.columns:
            has_strength_est = True
            strength_column_est = 'Strength'
        elif len(estimated_df.columns) > 2:
            has_strength_est = True
            strength_column_est = estimated_df.columns[2]
            
    except Exception as e:
        print(f"读取CSV文件出错: {e}")
        return
    
    # 获取所有唯一的节点（两个文件合并）
    all_nodes = sorted(list(set(true_df['source'].unique()) | 
                           set(true_df['target'].unique()) | 
                           set(estimated_df['source'].unique()) | 
                           set(estimated_df['target'].unique())))
    n = len(all_nodes)
    
    # 创建节点索引映射
    node_to_idx = {node: i for i, node in enumerate(all_nodes)}
    
    # 初始化因果矩阵
    true_matrix = np.zeros((n, n))
    estimated_matrix = np.zeros((n, n))
    
    # 如果有强度列，创建强度矩阵
    if has_strength_true:
        true_strength_matrix = np.zeros((n, n))
        true_strength_matrix.fill(np.nan)
        
    if has_strength_est:
        est_strength_matrix = np.zeros((n, n))
        est_strength_matrix.fill(np.nan)
    
    # 填充真实因果矩阵
    for _, row in true_df.iterrows():
        source_idx = node_to_idx[row['source']]
        target_idx = node_to_idx[row['target']]
        true_matrix[target_idx, source_idx] = 1
        
        # 如果有强度信息，填充强度矩阵
        if has_strength_true:
            true_strength_matrix[target_idx, source_idx] = row[strength_column_true]
    
    # 填充估计因果矩阵
    for _, row in estimated_df.iterrows():
        source_idx = node_to_idx[row['source']]
        target_idx = node_to_idx[row['target']]
        estimated_matrix[target_idx, source_idx] = 1
        
        # 如果有强度信息，填充强度矩阵
        if has_strength_est:
            est_strength_matrix[target_idx, source_idx] = row[strength_column_est]
    
    # 创建图形
    fig, axarr = plt.subplots(1, 2, figsize=(16, 7))
    
    # 可视化真实的因果矩阵 - 使用固定颜色，不显示颜色条
    axarr[0].imshow(true_matrix, cmap='Blues', interpolation='nearest', vmin=0, vmax=1)
    axarr[0].set_title('True Causality Matrix')
    axarr[0].set_ylabel('Affected series (Target)')
    axarr[0].set_xlabel('Causal series (Source)')
    
    # 可视化估计的因果矩阵 - 使用固定颜色，不显示颜色条
    axarr[1].imshow(estimated_matrix, cmap='Blues', interpolation='nearest', vmin=0, vmax=1)
    axarr[1].set_title('Estimated Causality Matrix')
    axarr[1].set_ylabel('Affected series (Target)')
    axarr[1].set_xlabel('Causal series (Source)')
    
    # 添加节点标签
    for ax in axarr:
        ax.set_xticks(np.arange(n))
        ax.set_yticks(np.arange(n))
        ax.set_xticklabels(all_nodes, rotation=45, ha="right")
        ax.set_yticklabels(all_nodes)
    
    # 标记不一致的因果关系
    for i in range(n):
        for j in range(n):
            # 在估计矩阵中标红不一致的地方
            if true_matrix[i, j] != estimated_matrix[i, j]:
                rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, 
                                    facecolor='none', edgecolor='red', linewidth=2)
                axarr[1].add_patch(rect)
            
            # 如果有强度信息，在相应单元格中添加强度值
            if has_strength_true and true_matrix[i, j] > 0:
                axarr[0].text(j, i, f"{true_strength_matrix[i, j]:.2f}", 
                             ha="center", va="center", color="white", fontweight='bold')
                
            if has_strength_est and estimated_matrix[i, j] > 0:
                axarr[1].text(j, i, f"{est_strength_matrix[i, j]:.2f}", 
                             ha="center", va="center", color="white", fontweight='bold')
    
    plt.tight_layout()
    if show:
        plt.show()
    if png_path:
        plt.savefig(png_path)
    
    result = [true_matrix, estimated_matrix, all_nodes]
    if has_strength_true:
        result.append(true_strength_matrix)
    if has_strength_est:
        result.append(est_strength_matrix)
    
    return tuple(result)
