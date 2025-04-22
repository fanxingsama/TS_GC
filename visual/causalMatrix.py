import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 从单个CSV文件绘制因果矩阵图
def visualize_single_causality_csv(csv_path):
    try:
        df = pd.read_csv(csv_path)
        # 确保CSV文件包含所需的列
        assert 'source' in df.columns and 'target' in df.columns, "CSV文件必须包含 'source' 和 'target' 列"
        
        # 检查是否有延迟列
        has_delay = False
        if len(df.columns) > 2 and df.columns[2] != '':
            has_delay = True
            delay_column = df.columns[2]
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
    
    # 如果有延迟列，创建延迟矩阵
    if has_delay:
        delay_matrix = np.zeros((n, n))
        delay_matrix.fill(np.nan)  # 用NaN填充以表示无关系
    
    # 填充因果矩阵
    for _, row in df.iterrows():
        source_idx = node_to_idx[row['source']]
        target_idx = node_to_idx[row['target']]
        causality_matrix[target_idx, source_idx] = 1  # target受source影响
        
        # 如果有延迟信息，填充延迟矩阵
        if has_delay:
            delay_matrix[target_idx, source_idx] = row[delay_column]
    
    # 创建图形并设置大小
    plt.figure(figsize=(10, 8))
    
    # 可视化因果矩阵
    cax = plt.imshow(causality_matrix, cmap='Blues', interpolation='nearest')
    plt.colorbar(cax)
    
    # 添加标题和轴标签
    plt.title("Causality Matrix")
    plt.xlabel('Causal series (Source)')
    plt.ylabel('Affected series (Target)')
    
    # 添加节点标签
    plt.xticks(np.arange(n), all_nodes, rotation=45, ha="right")
    plt.yticks(np.arange(n), all_nodes)
    
    # 添加网格线以便更好地区分单元格
    plt.grid(False)
    
    # 如果有延迟列，在每个有因果关系的单元格中添加延迟值
    if has_delay:
        for i in range(n):
            for j in range(n):
                if causality_matrix[i, j] > 0:
                    text_color = "white" if causality_matrix[i, j] > 0.5 else "black"
                    plt.text(j, i, f"{delay_matrix[i, j]}", 
                             ha="center", va="center", color=text_color)
    
    plt.tight_layout()
    plt.show()
    
    if has_delay:
        return causality_matrix, delay_matrix, all_nodes
    else:
        return causality_matrix, all_nodes

# 比较两个因果关系CSV文件，并标出不一致的地方
def compare_causality_csvs(true_csv_path, estimated_csv_path):
    try:
        true_df = pd.read_csv(true_csv_path)
        estimated_df = pd.read_csv(estimated_csv_path)
        
        # 确保CSV文件包含所需的列
        assert 'source' in true_df.columns and 'target' in true_df.columns, "真实CSV必须包含 'source' 和 'target' 列"
        assert 'source' in estimated_df.columns and 'target' in estimated_df.columns, "估计CSV必须包含 'source' 和 'target' 列"
        
        # 检查是否有延迟列
        has_delay_true = False
        has_delay_est = False
        
        if len(true_df.columns) > 2 and true_df.columns[2] != '':
            has_delay_true = True
            delay_column_true = true_df.columns[2]
            
        if len(estimated_df.columns) > 2 and estimated_df.columns[2] != '':
            has_delay_est = True
            delay_column_est = estimated_df.columns[2]
            
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
    
    # 如果有延迟列，创建延迟矩阵
    if has_delay_true:
        true_delay_matrix = np.zeros((n, n))
        true_delay_matrix.fill(np.nan)
        
    if has_delay_est:
        est_delay_matrix = np.zeros((n, n))
        est_delay_matrix.fill(np.nan)
    
    # 填充真实因果矩阵
    for _, row in true_df.iterrows():
        source_idx = node_to_idx[row['source']]
        target_idx = node_to_idx[row['target']]
        true_matrix[target_idx, source_idx] = 1
        
        # 如果有延迟信息，填充延迟矩阵
        if has_delay_true:
            true_delay_matrix[target_idx, source_idx] = row[delay_column_true]
    
    # 填充估计因果矩阵
    for _, row in estimated_df.iterrows():
        source_idx = node_to_idx[row['source']]
        target_idx = node_to_idx[row['target']]
        estimated_matrix[target_idx, source_idx] = 1
        
        # 如果有延迟信息，填充延迟矩阵
        if has_delay_est:
            est_delay_matrix[target_idx, source_idx] = row[delay_column_est]
    
    # 创建图形
    fig, axarr = plt.subplots(1, 2, figsize=(16, 7))
    
    # 可视化真实的因果矩阵
    im1 = axarr[0].imshow(true_matrix, cmap='Blues', interpolation='nearest')
    axarr[0].set_title('True Causality Matrix')
    axarr[0].set_ylabel('Affected series (Target)')
    axarr[0].set_xlabel('Causal series (Source)')
    
    # 可视化估计的因果矩阵
    im2 = axarr[1].imshow(estimated_matrix, cmap='Blues', interpolation='nearest', vmin=0, vmax=1)
    axarr[1].set_title('Estimated Causality Matrix')
    axarr[1].set_ylabel('Affected series (Target)')
    axarr[1].set_xlabel('Causal series (Source)')
    
    # 添加颜色条
    fig.colorbar(im1, ax=axarr[0], fraction=0.046, pad=0.04)
    fig.colorbar(im2, ax=axarr[1], fraction=0.046, pad=0.04)
    
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
            
            # 如果有延迟信息，在相应单元格中添加延迟值
            if has_delay_true and true_matrix[i, j] > 0:
                text_color = "white" if true_matrix[i, j] > 0.5 else "black"
                axarr[0].text(j, i, f"{true_delay_matrix[i, j]}", 
                             ha="center", va="center", color=text_color)
                
            if has_delay_est and estimated_matrix[i, j] > 0:
                text_color = "white" if estimated_matrix[i, j] > 0.5 else "black"
                axarr[1].text(j, i, f"{est_delay_matrix[i, j]}", 
                             ha="center", va="center", color=text_color)
    
    plt.tight_layout()
    plt.show()
    
    result = [true_matrix, estimated_matrix, all_nodes]
    if has_delay_true:
        result.append(true_delay_matrix)
    if has_delay_est:
        result.append(est_delay_matrix)
    
    return tuple(result)

# 示例用法
if __name__ == "__main__":
    gt_path = '../data/fMRI/sim15_gt_processed.csv'
    file_path = '../saved/models/0410_152831/FMRI15/causal_structure.csv'
    
    visualize_single_causality_csv(file_path)

    # # 创建示例数据
    # n = 5  # 5个节点
    # nodes = [f"Node_{i}" for i in range(n)]
    
    # # 生成真实因果关系，含延迟列
    # sources_true = []
    # targets_true = []
    # delays_true = []
    
    # for i in range(n):
    #     for j in range(n):
    #         if i != j and np.random.random() < 0.3:  # 30%的概率有因果关系
    #             sources_true.append(f"Node_{i}")
    #             targets_true.append(f"Node_{j}")
    #             delays_true.append(np.random.randint(1, 5))  # 随机1-4的延迟
    
    # # 生成估计因果关系（有些关系可能被遗漏或错误添加），含延迟列
    # sources_est = sources_true.copy()
    # targets_est = targets_true.copy()
    # delays_est = delays_true.copy()
    
    # # 随机删除一些关系
    # for _ in range(1):
    #     if len(sources_est) > 0:
    #         idx = np.random.randint(0, len(sources_est))
    #         sources_est.pop(idx)
    #         targets_est.pop(idx)
    #         delays_est.pop(idx)
    
    # # 随机添加一些错误关系
    # for _ in range(2):
    #     i = np.random.randint(0, n)
    #     j = np.random.randint(0, n)
    #     if i != j:
    #         new_pair = True
    #         for idx, (s, t) in enumerate(zip(sources_est, targets_est)):
    #             if s == f"Node_{i}" and t == f"Node_{j}":
    #                 new_pair = False
    #                 break
    #         if new_pair:
    #             sources_est.append(f"Node_{i}")
    #             targets_est.append(f"Node_{j}")
    #             delays_est.append(np.random.randint(1, 5))
    
    # # 创建并保存含延迟的示例CSV
    # true_df_with_delay = pd.DataFrame({
    #     'source': sources_true, 
    #     'target': targets_true, 
    #     'delay': delays_true
    # })
    # true_df_with_delay.to_csv("true_causality_with_delay.csv", index=False)
    
    # est_df_with_delay = pd.DataFrame({
    #     'source': sources_est, 
    #     'target': targets_est, 
    #     'delay': delays_est
    # })
    # est_df_with_delay.to_csv("estimated_causality_with_delay.csv", index=False)
    
    # # 创建并保存不含延迟的示例CSV
    # true_df_no_delay = pd.DataFrame({'source': sources_true, 'target': targets_true})
    # true_df_no_delay.to_csv("true_causality_no_delay.csv", index=False)
    
    # est_df_no_delay = pd.DataFrame({'source': sources_est, 'target': targets_est})
    # est_df_no_delay.to_csv("estimated_causality_no_delay.csv", index=False)
    
    # # 演示方法1: 可视化单个CSV文件 (带延迟)
    # print("\n演示方法1-A: 可视化单个CSV文件 (带延迟)")
    # visualize_single_causality_csv("true_causality_with_delay.csv", "True Causality Matrix with Delay")
    
    # # 演示方法1: 可视化单个CSV文件 (不带延迟)
    # print("\n演示方法1-B: 可视化单个CSV文件 (不带延迟)")
    # visualize_single_causality_csv("true_causality_no_delay.csv", "True Causality Matrix without Delay")
    
    # # 演示方法2: 比较两个CSV文件 (带延迟)
    # print("\n演示方法2-A: 比较两个CSV文件 (带延迟)")
    # compare_causality_csvs("true_causality_with_delay.csv", "estimated_causality_with_delay.csv")
    
    # # 演示方法2: 比较两个CSV文件 (不带延迟)
    # print("\n演示方法2-B: 比较两个CSV文件 (不带延迟)")
    # compare_causality_csvs("true_causality_no_delay.csv", "estimated_causality_no_delay.csv")