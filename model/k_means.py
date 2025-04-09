import numpy as np
from sklearn.cluster import KMeans


# 使用K-means进行因果关系分析并构建因果图
def K_means_analyze(relA, relK, m, n, time_step):
    """
        relA (List[torch.Tensor]): 每个时间序列的注意力矩阵的相关性分数。
        relK (List[torch.Tensor]): 每个时间序列的因果卷积核的相关性分数。
        m (int): 考虑的顶部聚类数量。
        n (int): 总聚类数量。
        time_step (int): 输入的时间步数。

    Returns:
        ans (List[Tuple[int, int, int]]): 表示因果图边的元组列表（原因、效果、滞后）
    """
    estimator = KMeans(n_clusters=n) # 搭建K-means模型
    ans = []
    # find causes of series i
    for i,relAi in enumerate(relA):
        if relAi.sum()==0.0: # all the weights to series i are zero
            continue
        data=np.array(relAi)
        estimator.fit(data.reshape(-1,1))
        cluster_labels = estimator.labels_
        cluster_centers = estimator.cluster_centers_
        cluster_centers = cluster_centers.reshape(-1)
        largest_m_clusters = np.argsort(cluster_centers)[-m:]
        for j in range(len(relAi)):
            if cluster_labels[j] in largest_m_clusters:
                relKij = relK[i][j]
                indices = np.argsort(-1 * relKij)
                ans.append((j,i,time_step-1-indices[0]))
    return ans
