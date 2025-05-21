import numpy as np
import pandas as pd

def make_var_stationary(beta, radius=0.97):
    p = beta.shape[0]
    lag = beta.shape[1] // p
    bottom = np.hstack((np.eye(p * (lag - 1)), np.zeros((p * (lag - 1), p))))
    beta_tilde = np.vstack((beta, bottom))
    eigvals = np.linalg.eigvals(beta_tilde)
    max_eig = max(np.abs(eigvals))
    nonstationary = max_eig > radius
    if nonstationary:
        return make_var_stationary(0.95 * beta, radius)
    else:
        return beta

# 模拟一个VAR模型的时间序列数据
def simulate_var(series_num, series_length, lag, sparsity=0.2, beta_value=1.0, sd=0.1, seed=0, 
                 distinct_frequencies=True, amplitudes=None):
    if seed is not None:
        np.random.seed(seed)
    
    GC = np.zeros((series_num, series_num), dtype=int)
    beta = np.eye(series_num) * beta_value

    # 创建更有区分度的因果关系
    num_nonzero = int(series_num * sparsity) - 1
    for i in range(series_num):
        choice = np.random.choice(series_num - 1, size=num_nonzero, replace=False)
        choice[choice >= i] += 1
        # 随机生成不同强度的因果关系
        if amplitudes is None:
            cause_betas = np.random.uniform(0.3, 0.7, size=len(choice))
        else:
            cause_betas = np.random.uniform(0.3, 0.7, size=len(choice)) * amplitudes[i]
        
        beta[i, choice] = cause_betas
        GC[choice, i] = 1  # 这里表示choice是i的原因，i是choice的结果

    beta = np.hstack([beta for _ in range(lag)])
    beta = make_var_stationary(beta)
    burn_in = 200  # 增加burn-in时间
    
    # 创建初始噪声，使用不同的标准差增加区分度
    base_sd = sd
    series_sds = np.random.uniform(0.8, 1.2, size=series_num) * base_sd
    
    # 为每个序列添加不同频率的季节性成分
    frequencies = None
    if distinct_frequencies:
        frequencies = np.linspace(0.01, 0.1, series_num)  # 不同的频率
    
    # 生成带有区分度的噪声
    errors = np.zeros((series_num, series_length + burn_in))
    for i in range(series_num):
        errors[i, :] = np.random.normal(scale=series_sds[i], size=series_length + burn_in)
        
        # 添加季节性成分
        if frequencies is not None:
            t = np.arange(series_length + burn_in)
            seasonal = np.sin(2 * np.pi * frequencies[i] * t) * (0.5 + i * 0.05)
            errors[i, :] += seasonal
    
    X = np.zeros((series_num, series_length + burn_in))
    X[:, :lag] = errors[:, :lag]
    
    for t in range(lag, series_length + burn_in):
        X[:, t] = np.dot(beta, X[:, (t-lag):t].flatten(order='F'))
        X[:, t] += errors[:, t-1]
    
    # 归一化到[-5, 5]范围
    X_subset = X.T[burn_in:]
    for i in range(series_num):
        # 确保每个序列都有自己的波动范围，增加区分度
        range_min = -5 + i * 0.05  # 微调每个序列的范围
        range_max = 5 - i * 0.05
        
        # 先归一化到[0,1]
        col_min = np.min(X_subset[:, i])
        col_max = np.max(X_subset[:, i])
        if col_max > col_min:
            X_subset[:, i] = (X_subset[:, i] - col_min) / (col_max - col_min)
            # 然后缩放到目标范围
            X_subset[:, i] = X_subset[:, i] * (range_max - range_min) + range_min

    return X_subset, beta, GC

# 生成数据集参数
series_num = 10       # 时间序列数量
series_length = 10000 # 数据点数量
lag = 2               # 时间延迟
sparsity = 0.3        # 因果关系稀疏程度
beta_value = 0.5      # 基础系数值
seed = 42             # 随机种子

# 为每个序列定义波动振幅的不同值来增加区分度
amplitudes = np.linspace(0.8, 1.5, series_num)

# 生成时间序列数据和因果关系矩阵，使用增强的区分度参数
X, beta, GC = simulate_var(
    series_num=series_num, 
    series_length=series_length, 
    lag=lag, 
    sparsity=sparsity, 
    beta_value=beta_value, 
    sd=0.2,  # 增加基础噪声
    seed=seed,
    distinct_frequencies=True,  # 添加不同频率的季节性
    amplitudes=amplitudes)  # 使用不同的振幅

# 将时间序列数据四舍五入到3位小数并保存为CSV
df = pd.DataFrame(np.round(X, 3), columns=[str(i) for i in range(series_num)])
df.to_csv('series_data2.csv', index=False)

# 提取和保存格兰杰因果关系
granger_causes = []
for i in range(series_num):
    for j in range(series_num):
        if GC[i, j] == 1:
            # 格式：因，果，延迟
            granger_causes.append([i, j, lag])

# 保存格兰杰因果关系为CSV（无表头）
pd.DataFrame(granger_causes).to_csv('granger_causality2.csv', header=False, index=False)

print(f"已生成 {series_length} 个数据点、{series_num} 个时间序列的数据集。")
print(f"发现了 {len(granger_causes)} 个格兰杰因果关系。")
print("时间序列数据已保存至 'timeseries_data.csv'")
print("格兰杰因果关系已保存至 'granger_causality.csv'")

# 显示因果关系预览
print("\n格兰杰因果关系预览（因 -> 果，延迟）:")
for cause in granger_causes:
    print(f"{cause[0]} -> {cause[1]}, lag={cause[2]}")

# 显示数据统计信息
print("\n时间序列数据统计信息:")
print(df.describe().round(3))

# 打印前5行数据查看结果
print("\n时间序列数据前5行预览:")
print(df.head())