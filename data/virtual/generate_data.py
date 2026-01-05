import numpy as np
import pandas as pd

def make_var_stationary(beta, radius=0.97):
    '''Rescale coefficients of VAR model to make stable.'''
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

# 模拟一个VAR模型的时间序列数据。
def simulate_var(p, T, lag, sparsity=0.2, beta_value=1.0, sd=0.1, seed=0):
    if seed is not None:
        np.random.seed(seed)

    # Set up coefficients and Granger causality ground truth.
    GC = np.eye(p, dtype=int)
    beta = np.eye(p) * beta_value

    num_nonzero = int(p * sparsity) - 1
    for i in range(p):
        choice = np.random.choice(p - 1, size=num_nonzero, replace=False)
        choice[choice >= i] += 1
        beta[i, choice] = beta_value
        GC[i, choice] = 1

    beta = np.hstack([beta for _ in range(lag)])
    beta = make_var_stationary(beta)

    # Generate data.
    burn_in = 100
    errors = np.random.normal(scale=sd, size=(p, T + burn_in))
    X = np.zeros((p, T + burn_in))
    X[:, :lag] = errors[:, :lag]
    for t in range(lag, T + burn_in):
        X[:, t] = np.dot(beta, X[:, (t-lag):t].flatten(order='F'))
        X[:, t] += + errors[:, t-1]

    return X.T[burn_in:], beta, GC

# 生成数据集参数
series_num = 10       # 时间序列数量
series_length = 1000  # 数据点数量
lag = 3               # 时间延迟
sparsity = 0.2        # 因果关系稀疏程度
beta_value = 1.0      # 基础系数值
seed = 0              # 随机种子
sd = 0.1              # 噪声标准差

# 生成时间序列数据和因果关系矩阵
X, beta, GC = simulate_var(
    p=series_num,        # 修正：使用正确的参数名
    T=series_length,     # 修正：使用正确的参数名
    lag=lag, 
    sparsity=sparsity, 
    beta_value=beta_value, 
    sd=sd,
    seed=seed
) 

print(f"生成的数据形状: {X.shape}")
print(f"格兰杰因果矩阵形状: {GC.shape}")
print(f"格兰杰因果矩阵:\n{GC}")

# 将时间序列数据四舍五入到3位小数并保存为CSV
df = pd.DataFrame(np.round(X, 3), columns=[f'{i}' for i in range(series_num)])
df.to_csv('time_series1000.csv', index=False)

# 提取和保存格兰杰因果关系
granger_causes = []
for i in range(series_num):
    for j in range(series_num):
        if GC[i, j] == 1:
            # 格式：因变量索引，果变量索引，强度（这里用1表示存在因果关系）
            granger_causes.append([j, i, 1])  # j影响i

# 保存格兰杰因果关系为CSV
gc_df = pd.DataFrame(granger_causes)
gc_df.to_csv('granger_causality1000.csv', index=False)

# 显示因果关系统计
print(f"\n因果关系统计:")
print(f"总共的因果关系数量: {len(granger_causes)}")
print(f"预期的因果关系数量: {series_num + series_num * (series_num - 1) * sparsity}")
print(f"实际稀疏度: {len(granger_causes) / (series_num * series_num):.3f}")

# 验证数据质量
print(f"\n数据质量检查:")
print(f"数据范围: [{X.min():.3f}, {X.max():.3f}]")
print(f"数据均值: {X.mean():.3f}")
print(f"数据标准差: {X.std():.3f}")

# 检查平稳性（简单检查方差是否稳定）
chunk_size = 100
chunks = [X[i:i+chunk_size] for i in range(0, len(X)-chunk_size, chunk_size)]
chunk_vars = [chunk.var(axis=0).mean() for chunk in chunks]
print(f"不同时间段的方差变化: {np.std(chunk_vars):.6f} (越小越好)")