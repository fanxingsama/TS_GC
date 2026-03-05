import numpy as np
import pandas as pd

def make_var_stationary(beta, radius=0.97):
    '''缩放 VAR 模型的系数以保证其在线性主干上的稳定性。'''
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

def simulate_nonlinear_var(p, T, lag, sparsity=0.2, beta_value=1.0, sd=0.1, seed=0):
    '''
    参考原始脚本结构，生成非线性 VAR 时间序列数据。
    引入了 tanh(x) 非线性变换。
    '''
    if seed is not None:
        np.random.seed(seed)

    # 1. 设置系数矩阵和因果关系 Ground Truth (与原脚本一致)
    GC = np.eye(p, dtype=int)
    beta = np.eye(p) * beta_value

    num_nonzero = int(p * sparsity) - 1
    for i in range(p):
        choice = np.random.choice(p - 1, size=num_nonzero, replace=False)
        choice[choice >= i] += 1
        beta[i, choice] = beta_value
        GC[i, choice] = 1

    # 2. 线性权重初始化与平稳化
    beta_stacked = np.hstack([beta * (0.8**l) for l in range(lag)]) # 加入衰减增强真实性
    beta_stationary = make_var_stationary(beta_stacked)

    # 3. 数据生成 (引入非线性)
    burn_in = 100
    errors = np.random.normal(scale=sd, size=(p, T + burn_in))
    X = np.zeros((p, T + burn_in))
    X[:, :lag] = errors[:, :lag]
    
    for t in range(lag, T + burn_in):
        # 核心修改点：在因果传递中加入 np.tanh 非线性
        # 我们对所有来自其他变量的输入应用非线性激活
        past_states = X[:, (t-lag):t].flatten(order='F')
        
        # 线性部分 + 非线性激活部分的组合
        # 这里模拟的是每个组件受过去影响时带有一定的非线性饱和效应
        X[:, t] = np.dot(beta_stationary, np.tanh(past_states)) 
        X[:, t] += errors[:, t-1]

    return X.T[burn_in:], beta_stationary, GC

# --- 脚本执行与原逻辑保持一致 ---
series_num = 10       
series_length = 500  
lag = 3               
sparsity = 0.2        
beta_value = 1.0      
seed = 0              
sd = 0.1              

# 生成非线性数据
X, beta, GC = simulate_nonlinear_var(
    p=series_num, 
    T=series_length, 
    lag=lag, 
    sparsity=sparsity, 
    beta_value=beta_value, 
    sd=sd,
    seed=seed
) 

# ========== 关键修改1：时序数据CSV列名改为x0、x1... ==========
df = pd.DataFrame(np.round(X, 3), columns=[f'x{i}' for i in range(series_num)])
df.to_csv('time_series_nonlinear.csv', index=False)
print(f"时序数据已保存到 time_series_nonlinear.csv")

# ========== 关键修改2：因果关系CSV中变量名改为x0、x1... ==========
# 替换原granger_causes生成逻辑，将数字编号改为x开头
granger_causes = [
    [f'x{j}', f'x{i}', 1]  # j是因变量编号，i是果变量编号，都改为x前缀
    for i in range(series_num) 
    for j in range(series_num) 
    if GC[i, j] == 1
]
pd.DataFrame(granger_causes).to_csv('causality_nonlinear.csv', index=False, header=False)
print(f"因果关系已保存到 causality_nonlinear.csv")

print(f"非线性数据已生成。形状: {X.shape}")
print(f"数据质量检查 (Mean/Std): {X.mean():.3f} / {X.std():.3f}")

# 验证平稳性
chunk_size = 100
chunk_vars = [X[i:i+chunk_size].var(axis=0).mean() for i in range(0, len(X)-chunk_size, chunk_size)]
print(f"不同时间段的方差波动: {np.std(chunk_vars):.6f}")