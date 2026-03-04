import numpy as np
import pandas as pd

def simulate_advanced_var(p=10, T=1000, lag=3, sparsity=0.2, beta_value=0.5, sd=0.1, seed=42):
    """
    生成平稳的 VAR(p) 模型数据并输出方程描述
    """
    np.random.seed(seed)
    
    # 1. 构建因果结构矩阵 (GC)
    GC = np.eye(p, dtype=int)
    # 每个变量额外受到 p * sparsity 个其他变量的影响
    num_other_influences = int(p * sparsity)
    for i in range(p):
        choices = np.random.choice([j for j in range(p) if j != i], size=num_other_influences, replace=False)
        GC[i, choices] = 1

    # 2. 初始化系数矩阵 beta (大小为 p x (p*lag))
    beta = np.zeros((p, p * lag))
    for l in range(lag):
        # 每一阶滞后都遵循 GC 结构，但权重随滞后增加而衰减
        decay = 0.8 ** l 
        beta[:, l*p:(l+1)*p] = (np.random.uniform(0.1, beta_value, (p, p)) * GC) * decay

    # 3. 平稳化处理 (确保特征值 < 1，数据不爆炸)
    def make_stationary(b):
        p_sub = b.shape[0]
        lag_sub = b.shape[1] // p_sub
        companion = np.vstack([b, np.hstack([np.eye(p_sub*(lag_sub-1)), np.zeros((p_sub*(lag_sub-1), p_sub))])])
        egg = np.linalg.eigvals(companion)
        if np.max(np.abs(egg)) >= 0.97:
            return make_stationary(b * 0.95)
        return b

    beta = make_stationary(beta)

    # 4. 迭代生成序列
    X = np.zeros((T + 100, p))
    for t in range(lag, T + 100):
        # 基础公式: Xt = A1*Xt-1 + A2*Xt-2 + ... + Noise
        res = np.zeros(p)
        for l in range(1, lag + 1):
            res += beta[:, (l-1)*p : l*p] @ X[t-l]
        X[t] = res + np.random.normal(0, sd, size=p)

    X_final = X[100:] # 舍弃前100个点以稳定

    # --- 打印 10 个序列的数学方程 ---
    print("="*30)
    print("生成的 10 个序列数学方程如下:")
    print("="*30)
    for i in range(p):
        equation = f"X{i}[t] = "
        terms = []
        for l in range(1, lag + 1):
            coeffs = beta[i, (l-1)*p : l*p]
            for j, val in enumerate(coeffs):
                if abs(val) > 0.001: # 只显示非零项
                    terms.append(f"{val:.3f}*X{j}[t-{l}]")
        equation += " + ".join(terms) + f" + e{i}[t]"
        print(equation)
        print("-" * 10)
    
    return X_final, GC

# 执行生成
data, gc_matrix = simulate_advanced_var(p=10, T=1000)

# 保存到本地
df = pd.DataFrame(data, columns=[f'Series_{i}' for i in range(10)])
df.to_csv('multi_series_data.csv', index=False)
print("\n数据已保存至 'multi_series_data.csv'")