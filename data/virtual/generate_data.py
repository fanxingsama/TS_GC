import numpy as np
import pandas as pd

def make_var_stationary(beta, radius=0.97):
    p = beta.shape[0]
    lag = beta.shape[1] // p
    bottom = np.hstack((np.eye(p * (lag - 1)), np.zeros((p * (lag - 1), p))))
    beta_tilde = np.vstack((beta, bottom))
    eigvals = np.linalg.eigvals(beta_tilde)
    max_eig = max(np.abs(eigvals))
    if max_eig > radius:
        return make_var_stationary(0.95 * beta, radius)
    else:
        return beta

def generate_merged_equations_data(p=8, T=1000, lag=3, sd=0.1, seed=0):
    np.random.seed(seed)
    
    # 1. 基础权重生成
    beta_base = np.zeros((p, p))
    for i in range(p):
        beta_base[i, i] = np.random.uniform(0.3, 0.5) # 自回归
    
    for i in range(p):
        num_others = np.random.randint(1, 4) # 每个序列受 1-3 个其他序列影响
        others = np.random.choice([j for j in range(p) if j != i], size=num_others, replace=False)
        for other in others:
            beta_base[i, other] = np.random.uniform(0.2, 0.7)
            
    # 2. 构造平稳系统（3阶滞后权重相同）
    beta_stacked = np.hstack([beta_base for _ in range(lag)])
    beta_stationary = make_var_stationary(beta_stacked)
    
    # 3. 生成数据并保存
    burn_in = 100
    errors = np.random.normal(scale=sd, size=(p, T + burn_in))
    x = np.zeros((p, T + burn_in))
    x[:, :lag] = errors[:, :lag]
    for t in range(lag, T + burn_in):
        x[:, t] = np.dot(beta_stationary, x[:, (t-lag):t].flatten(order='F'))
        x[:, t] += errors[:, t-1]
    
    x_final = x.T[burn_in:]
    pd.DataFrame(np.round(x_final, 3), columns=[f'x{i}' for i in range(p)]).to_csv('time_series_linear.csv', index=False)
    
    # 4. 【核心修改】合并系数并打印公式
    print("\n" + "="*80)
    print(f"生成的 8 维系统合并数学公式:")
    print("="*80)
    
    for i in range(p):
        # 统计每个变量 j 在所有滞后阶数上的系数总和
        total_weights = np.zeros(p)
        for l in range(lag):
            # 提取对应滞后的系数块
            start_col = l * p
            total_weights += beta_stationary[i, start_col : start_col + p]
            
        # 拼接合并后的公式
        formula_parts = []
        for j in range(p):
            val = total_weights[j]
            if abs(val) > 1e-4:
                # 确定正负号
                sign = " + " if (len(formula_parts) > 0 and val > 0) else (" - " if val < 0 else "")
                if len(formula_parts) == 0 and val < 0: sign = "-"
                # 不再显示 [t-1] 等滞后符号，直接显示变量名
                formula_parts.append(f"{sign}{abs(val):.3f}*x{j}")
        
        full_formula = f"x{i}[t] =" + "".join(formula_parts) + f" + ε_{i}[t]"
        print(full_formula)
        print("-" * 40)
        
    # 5. 提取并保存因果关系（新增部分）
    causal_links = []
    for i in range(p): # 果 (Target)
        # 统计每个变量 j 在所有滞后阶数上的系数
        for j in range(p): # 因 (Source)
            # 检查该变量在所有 lag 中是否有非零系数
            is_causal = False
            for l in range(lag):
                if abs(beta_stationary[i, l * p + j]) > 1e-4:
                    is_causal = True
                    break
            
            if is_causal:
                # 第一列是因(xj)，第二列是果(xi)
                causal_links.append([f"x{j}", f"x{i}", 1])
    
    # 转换为 DataFrame 并导出，不包含表头和索引
    pd.DataFrame(causal_links).to_csv('causal_relations.csv', index=False, header=False)
    print(f"\n因果关系已保存至: causal_relations.csv (共 {len(causal_links)} 条边)")

# 执行
generate_merged_equations_data()