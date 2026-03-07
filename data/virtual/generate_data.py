import numpy as np
import pandas as pd
import os

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
    
    # 1. 构建包含所有滞后信息的完整权重矩阵 (p 行, p * lag 列)
    beta_stacked = np.zeros((p, p * lag))
    
    # 自回归：序列自身的平稳连贯性，设定在 t-1 (即滞后1步)
    for i in range(p):
        beta_stacked[i, i] = np.random.uniform(0.3, 0.5) 
    
    # 交叉因果关系：为每一对关系随机指定一个特定的滞后步数
    for i in range(p):
        num_others = np.random.randint(1, 4) 
        others = np.random.choice([j for j in range(p) if j != i], size=num_others, replace=False)
        for other in others:
            chosen_lag = np.random.randint(1, lag + 1)
            sign = np.random.choice([1, -1]) 
            weight = np.random.uniform(0.2, 0.7) * sign
            
            col_index = (chosen_lag - 1) * p + other
            beta_stacked[i, col_index] = weight
            
    # 2. 构造平稳系统
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
    
    # 4. 提取信息、生成公式并保存到日志和 CSV
    causal_links = []
    log_lines = []
    
    log_lines.append("=" * 80)
    log_lines.append(f"生成的 {p} 维系统数学公式 (保留单一显著滞后信息):")
    log_lines.append("=" * 80)
    
    for i in range(p): # 果 (Target)
        formula_parts = []
        for j in range(p): # 因 (Source)
            has_causal = False
            for l in range(lag):
                val = beta_stationary[i, l * p + j]
                if abs(val) > 1e-4:
                    has_causal = True
                    sign_str = " + " if (len(formula_parts) > 0 and val > 0) else (" - " if val < 0 else "")
                    if len(formula_parts) == 0 and val < 0: sign_str = "-"
                    formula_parts.append(f"{sign_str}{abs(val):.3f}*x{j}[t-{l+1}]")
            
            # 只要在任何一个 lag 上有影响，就在 CSV 中记录这条因果边 (权重统一设为 1)
            if has_causal:
                causal_links.append([f"x{j}", f"x{i}", 1])
        
        full_formula = f"x{i}[t] =" + "".join(formula_parts) + f" + ε_{i}[t]"
        log_lines.append(full_formula)
        log_lines.append("-" * 40)
        
    # 打印日志到控制台
    for line in log_lines:
        print(line)
        
    # 将日志写入 txt 文件
    with open('data_generation_linear_log.txt', 'w', encoding='utf-8') as f:
        f.write("\n".join(log_lines))
        
    # 5. 保存纯净版的三列 CSV (无表头)
    pd.DataFrame(causal_links).to_csv('causal_relations.csv', index=False, header=False)
    
    print(f"\n✅ 数据已生成: time_series_linear.csv")
    print(f"✅ 因果边(3列无表头)已保存至: causal_relations.csv (共 {len(causal_links)} 条边)")
    print(f"✅ 滞后详情日志已保存至: data_generation_linear_log.txt")

# 执行
generate_merged_equations_data()