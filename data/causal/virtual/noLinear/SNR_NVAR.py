import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def generate_nonlinear_data_with_noise(p=8, T=2000, lag=3, sd=0.1, seed=0, snr_list=[20, 10, 5, 0, -5]):
    np.random.seed(seed)
    burn_in = 500
    total = T + burn_in

    x = np.zeros((p, total))
    eps = np.random.normal(scale=sd, size=(p, total))
    x[:, :lag] = np.random.normal(scale=0.1, size=(p, lag))

    # 生成原始时间序列
    for t in range(lag, total):
        l1 = x[:, t - 1]
        l2 = x[:, t - 2]
        l3 = x[:, t - 3]

        # X1: 自回归 tanh(X1,t-1) + X3,t-2
        x[0, t] = 0.7 * np.tanh(l1[0]) + 0.4 * l2[2] + eps[0, t]
        # X2: 自回归 X2,t-1 + 乘积耦合(X1,t-1 × X5,t-3)
        x[1, t] = 0.5 * l1[1] + 0.3 * (l1[0] * l3[4]) + eps[1, t]
        # X3: 自回归 sin(X3,t-2) - X1,t-1
        x[2, t] = 0.5 * np.sin(l2[2]) - 0.4 * l1[0] + eps[2, t]
        # X4: 自回归 X4,t-1 + X3,t-2^2 - X2,t-1
        x[3, t] = 0.5 * l1[3] + 0.2 * (l2[2] ** 2) - 0.4 * l1[1] + eps[3, t]
        # X5: X3,t-1 + 自回归 tanh(X5,t-2) - X4,t-1
        x[4, t] = 0.4 * l1[2] + 0.5 * np.tanh(l2[4]) - 0.3 * l1[3] + eps[4, t]
        # X6: 自回归 X6,t-1 + sin(X4,t-2) - X5,t-1
        x[5, t] = 0.4 * l1[5] + 0.4 * np.sin(l2[3]) - 0.3 * l1[4] + eps[5, t]
        # X7: 自回归 X7,t-1 + 乘积耦合(X6,t-1 × X2,t-3) - X5,t-2
        x[6, t] = 0.5 * l1[6] + 0.3 * (l1[5] * l3[1]) - 0.4 * l2[4] + eps[6, t]
        # X8: 自回归 X8,t-1 + X6,t-2^2 - X7,t-3
        x[7, t] = 0.4 * l1[7] + 0.2 * (l2[5] ** 2) - 0.3 * l3[6] + eps[7, t]

    x_final = x.T[burn_in:]
    
    # 将无噪声的原始信号进行标准化（此时信号功率 P_signal 方差约为 1）
    x_final_clean = StandardScaler().fit_transform(x_final)

    # 1. 保存完全无噪声的基准数据 (相当于理论上限)
    pd.DataFrame(np.round(x_final_clean, 3), columns=[f'x{i}' for i in range(p)]).to_csv(
        'time_series_nolinear_clean.csv', index=False)
    print(f"✅ 基准无噪声时序数据已保存: time_series_nolinear_clean.csv")

    # 2. 根据不同的信噪比要求，循环添加白噪声并保存
    for snr in snr_list:
        # 当 snr = -5 时，10^(-0.5) 约等于 0.316，所以 noise_var = 1 / 0.316 约等于 3.16
        # 此时噪声的方差(功率)是信号方差的3倍多，属于极度恶劣的环境
        noise_var = 1.0 / (10 ** (snr / 10))
        noise_std = np.sqrt(noise_var)
        
        # 生成与信号 shape 相同的正态分布白噪声
        noise = np.random.normal(loc=0, scale=noise_std, size=x_final_clean.shape)
        
        # 在原始信号上叠加噪声
        x_noisy = x_final_clean + noise
        
        # 建议：加噪后整体数据的方差会被改变，再次进行 StandardScaler() 保持量纲统一
        x_noisy_scaled = StandardScaler().fit_transform(x_noisy)
        
        # 处理文件命名（针对 -5 会生成类似 time_series_nolinear_snr-5.csv 的文件）
        file_name = f'time_series_nolinear_snr{snr}.csv'
        pd.DataFrame(np.round(x_noisy_scaled, 3), columns=[f'x{i}' for i in range(p)]).to_csv(
            file_name, index=False)
        print(f"✅ 加噪数据已保存 (SNR={snr:2}dB, 噪声方差约={noise_var:.2f}): {file_name}")

    # 3. 因果边（含自回归）固有的因果关系不随观测噪声改变
    causal_links = [
        # 自回归边
        ['x0', 'x0', 1], ['x1', 'x1', 1], ['x2', 'x2', 1], ['x3', 'x3', 1],
        ['x4', 'x4', 1], ['x5', 'x5', 1], ['x6', 'x6', 1], ['x7', 'x7', 1],
        # 跨变量因果边
        ['x2', 'x0', 1],   # X3 → X1
        ['x0', 'x1', 1],   # X1 → X2
        ['x4', 'x1', 1],   # X5 → X2
        ['x0', 'x2', 1],   # X1 → X3
        ['x2', 'x3', 1],   # X3 → X4
        ['x1', 'x3', 1],   # X2 → X4
        ['x2', 'x4', 1],   # X3 → X5
        ['x3', 'x4', 1],   # X4 → X5
        ['x3', 'x5', 1],   # X4 → X6
        ['x4', 'x5', 1],   # X5 → X6
        ['x5', 'x6', 1],   # X6 → X7
        ['x1', 'x6', 1],   # X2 → X7
        ['x4', 'x6', 1],   # X5 → X7
        ['x5', 'x7', 1],   # X6 → X8
        ['x6', 'x7', 1],   # X7 → X8
    ]
    pd.DataFrame(causal_links).to_csv('causal_nolinear.csv', index=False, header=False)
    print(f"✅ 因果关系图已保存: causal_nolinear.csv  共 {len(causal_links)} 条边")

# 运行代码
generate_nonlinear_data_with_noise(snr_list=[20, 10, 5, 0, -5])