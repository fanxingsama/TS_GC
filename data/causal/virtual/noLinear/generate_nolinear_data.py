import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def generate_nonlinear_data(p=8, T=2000, lag=3, sd=0.1, seed=0):
    np.random.seed(seed)
    burn_in = 500
    total = T + burn_in

    x = np.zeros((p, total))
    eps = np.random.normal(scale=sd, size=(p, total))
    x[:, :lag] = np.random.normal(scale=0.1, size=(p, lag))

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
    x_final = StandardScaler().fit_transform(x_final)

    pd.DataFrame(np.round(x_final, 3), columns=[f'x{i}' for i in range(p)]).to_csv(
        'time_series_nolinear.csv', index=False)

    # 因果边（含自回归）
    causal_links = [
        # 自回归边
        ['x0', 'x0', 1],
        ['x1', 'x1', 1],
        ['x2', 'x2', 1],
        ['x3', 'x3', 1],
        ['x4', 'x4', 1],
        ['x5', 'x5', 1],
        ['x6', 'x6', 1],
        ['x7', 'x7', 1],
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

    print(f"✅ 时序数据已保存: time_series_nolinear.csv  shape={x_final.shape}")
    print(f"✅ 因果关系已保存: causal_nolinear.csv  共 {len(causal_links)} 条边（含 {p} 条自回归）")
    print(f"   均值={x_final.mean():.4f}, 标准差={x_final.std():.4f}")

    chunk_size = 200
    chunk_vars = [x_final[i:i + chunk_size].var(axis=0).mean()
                  for i in range(0, len(x_final) - chunk_size, chunk_size)]
    print(f"   分段方差(平稳性验证): {[f'{v:.3f}' for v in chunk_vars]}")


generate_nonlinear_data()