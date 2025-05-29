import numpy as np
import pandas as pd
import torch


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

def create_data(data_path, input_window, output_window):
    prediction_input_df = pd.read_csv(data_path)
    all_series_cols = prediction_input_df.columns.tolist()
    prediction_input_df = prediction_input_df.iloc[:, :len(all_series_cols)]

    # 创建序列数据
    
    data_np = prediction_input_df[all_series_cols].values.astype(np.float32) 
    num_timesteps, num_series = data_np.shape

    X_list, Y_list = [], []
    for i in range(num_timesteps - input_window - output_window + 1):
        X_list.append(data_np[i : i + input_window, :])
        Y_list.append(data_np[i + input_window : i + input_window + output_window, :])
    
    X_data = torch.tensor(np.array(X_list), dtype=torch.float32)
    Y_data = torch.tensor(np.array(Y_list), dtype=torch.float32)
    return X_data, Y_data, num_series
