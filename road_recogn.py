import pandas as pd
import numpy as np
import torch
from scipy.stats import linregress # 用于线性回归计算斜率

# (你现有的其他imports...)
# from config import *
# from model.TS_GC import MutiTS_GC
# from visual.plot_causal_link import save_causal_links
# ... (rcParams等设置)

# 假设你的 X_DATA, Y_DATA, SERIES_NUM, INPUT_WINDOW, OUTPUT_WINDOW 已经从 config.py 加载
# 为了独立运行此示例，我们先用占位符，实际使用时请确保这些变量已定义
# X_DATA = None # torch.Tensor, e.g., shape [num_total_samples, input_window, series_num]
# Y_DATA = None # torch.Tensor, e.g., shape [num_total_samples, output_window, series_num]
# INPUT_WINDOW = 5 # Example value
# OUTPUT_WINDOW = 1 # Example value
# SERIES_NUM = None # Example value, e.g., 5
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_causal_graph_df(csv_path):
    """从CSV加载因果图为DataFrame"""
    try:
        df = pd.read_csv(csv_path, header=None, names=['source', 'target', 'strength'])
        return df
    except FileNotFoundError:
        print(f"错误：因果图文件 {csv_path} 未找到。")
        return None
    except pd.errors.EmptyDataError:
        print(f"错误：因果图文件 {csv_path} 为空。")
        return None

def get_effect_variables(causal_df, cause_var_idx):
    """获取指定原因变量的所有直接效应变量及其强度"""
    if causal_df is None:
        return []
    effects = causal_df[causal_df['source'] == cause_var_idx]
    return list(zip(effects['target'].astype(int), effects['strength']))

def predict_future_values_for_series(model, X_data_full, fault_start_sample_idx,
                                     series_idx_to_predict, num_predictions_n_f,
                                     input_window, device):
    """
    为特定序列预测未来 n_f 个值。
    fault_start_sample_idx: 是 X_data_full 中的索引，表示故障开始影响的那个Y对应的X的起始索引。
                           即 Y_data_full[fault_start_sample_idx] 是第一个受影响的真实值。
    """
    model.eval()
    predictions = []
    current_X_data_idx = fault_start_sample_idx

    for _ in range(num_predictions_n_f):
        # 确保我们有足够的历史数据来构建输入窗口
        # X_input 的最后一个时间点应该是 current_X_data_idx + input_window -1
        # 我们要预测的是 Y_data[current_X_data_idx] 之后的值
        # 所以，预测 Y[t_f+1] 时，使用的 X 是 X_data[t_f] (即输入窗口结束于 t_f)
        # 预测 Y[t_f+k] 时，使用的 X 是 X_data[t_f+k-1]

        # X_input 对应于 X_data[current_X_data_idx]
        if current_X_data_idx >= X_data_full.shape[0]:
            print(f"警告: 预测时索引 {current_X_data_idx} 超出X_data范围 ({X_data_full.shape[0]})。可能预测不完整。")
            break

        # 输入给模型的数据是 X_data_full[current_X_data_idx]
        # 这将预测 Y_data_full[current_X_data_idx]
        # 我们需要的是 Y_data_full[fault_start_sample_idx + 1] 到 Y_data_full[fault_start_sample_idx + n_f]

        # 为了预测 fault_start_sample_idx + k (k从1到n_f) 的值
        # 我们需要使用 X_data_full[fault_start_sample_idx + k -1] 作为输入
        idx_for_input = fault_start_sample_idx + len(predictions) # 这是Y的索引，对应X的索引是 Y_idx -1 +1 = Y_idx
                                                              # 即 X_data[idx_for_input] 用于预测 Y_data[idx_for_input]
        
        if idx_for_input >= X_data_full.shape[0]:
             print(f"警告: 预测时构造输入数据索引 {idx_for_input} 超出X_data范围。")
             break

        input_slice = X_data_full[idx_for_input, :, :].unsqueeze(0).to(device) # [1, input_window, series_num]

        with torch.no_grad():
            model_pred = model(input_slice) # [1, output_window, series_num]
        
        # 假设 output_window = 1, 或者我们只关心第一个预测步
        predicted_value = model_pred[0, 0, series_idx_to_predict].item()
        predictions.append(predicted_value)
        
        # current_X_data_idx +=1 # 移动到下一个时间点进行预测
                               # (这里假设模型不依赖于前一个预测值作为输入，而是用真实的X序列)
                               # 论文中并未明确指出预测是自回归式的，而是基于CF-LSTM
                               # 我们的MutiTS_GC模型是基于输入窗口X来预测Y的。

    return np.array(predictions)


def calculate_disturbance_rate(historical_values, predicted_values):
    """
    根据公式 (33) 计算扰动率 d_j。
    historical_values: 故障时刻及之前的 n_f+1 个实际值 (长度 n_f+1)
    predicted_values: 故障时刻之后的 n_f 个预测值 (长度 n_f)
    """
    if len(historical_values) == 0 and len(predicted_values) == 0:
        return 0.0
    
    # 组合历史值和预测值，形成论文中的 xj_k 序列，长度为 2*n_f + 1
    # 论文中 tk = 1, 2, ..., 2nf+1
    # xj_k = aj*tk + bj
    # 历史值对应 t_f-n_f, ..., t_f
    # 预测值对应 t_f+1, ..., t_f+n_f
    # 我们有 n_f+1 个历史值 (Y[t_f-n_f]...Y[t_f]) 和 n_f 个预测值 (Y_hat[t_f+1]...Y_hat[t_f+n_f])
    
    combined_values = np.concatenate((historical_values, predicted_values))

    if len(combined_values) < 2 : # 需要至少两个点来拟合一条线
        return 0.0

    # 标准化 combined_values
    mean_val = np.mean(combined_values)
    std_val = np.std(combined_values)

    if std_val == 0: # 如果标准差为0，说明所有值都一样，斜率为0
        return 0.0

    standardized_values = (combined_values - mean_val) / std_val
    
    # 创建时间轴 t_k (从1到 len(standardized_values))
    time_axis = np.arange(1, len(standardized_values) + 1)
    
    # 最小二乘线性拟合
    # slope (a_j), intercept (b_j), r_value, p_value, std_err
    slope, _, _, _, _ = linregress(time_axis, standardized_values)
    
    return abs(slope) # d_j = |a_j|

def determine_fault_propagation_path(model, causal_graph_df, root_cause_variable_idx,
                                     X_data_full, Y_data_full, fault_start_sample_idx,
                                     series_num, input_window, output_window,
                                     n_f=10, mu=1.0, device=torch.device("cpu"), max_path_length=None):
    """
    根据论文逻辑确定故障传播路径。
    model: 训练好的MutiTS_GC模型。
    causal_graph_df: DataFrame, 列应为 ['source', 'target', 'strength']。
    root_cause_variable_idx: 故障根源变量的索引。
    X_data_full: 完整的输入时间序列数据 (torch.Tensor)。
    Y_data_full: 完整的输出真实标签数据 (torch.Tensor)。
    fault_start_sample_idx: Y_data_full中的索引，表示第一个实际发生故障的数据点。
                            例如，如果Y_data[100]是第一个故障点，则此值为100。
    series_num: 总的序列数。
    input_window: 模型输入窗口大小。
    output_window: 模型输出窗口大小 (这里我们假设主要用第一个输出步)。
    n_f: 用于计算扰动率的未来预测样本数 (论文中的 n_f)。
    mu: 影响因子公式中的调整参数 (论文中的 mu)。
    device: 计算设备。
    max_path_length: 可选，最大路径长度，防止因果图中的环导致无限循环。
    """
    if causal_graph_df is None or causal_graph_df.empty:
        print("因果图为空，无法确定传播路径。")
        return [root_cause_variable_idx]

    if max_path_length is None:
        max_path_length = series_num # 默认最大长度为序列总数

    current_var_idx = root_cause_variable_idx
    propagation_path = [current_var_idx]
    
    # 确保 Y_data_full 是 numpy array 以便索引
    if isinstance(Y_data_full, torch.Tensor):
        Y_data_full_np = Y_data_full.cpu().numpy()
    else:
        Y_data_full_np = Y_data_full

    while len(propagation_path) < max_path_length:
        effect_variables_strengths = get_effect_variables(causal_graph_df, current_var_idx)
        
        # 过滤掉已经存在于路径中的变量，防止简单环路 (更复杂的环路仍可能需要额外处理)
        effect_variables_strengths = [(eff_idx, stren) for eff_idx, stren in effect_variables_strengths if eff_idx not in propagation_path]

        if not effect_variables_strengths:
            print(f"变量 {current_var_idx} 没有新的下游效应变量，路径终止。")
            break

        if len(effect_variables_strengths) == 1:
            next_var_idx = effect_variables_strengths[0][0]
            print(f"变量 {current_var_idx} 只有一个效应变量: {next_var_idx}。")
        else:
            print(f"变量 {current_var_idx} 有多个效应变量: {[v[0] for v in effect_variables_strengths]}。计算影响因子...")
            influence_factors = []
            for effect_idx, causality_strength_c_ij in effect_variables_strengths:
                # 1. 准备用于计算扰动率的数据窗口
                # 历史真实值: Y[t_f-n_f] 到 Y[t_f] (共 n_f + 1 个点)
                # Y_data_full_np 的形状是 [samples, output_window, series_num]
                # 我们取 output_window 的第一个维度 (索引0)
                hist_start_idx = fault_start_sample_idx - n_f
                hist_end_idx = fault_start_sample_idx + 1 # 不包含此索引，所以实际是到 fault_start_sample_idx
                
                if hist_start_idx < 0:
                    print(f"警告: 效应变量{effect_idx}的历史数据窗口起始点({hist_start_idx})为负。可能数据不足。")
                    historical_data_points = Y_data_full_np[0:hist_end_idx, 0, effect_idx]
                else:
                    historical_data_points = Y_data_full_np[hist_start_idx:hist_end_idx, 0, effect_idx]

                if len(historical_data_points) == 0 and n_f > 0: # 如果没有历史数据点，但需要预测
                     print(f"警告: 效应变量{effect_idx}的历史数据点为空。扰动率可能不准确。")
                
                # 2. 预测未来值: Y_hat[t_f+1] 到 Y_hat[t_f+n_f] (共 n_f 个点)
                #    预测从 Y_data_full[fault_start_sample_idx+1] 开始
                predicted_data_points = predict_future_values_for_series(
                    model, X_data_full,
                    fault_start_sample_idx, # 传递的是Y中第一个故障点的索引
                    effect_idx, num_predictions_n_f=n_f,
                    input_window=input_window, device=device
                )
                if len(predicted_data_points) < n_f:
                    print(f"警告: 效应变量{effect_idx}的预测点数 ({len(predicted_data_points)}) 少于 n_f ({n_f})。")


                # 3. 计算扰动率 d_j
                disturbance_rate_d_j = calculate_disturbance_rate(historical_data_points, predicted_data_points)
                print(f"  - 效应变量 {effect_idx}: 历史点数={len(historical_data_points)}, 预测点数={len(predicted_data_points)}, d_{effect_idx} = {disturbance_rate_d_j:.4f}, c_{current_var_idx}->{effect_idx} = {causality_strength_c_ij:.4f}")

                # 4. 计算影响因子 gamma_i_j
                gamma_i_j = mu * disturbance_rate_d_j * causality_strength_c_ij
                influence_factors.append({'effect_idx': effect_idx, 'gamma': gamma_i_j, 'd_j': disturbance_rate_d_j, 'c_ij': causality_strength_c_ij})
            
            if not influence_factors:
                print(f"变量 {current_var_idx} 的所有下游效应变量无法计算影响因子，路径终止。")
                break

            # 选择gamma最大的作为下一个节点
            best_next_step = max(influence_factors, key=lambda x: x['gamma'])
            next_var_idx = best_next_step['effect_idx']
            print(f"  选择的下一节点: {next_var_idx} (Gamma={best_next_step['gamma']:.4f}, d_j={best_next_step['d_j']:.4f}, c_ij={best_next_step['c_ij']:.4f})")

        propagation_path.append(next_var_idx)
        current_var_idx = next_var_idx
        
        if len(propagation_path) == max_path_length :
             print(f"达到最大路径长度 {max_path_length}，路径终止。")


    return propagation_path