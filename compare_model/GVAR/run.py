import pandas as pd
import numpy as np
import sys
import os

# 将 GVAR 目录加入系统路径，以便导入其内部模块
sys.path.append(os.path.join(os.path.dirname(__file__), 'GVAR'))

from training import training_procedure_trgc

def run_gvar_fMRI_baseline(ts_csv_path, output_csv_path):
    # 1. 读取时间序列数据
    ts_df = pd.read_csv(ts_csv_path)
    ts_data = ts_df.values
    p = ts_data.shape[1]
    
    # 获取第一行的列名作为序列名称
    series_names = ts_df.columns.tolist()
    
    # 数据标准化 (采用 GVAR 作者在 run_grid_search.py 中的预处理方式)
    for j in range(p):
        ts_data[:, j] = (ts_data[:, j] - np.mean(ts_data[:, j])) / np.std(ts_data[:, j])

    # 2. 设置模型超参数
    # 这些参数提取自官方的 run_grid_search_fMRI 配置文件
    K = 1                       # 模型阶数 (lag)
    num_hidden_layers = 1       # 隐藏层数量
    hidden_layer_size = 50      # 隐藏层大小
    num_epochs = 1000           # 迭代次数
    batch_size = 64             # 批大小
    initial_lr = 0.0001         # 初始学习率
    seed = 42                   # 随机种子
    
    # 惩罚项权重。官方代码是通过循环跑多组来寻找最佳值的，这里设置一个居中的默认值。
    # 如果出来的矩阵太稀疏，可以调小 lmbd；如果太密集，可以调大 lmbd。
    lmbd = 1.5   
    gamma = 0.05 
    
    # Adam优化器参数
    beta_1 = 0.9
    beta_2 = 0.999

    print("开始使用 GVAR 训练 fMRI 时间序列...")
    
    # 3. 运行模型训练与因果推断
    a_hat_binary, a_hat, coeffs_full = training_procedure_trgc(
        data=ts_data, 
        order=K, 
        hidden_layer_size=hidden_layer_size,
        end_epoch=num_epochs, 
        lmbd=lmbd, 
        gamma=gamma, 
        batch_size=batch_size,
        seed=seed, 
        num_hidden_layers=num_hidden_layers,
        initial_learning_rate=initial_lr, 
        beta_1=beta_1, 
        beta_2=beta_2,
        verbose=True,
        signed=False
    )

    # 4. 提取预测出的二值化因果关系
    pred_causes, pred_effects = np.where(a_hat_binary == 1)
    
    # 映射回原始的序列名称，生成两列的 DataFrame 
    cause_names = [series_names[i] for i in pred_causes]
    effect_names = [series_names[i] for i in pred_effects]
    
    pred_df = pd.DataFrame({
        'cause': cause_names,
        'effect': effect_names
    })
    
    # 无表头保存为 CSV，完美适配你之前写的 plot_compare_matx.py 的读取逻辑
    pred_df.to_csv(output_csv_path, index=False, header=False)
    print(f"\n训练完成！预测的因果矩阵已保存至：{output_csv_path}")

if __name__ == "__main__":
    # 替换为你实际的路径
    TS_DATA_PATH = "../fMRI/timeseries6.csv"
    OUTPUT_PRED_PATH = "GVAR_timeseries6.csv"
    
    run_gvar_fMRI_baseline(TS_DATA_PATH, OUTPUT_PRED_PATH)