import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import argparse
import os
import sys

# 根据你的项目结构调整导入路径
# 假设此文件位于 'model' 目录中
from TCN_granger.data_create import simulate_var
from train_causalformer import train_and_evaluate, create_sequences
from optuna_ import run_tuning

# --- 常量 ---
# 数据模拟参数
P = 5           # 序列数量
T = 1000        # 总时间点
LAG = 2         # 真实的 VAR 滞后阶数
SPARSITY = 0.4  # GC 矩阵的稀疏度
BETA_VALUE = 0.8# 系数值
SD = 0.1        # 噪声标准差
DATA_SEED = 42  # 用于可复现性的随机种子

# 数据规格
FEATURE_DIM = 1 # 假设目前是单变量序列
OUTPUT_DIM = 1  # 预测序列值本身

# 训练/调优参数 (单次运行的默认值)
DEFAULT_INPUT_WINDOW = 20
DEFAULT_OUTPUT_WINDOW = 1
DEFAULT_EPOCHS = 50
DEFAULT_BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 单次运行的默认超参数 (如果不进行调优) ---
# 这些可能是先前调优运行中找到的最佳参数
DEFAULT_PARAMS = {
    'input_window': DEFAULT_INPUT_WINDOW,
    'd_model': 64,
    'n_head': 4,
    'n_layers': 2,
    'ffn_hidden': 128,
    'dropout': 0.1,
    'tau': 1.0,
    'tcn_layers': 3,
    'tcn_channels': 32,
    'tcn_kernel_size': 3,
    'tcn_dropout': 0.1,
    'learning_rate': 0.001,
    'lambda_reg': 0.001,
    'penalty_type': 'GL', # 或 'GSGL'
    # 'alpha_gsgl': 0.5, # 如果 penalty_type 是 'GSGL'，则添加
}

def main(args):
    """主函数，运行数据准备以及训练或调优。"""

    print("="*30)
    print(" 开始实验运行 ")
    print("="*30)
    print(f"使用设备: {DEVICE}")

    # --- 1. 生成并预处理数据 ---
    print("\n[阶段 1: 数据生成和预处理]")
    X_np, _, GC_true_np = simulate_var(p=P, T=T, lag=LAG, sparsity=SPARSITY,
                                       beta_value=BETA_VALUE, sd=SD, seed=DATA_SEED)
    print(f"生成的原始数据 X 形状: {X_np.shape}")
    print(f"真实的格兰杰因果关系 (前 5x5):\n{GC_true_np[:5, :5]}")

    # 添加特征维度
    if FEATURE_DIM == 1:
        X_np = X_np[:, :, np.newaxis] # 形状: [T, P, 1]
    else:
        raise NotImplementedError("数据生成未处理 FEATURE_DIM > 1 的情况。")

    # 分割数据
    # 80% 训练+验证, 20% 测试
    X_train_val_np, X_test_np = np.split(X_np, [int(0.8 * T)])
    # 60% 训练, 20% 验证
    X_train_np, X_val_np = np.split(X_train_val_np, [int(0.75 * len(X_train_val_np))])
    print(f"训练数据形状: {X_train_np.shape}")
    print(f"验证数据形状: {X_val_np.shape}")
    print(f"测试数据形状: {X_test_np.shape}")

    # --- 2. 执行训练或调优 ---
    if args.tune:
        print("\n[阶段 2: Optuna 超参数调优]")
        # 将必要的数据传递给调优函数
        best_params = run_tuning(X_train_np, X_val_np, GC_true_np, P)

        if best_params:
            print("\n[阶段 3: 可选 - 使用最佳参数在完整的训练+验证数据上重新训练]")
            print("重新训练逻辑未在此完全实现，但 best_params 可用。")
            pass # 重新训练逻辑的占位符
        else:
            print("\n未从调优中找到最佳参数。")

    else:
        print("\n[阶段 2: 使用默认参数进行单次训练运行]")
        print("使用默认参数:")
        for k, v in DEFAULT_PARAMS.items(): print(f"  {k}: {v}")

        # 为单次运行准备数据加载器
        input_window = DEFAULT_PARAMS['input_window']
        output_window = DEFAULT_OUTPUT_WINDOW
        X_train_seq, y_train_seq = create_sequences(X_train_np, input_window, output_window)
        X_val_seq, y_val_seq = create_sequences(X_val_np, input_window, output_window)

        if X_train_seq.shape[0] == 0 or X_val_seq.shape[0] == 0:
             print("错误：数据不足以使用默认输入窗口创建序列。")
             sys.exit(1)

        X_train_tensor = torch.tensor(X_train_seq, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train_seq, dtype=torch.float32)
        X_val_tensor = torch.tensor(X_val_seq, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val_seq, dtype=torch.float32)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        train_loader = DataLoader(train_dataset, batch_size=DEFAULT_BATCH_SIZE, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=DEFAULT_BATCH_SIZE, shuffle=False)

        # 为模型设置配置
        config = {
            'data_loader': {
                'args': {
                    'time_step': input_window,
                    'output_window': output_window,
                    'series_num': P,
                    'feature_dim': FEATURE_DIM,
                    'output_dim': OUTPUT_DIM
                }
            },
            'device': DEVICE.type
        }

        # 运行训练
        val_auroc, val_mse = train_and_evaluate(
            DEFAULT_PARAMS, config, train_loader, val_loader, GC_true_np, DEVICE, DEFAULT_EPOCHS
        )
        print("\n[阶段 3: 单次运行结果]")
        print(f"最终验证集 AUROC: {val_auroc:.4f}")
        print(f"最终验证集 MSE: {val_mse:.6f}")
        # 如果需要，在此处添加测试集评估

    print("\n脚本执行完毕。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="运行 CausalFormer 训练或 Optuna 调优")
    parser.add_argument('--tune', action='store_true', help='运行 Optuna 超参数调优而不是单次训练运行。')
    args = parser.parse_args()
    main(args)
