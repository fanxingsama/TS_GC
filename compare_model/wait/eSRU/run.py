import pandas as pd
import numpy as np
import torch
import sys
import os
import time
import traceback

# 加上 try-except 捕获所有错误，防止窗口闪退
try:
    # 将 SRU_for_GCI 目录加入系统路径，以便导入其内部模块
    # 注意：确保这个脚本和 SRU_for_GCI 文件夹在同一个目录下
    sys.path.append(os.path.join(os.path.dirname(__file__), 'SRU_for_GCI'))

    from models.esru_2LF import eSRU_2LF, train_eSRU_2LF
    from utils.utilFuncs import env_config

    def run_esru_fMRI_baseline(ts_csv_path, output_csv_path):
        # 1. 设备配置 (强制使用 GPU 以加速训练)
        device, seed = env_config(True, "cuda:0")
        print(f"正在使用的计算设备: {device}")

        # 2. 读取和预处理时间序列数据
        if not os.path.exists(ts_csv_path):
            raise FileNotFoundError(f"找不到数据文件，请检查路径是否正确: {ts_csv_path}")
            
        ts_df = pd.read_csv(ts_csv_path)
        series_names = ts_df.columns.tolist()
        
        # 将形状转换为 (n_nodes, T) 以符合模型输入要求
        ts_data = ts_df.values.T 
        n, T = ts_data.shape
        
        # ------------------ 修改点 1：数据增强标准化 ------------------
        # 转换为 PyTorch 张量
        Xtrain = torch.from_numpy(ts_data).float().to(device)
        # Z-score 标准化：不仅减去均值，还要除以标准差，确保特征尺度一致
        Xtrain = Xtrain - Xtrain.mean(dim=1, keepdim=True)
        Xtrain = Xtrain / (Xtrain.std(dim=1, keepdim=True) + 1e-8)

        # 3. 配置模型和训练超参数
        model_name = 'eSRU_2LF'
        A = [0.0, 0.01, 0.1, 0.99]
        dim_iid_stats = 10 
        dim_rec_stats = 10 
        dim_final_stats = 10 
        dim_rec_stats_feedback = 10 
        batchSize = 10 
        blk_size = int(batchSize / 2) 
        numBatches = int(T / batchSize)
        
        max_iter = 2000     # 最大迭代次数
        
        # ------------------ 修改点 2：大幅调小正则化参数 ------------------
        lambda1 = 0.00215   # 原 0.021544 
        lambda2 = 0.00316   # 原 0.031623
        lambda3 = 0.04641   # 原 0.464159 
        
        lr = 0.001
        lr_gamma = 0.99
        lr_update_gap = 4
        staggerTrainWin = 1 
        stoppingThresh = 1e-5
        trainVerboseLvl = 0 # 设为 0 减少刷屏，只看最终结果
        
        Gest = torch.zeros(n, n, requires_grad=False)
        
        print(f"\n开始使用 {model_name} 训练时间序列 (共 {n} 个节点，长度 {T})...")
        
        # 4. 逐个节点(Target)进行训练
        start_time = time.time()
        for predictedNode in range(n):
            print(f"-> 正在拟合目标节点: {series_names[predictedNode]} ({predictedNode+1}/{n})")
            
            # 初始化模型
            model = eSRU_2LF(n, 1, dim_iid_stats, dim_rec_stats, 
                             dim_rec_stats_feedback, dim_final_stats, A, device)
            model.to(device)
            
            # 训练模型
            model, lossVec = train_eSRU_2LF(
                model, Xtrain, device, numBatches, batchSize, blk_size, 
                predictedNode, max_iter, lambda1, lambda2, lambda3, lr, 
                lr_gamma, lr_update_gap, staggerTrainWin, stoppingThresh, trainVerboseLvl
            )
            
            # 提取当前目标的输入权重 (L2范数)
            Gest.data[predictedNode, :] = torch.norm(model.lin_xr2phi.weight.data[:, :n], p=2, dim=0)

        print(f"\n所有节点训练完毕！总耗时: {time.time() - start_time:.2f} 秒")

        # ------------------ 修改点 3：增加监控打印 ------------------
        Gest_np = Gest.cpu().numpy()
        max_weight = Gest_np.max()
        print(f"推断出的权重矩阵最大值: {max_weight:.4f}")
        
        # eSRU 稀疏化后非因果边会绝对等于 0
        pred_effects, pred_causes = np.where(Gest_np > 0)
        
        non_zero_edges = len(pred_causes)
        print(f"当前推断出的因果边数量为: {non_zero_edges}")
        
        if non_zero_edges == 0:
            print("⚠️ 警告：所有的边依然被正则化裁剪为 0！请尝试将 lambda2 和 lambda3 再缩小 10 倍。")
        
        cause_names = [series_names[i] for i in pred_causes]
        effect_names = [series_names[i] for i in pred_effects]
        
        pred_df = pd.DataFrame({
            'cause': cause_names,
            'effect': effect_names
        })
        
        # 保存结果 
        pred_df.to_csv(output_csv_path, index=False, header=False)
        print(f"预测的因果矩阵已保存至：{output_csv_path}")

    if __name__ == "__main__":
        # 替换为你实际的路径
        TS_DATA_PATH = "../fMRI/timeseries6.csv"
        OUTPUT_PRED_PATH = "eSRU_timeseries6.csv"
        
        run_esru_fMRI_baseline(TS_DATA_PATH, OUTPUT_PRED_PATH)

except Exception as e:
    print("\n" + "="*50)
    print("程序运行遇到错误崩溃了！错误信息如下：\n")
    traceback.print_exc()  # 打印详细错误追踪
    print("="*50 + "\n")

# 防止终端闪退，等待用户按下回车键才关闭
input("请按 Enter 键退出...")