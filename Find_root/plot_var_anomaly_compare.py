import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.ticker import ScalarFormatter
import pickle
import os

# 导入你的自定义类
from RSPCA import WaveletDenoiser, RobustScaler, RNSPCA
from pca_config import Config

# 设置支持中文显示与全局字体放大
rcParams['font.family'] = 'SimHei' 
rcParams['axes.unicode_minus'] = False
rcParams['axes.titlesize'] = 37      # 子图标题字体放大
rcParams['axes.labelsize'] = 37      # 坐标轴标签字体放大
rcParams['xtick.labelsize'] = 37     # X轴刻度字体放大
rcParams['ytick.labelsize'] = 37     # Y轴刻度字体放大
# 图例字体稍微调小一点点（14），防止在子图里遮挡曲线
     

def plot_comparisons_with_loaded_model(normal_csv_path, abnormal_csv_path, model_pkl_path, output_filename="Top10_Variables_Comparison.pdf", custom_labels=None):
    """
    绘制变量的正常、异常和异常分数对比图
    
    参数:
        normal_csv_path: 正常数据CSV路径
        abnormal_csv_path: 异常数据CSV路径
        model_pkl_path: 训练好的模型PKL路径
        output_filename: 输出图片文件名
        custom_labels: 自定义y轴标签名称列表，长度应为10。如果为None则使用原始变量名
    """
    # ==========================================
    # 1. 加载数据
    # ==========================================
    print(f">>> 正在读取数据...\n正常数据: {normal_csv_path}\n异常数据: {abnormal_csv_path}")
    df_normal = pd.read_csv(normal_csv_path)
    df_abnormal = pd.read_csv(abnormal_csv_path)
    feature_names = df_normal.columns.tolist()

    # ==========================================
    # 2. 加载已训练的模型
    # ==========================================
    if not os.path.exists(model_pkl_path):
        raise FileNotFoundError(f"找不到模型文件: {model_pkl_path}。请先运行 train_PAC.py 进行训练。")
        
    print(f">>> 正在加载预训练模型: {model_pkl_path} ...")
    with open(model_pkl_path, 'rb') as f:
        pipeline = pickle.load(f)
        
    model = pipeline['model']
    denoiser = pipeline.get('denoiser') 
    scaler = pipeline['scaler']

    # ==========================================
    # 3. 诊断与分数计算 (使用异常数据)
    # ==========================================
    print(">>> 正在处理异常数据，计算异常分数与贡献度...")
    if denoiser is not None:
        X_abn_prep = denoiser.transform(df_abnormal)
    else:
        X_abn_prep = df_abnormal.values
        
    X_abn_scaled = scaler.transform(X_abn_prep)

    # 计算贡献度 (获取 Top 10)
    contrib_mode = getattr(Config, 'CONTRIB_MODE', 'spe')
    if contrib_mode == 'combined':
        diag_res = model.trigger_diagnose(
            X_abn_scaled, mode='combined', 
            X_raw_fault=df_abnormal.values, 
            X_raw_normal_median=scaler.median
        )
    else:
        diag_res = model.trigger_diagnose(X_abn_scaled, mode='spe')

    dcc_scores = diag_res['dcc_norm']
    top_k = 10
    top_indices = np.argsort(dcc_scores)[::-1][:top_k]

    # 获取异常数据中每个变量的 SPE 时间序列分数
    var_spe_series = model.get_variable_spe_series(X_abn_scaled)

    # ==========================================
    # 4. 绘制 10行 x 3列 图像
    # ==========================================
    print(">>> 正在绘制 10x3 图像矩阵...")
    fig, axes = plt.subplots(nrows=10, ncols=3, figsize=(24, 32))

    line_color = '#1f77b4'

    for row_idx, feat_idx in enumerate(top_indices):
        # 使用自定义标签或原始变量名
        if custom_labels is not None and row_idx < len(custom_labels):
            display_name = custom_labels[row_idx]
        else:
            display_name = feature_names[feat_idx]
        
        norm_vals = df_normal.iloc[:, feat_idx].values
        abn_vals = df_abnormal.iloc[:, feat_idx].values
        
        norm_min, norm_max = np.min(norm_vals), np.max(norm_vals)
        abn_min, abn_max = np.min(abn_vals), np.max(abn_vals)
        
        norm_range = norm_max - norm_min if norm_max > norm_min else 1.0
        abn_range = abn_max - abn_min if abn_max > abn_min else 1.0

        # -----------------------------
        # 第 1 列: 正常时间序列数据
        # -----------------------------
        ax_norm = axes[row_idx, 0]
        ax_norm.plot(norm_vals, color=line_color, linewidth=2.0)
        
        ax_norm.set_ylim(norm_min - norm_range * 2.5, norm_max + norm_range * 2.5)
        # 大数值时自动使用科学计数法 (×10^n)
        ax_norm.ticklabel_format(style='sci', axis='y', scilimits=(-3, 3), useMathText=True)
        ax_norm.yaxis.get_offset_text().set_fontsize(32)
        
        if row_idx == 0:
            ax_norm.set_title("正常时间序列数据", fontweight='bold', pad=20)
        
        # 缩小 labelpad，因为稍后会强制统一对齐
        ax_norm.set_ylabel(f"{display_name}", fontweight='bold', labelpad=10)
        ax_norm.grid(True, linestyle=':', alpha=0.6)

        # -----------------------------
        # 第 2 列: 异常时间序列数据
        # -----------------------------
        ax_abn = axes[row_idx, 1]
        ax_abn.plot(abn_vals, color=line_color, linewidth=2.0)
        
        ax_abn.set_ylim(abn_min - abn_range * 0.1, abn_max + abn_range * 0.1)
        # 大数值时自动使用科学计数法 (×10^n)
        ax_abn.ticklabel_format(style='sci', axis='y', scilimits=(-3, 3), useMathText=True)
        ax_abn.yaxis.get_offset_text().set_fontsize(32)
        
        if row_idx == 0:
            ax_abn.set_title("异常时间序列数据", fontweight='bold', pad=20)
        ax_abn.grid(True, linestyle=':', alpha=0.6)

        # -----------------------------
        # 第 3 列: 变量 SPE 异常分数
        # -----------------------------
        ax_spe = axes[row_idx, 2]
        spe_scores = var_spe_series[:, feat_idx]
        ax_spe.plot(spe_scores, color=line_color, linewidth=2.0)
        
        if row_idx == 0:
            ax_spe.set_title("变量异常分数", fontweight='bold', pad=20)

        # 绘制阈值红线
        var_thresh = model.var_SPE_thresholds[feat_idx]
        ax_spe.axhline(var_thresh, color='red', linestyle='--', linewidth=2.0)
        
        is_anomaly = np.nan_to_num(spe_scores) > var_thresh
        diff = np.diff(is_anomaly.astype(int))
        change_points = np.where(diff != 0)[0] + 1
        split_indices = [0] + list(change_points) + [len(spe_scores) - 1]

        for j in range(len(split_indices) - 1):
            start = split_indices[j]
            end = split_indices[j+1]
            if is_anomaly[start]:
                ax_spe.axvspan(start, end, color='lightcoral', alpha=0.3)
            else:
                ax_spe.axvspan(start, end, color='lightskyblue', alpha=0.1)

        ax_spe.grid(True, linestyle=':', alpha=0.6)

    # ★ 关键修改：强制统一对齐最左侧（第0列）的所有 Y 轴标签
    fig.align_ylabels(axes[:, 0])

    plt.tight_layout(pad=2.0)
    plt.savefig(output_filename, dpi=200, bbox_inches='tight')
    plt.close()
    print(f">>> 绘制完成！对比图已成功保存为: {output_filename}")

if __name__ == "__main__":
    NORMAL_CSV = Config.TRAIN_DATA_PATH
    ABNORMAL_CSV = Config.TEST_DATA_PATH
    MODEL_PKL = Config.MODEL_SAVE_PATH
    OUTPUT_IMAGE = "Top10_Variables_10x3_Comparison.pdf"
    
    # 方式1: 使用原始变量名(默认)
    # plot_comparisons_with_loaded_model(NORMAL_CSV, ABNORMAL_CSV, MODEL_PKL, OUTPUT_IMAGE)
    
    # 方式2: 使用自定义变量名
    # specified_vars = ['PDRSA1104.PV', 'TIC1155.MV', 'FRCA1110.PV', 'LRCA1101.PV', 'PRA1102.PV', 
    #                   'WR1104.PV', 'DR1107.PV', 'PDR1103.PV', 'TIC1155.PV', 'CCSFIC1401.PV']
    specified_vars = ['DR1107.PV', 'WR1104.PV', 'TIC1155.MV','CCSFIC1401.PV','TIC1155.PV',
                      'LRCA1101.PV','FRCA1110.PV', 'PRA1102.PV', 'PDR1103.PV','FRCA1111.PV']
    plot_comparisons_with_loaded_model(NORMAL_CSV, ABNORMAL_CSV, MODEL_PKL, OUTPUT_IMAGE, custom_labels=specified_vars)