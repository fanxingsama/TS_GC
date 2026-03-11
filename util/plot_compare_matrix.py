import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from util import get_latest_run_id
from pathlib import Path

rcParams['font.family'] = 'SimHei'
rcParams['axes.unicode_minus'] = False

# 标准化名称函数（确保匹配准确）
def normalize_name(name):
    """标准化名称，去除空格、转小写，确保匹配"""
    return str(name).strip().lower()

def plot_causal_matrix_with_auto_names(true_csv_path, pred_csv_path, ts_data_path, 
                                       pred_title="模型预测", true_title="真实格兰杰因果矩阵", 
                                       save_path=None, show_weight_gradient=True):
    """
    自动读取时间序列名称，绘制因果矩阵对比图并标注错误位置。
    
    参数:
        show_weight_gradient: 
            True  -> 按权重大小渐变显示（权重大颜色深，权重小颜色浅）
            False -> 只要有因果关系就统一显示为深蓝色（二值化显示）
    """
    # -------------------------- 核心：自动读取序列名称 --------------------------
    try:
        ts_header = pd.read_csv(ts_data_path, nrows=0).columns.tolist()
        series_names = [name for name in ts_header if str(name).strip() != ""]
        if not series_names:
            raise ValueError("时间序列文件第一行未检测到有效序列名称！")
        series_num = len(series_names)
        print(f"自动识别到 {series_num} 个序列名称：{series_names}")
    except Exception as e:
        print(f"读取时间序列名称失败：{e}")
        return
    
    name_to_idx = {normalize_name(name): idx for idx, name in enumerate(series_names)}
    
    # -------------------------- 加载因果矩阵（与 test_TS_GC 一致的逻辑） --------------------------
    def load_gc_matrix(csv_path, has_weights=True):
        """加载GC矩阵，支持名称解析，返回二值矩阵和权重矩阵"""
        try:
            df = pd.read_csv(csv_path, header=None, dtype={0: str, 1: str})
        except pd.errors.EmptyDataError:
            print(f"警告: 文件 {csv_path} 为空，返回零矩阵。")
            return np.zeros((series_num, series_num), dtype=int), np.zeros((series_num, series_num))

        binary_matrix = np.zeros((series_num, series_num), dtype=int)
        weight_matrix = np.zeros((series_num, series_num))
        
        for _, row in df.iterrows():
            if len(row) < 2:
                continue
                
            cause_str = normalize_name(str(row.iloc[0]))
            effect_str = normalize_name(str(row.iloc[1]))
            
            try:
                if cause_str in name_to_idx:
                    cause = name_to_idx[cause_str]
                else:
                    cause = int(float(cause_str))
                
                if effect_str in name_to_idx:
                    effect = name_to_idx[effect_str]
                else:
                    effect = int(float(effect_str))
            except (ValueError, KeyError):
                continue

            if 0 <= cause < series_num and 0 <= effect < series_num:
                binary_matrix[effect, cause] = 1
                if has_weights and row.shape[0] >= 3:
                    weight_matrix[effect, cause] = float(row.iloc[2])
        
        return binary_matrix, weight_matrix
    
    # 加载真实矩阵和预测（约束后）矩阵
    GC_true, _ = load_gc_matrix(true_csv_path, has_weights=False)
    GC_pred_binary, GC_pred_weights = load_gc_matrix(pred_csv_path, has_weights=True)
    
    # 根据配置决定预测矩阵的显示方式
    has_weights = np.any(GC_pred_weights > 0)
    use_gradient = show_weight_gradient and has_weights
    
    # -------------------------- 绘图（与 test_TS_GC 风格一致） --------------------------
    fig, axarr = plt.subplots(1, 2, figsize=(16, 8))
    
    # --- 1. 绘制真实格兰杰因果矩阵 ---
    axarr[0].imshow(GC_true, cmap='Blues', aspect='auto')
    # axarr[0].set_title(true_title, fontsize=18)
    # axarr[0].set_xticks(np.arange(series_num))
    axarr[0].set_xticks([])
    axarr[0].set_yticks([])
    # axarr[0].set_yticks(np.arange(series_num))
    # axarr[0].set_xticklabels(series_names, rotation=45, ha='right')
    # axarr[0].set_yticklabels(series_names)
    axarr[0].tick_params(axis='both', which='both', length=0)
    
    # --- 2. 绘制约束后预测矩阵 ---
    if use_gradient:
        # 按权重渐变显示
        axarr[1].imshow(GC_pred_weights, cmap='Blues', aspect='auto',
                        extent=(-0.5, series_num-0.5, series_num-0.5, -0.5))
    else:
        # 二值化显示：有因果关系就统一深蓝色
        axarr[1].imshow(GC_pred_binary, cmap='Blues', aspect='auto')
    
    # axarr[1].set_title(pred_title, fontsize=18)
    axarr[1].set_xticks(np.arange(series_num))
    axarr[1].set_yticks(np.arange(series_num))
    # axarr[1].set_xticks([])
    # axarr[1].set_yticks([])
    axarr[1].set_xticklabels(series_names, rotation=45, ha='right')
    axarr[1].set_yticklabels(series_names)
    axarr[1].tick_params(axis='both', which='both', length=0)
    
    # --- 在预测矩阵上添加权重文本和错误标记 ---
    for i in range(series_num):
        for j in range(series_num):
            # 添加权重文本（仅在渐变模式下显示）
            if use_gradient and GC_pred_binary[i, j] == 1 and GC_pred_weights[i, j] > 0:
                weight_val = GC_pred_weights[i, j]
                text = f"{weight_val:.3f}"
                
                max_val = GC_pred_weights.max() if GC_pred_weights.max() > 0 else 1.0
                text_color = 'white' if weight_val > (max_val * 0.5) else 'black'
                
                axarr[1].text(j, i, text, ha="center", va="center",
                             color=text_color, fontsize=8, weight='bold')
            
            # 红色边框标注错误位置
            if GC_true[i, j] != GC_pred_binary[i, j]:
                rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor='none',
                                   edgecolor='red', linewidth=3)
                axarr[1].add_patch(rect)
    
    fig.tight_layout(pad=3.0)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        mode_str = "权重渐变" if use_gradient else "二值化"
        print(f"\n绘图完成（{mode_str}模式）！图片已保存至：{save_path}")
    
    plt.show()
    plt.close()

# -------------------------- 调用示例 --------------------------
if __name__ == "__main__":
    
    run_id = get_latest_run_id(('../saved'))  # 确保函数可用
    
    TRUE_CSV = "../data/virtual/causal_linear.csv"  # 真实因果矩阵的路径
    # PRED_CSV = Path('../saved') / run_id / 'GC_matrix_constrained.csv'
    PRED_CSV = '../saved/2026-03-08_18-15-59_linear/GC_matrix_constrained.csv'  # 约束后预测矩阵的路径
    TS_DATA  = "../data/virtual/time_series_linear.csv"  # 时间序列的文件夹
    SAVE_PATH = "causal_matrix_names.png"
    
    MY_MODEL_TITLE = "预测结果"
    MY_TRUE_TITLE = "真实因果图"
    
    # ============ 配置项 ============
    # True  -> 按权重渐变显示（权重大颜色深，权重小颜色浅，格子内标注数值）
    # False -> 二值化显示（有因果关系统一深蓝色，无因果关系白色）
    SHOW_WEIGHT_GRADIENT = False
    # ================================
    
    plot_causal_matrix_with_auto_names(
        true_csv_path=TRUE_CSV, 
        pred_csv_path=PRED_CSV, 
        ts_data_path=TS_DATA,
        pred_title=MY_MODEL_TITLE,
        true_title=MY_TRUE_TITLE,
        save_path=SAVE_PATH,
        show_weight_gradient=SHOW_WEIGHT_GRADIENT
    )