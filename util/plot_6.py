import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from matplotlib.gridspec import GridSpec
from util import get_latest_run_id
from pathlib import Path

rcParams['font.family'] = 'SimHei'
rcParams['axes.unicode_minus'] = False

# 标准化名称函数（确保匹配准确）
def normalize_name(name):
    """标准化名称，去除空格、转小写，确保匹配"""
    return str(name).strip().lower()


def plot_multi_causal_matrix(true_csv_path, pred_csv_paths, pred_titles, ts_data_path,
                              true_title="Ground Truth", save_path=None, show_weight_gradient=False):
    """
    绘制多模型因果矩阵对比图。
    左侧大图为真实因果矩阵，右侧2行3列为6个模型的预测矩阵。
    
    参数:
        true_csv_path:       真实因果矩阵CSV路径
        pred_csv_paths:      预测因果矩阵CSV路径列表（最多6个）
        pred_titles:         每个预测矩阵的标题列表（与pred_csv_paths一一对应）
        ts_data_path:        时间序列数据CSV路径（用于读取序列名称）
        true_title:          真实矩阵的标题
        save_path:           保存路径，None则不保存
        show_weight_gradient:
            True  -> 按权重大小渐变显示
            False -> 二值化显示（有因果关系统一深蓝色）
    """
    # -------------------------- 读取序列名称 --------------------------
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

    # -------------------------- 加载因果矩阵 --------------------------
    def load_gc_matrix(csv_path, has_weights=True):
        try:
            df = pd.read_csv(csv_path, header=None, dtype={0: str, 1: str})
        except pd.errors.EmptyDataError:
            return np.zeros((series_num, series_num), dtype=int), np.zeros((series_num, series_num))

        binary_matrix = np.zeros((series_num, series_num), dtype=int)
        weight_matrix = np.zeros((series_num, series_num))

        for _, row in df.iterrows():
            if len(row) < 2:
                continue
            cause_str = normalize_name(str(row.iloc[0]))
            effect_str = normalize_name(str(row.iloc[1]))
            try:
                cause = name_to_idx.get(cause_str, int(float(cause_str)) if cause_str.replace('.','',1).isdigit() else -1)
                effect = name_to_idx.get(effect_str, int(float(effect_str)) if effect_str.replace('.','',1).isdigit() else -1)
            except (ValueError, KeyError):
                continue
            if 0 <= cause < series_num and 0 <= effect < series_num:
                binary_matrix[effect, cause] = 1
                if has_weights and row.shape[0] >= 3:
                    weight_matrix[effect, cause] = float(row.iloc[2])

        return binary_matrix, weight_matrix

    # 加载真实矩阵
    GC_true, _ = load_gc_matrix(true_csv_path, has_weights=False)

    # 加载所有预测矩阵
    pred_data = []
    for csv_path in pred_csv_paths:
        binary, weights = load_gc_matrix(csv_path, has_weights=True)
        pred_data.append((binary, weights))

    # -------------------------- 绘图 --------------------------
    # 左侧真实矩阵占1列，右侧3列给预测矩阵，用 width_ratios 控制比例
    fig = plt.figure(figsize=(16, 7))
    gs = GridSpec(2, 4, figure=fig, wspace=0.25, hspace=0.35,
                  width_ratios=[1.5, 1, 1, 1],
                  height_ratios=[1, 1])

    # --- 左侧大图：真实因果矩阵（占左侧1列，跨2行） ---
    ax_true = fig.add_subplot(gs[:, 0])
    ax_true.imshow(GC_true, cmap='Blues', aspect='equal')
    ax_true.set_title(true_title, fontsize=16, fontweight='bold', pad=8)
    ax_true.set_xticks(np.arange(series_num))
    ax_true.set_yticks(np.arange(series_num))
    ax_true.set_xlabel("VAR数据集", fontsize=20, labelpad=10)
    ax_true.set_xticklabels(series_names, rotation=0, ha='center', fontsize=10)
    ax_true.set_yticklabels(series_names, fontsize=10)
    ax_true.tick_params(axis='both', which='both', length=0)

    # --- 右侧2×3：预测矩阵 ---
    for idx in range(min(len(pred_data), 6)):
        row = idx // 3      # 0 或 1
        col = idx % 3       # 0, 1, 2
        ax = fig.add_subplot(gs[row, col + 1])  # 从第2列开始

        binary, weights = pred_data[idx]
        has_w = np.any(weights > 0)
        use_grad = show_weight_gradient and has_w

        if use_grad:
            ax.imshow(weights, cmap='Blues', aspect='equal')
        else:
            ax.imshow(binary, cmap='Blues', aspect='equal')

        # 标题
        title = pred_titles[idx] if idx < len(pred_titles) else f"Model {idx+1}"
        ax.set_title(title, fontsize=14, pad=6)
        # ax.set_xticks([])
        # ax.set_yticks([])
        ax.set_xticks(np.arange(series_num))
        ax.set_yticks(np.arange(series_num))
        ax.set_xticklabels(series_names, rotation=0, ha='center', fontsize=10)
        ax.set_yticklabels(series_names, fontsize=10)
        ax.tick_params(axis='both', which='both', length=0)

        # 红色边框标注与真实矩阵不一致的位置
        for i in range(series_num):
            for j in range(series_num):
                # 渐变模式下显示权重数值
                if use_grad and binary[i, j] == 1 and weights[i, j] > 0:
                    weight_val = weights[i, j]
                    max_val = weights.max() if weights.max() > 0 else 1.0
                    text_color = 'white' if weight_val > (max_val * 0.5) else 'black'
                    ax.text(j, i, f"{weight_val:.2f}", ha="center", va="center",
                            color=text_color, fontsize=6, weight='bold')

                if GC_true[i, j] != binary[i, j]:
                    rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor='none',
                                         edgecolor='red', linewidth=2)
                    ax.add_patch(rect)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n多模型对比图已保存至：{save_path}")

    plt.show()
    plt.close()


# ======================== 原有的双图对比函数保留 ========================
def plot_causal_matrix_with_auto_names(true_csv_path, pred_csv_path, ts_data_path,
                                       pred_title="模型预测", true_title="真实格兰杰因果矩阵",
                                       save_path=None, show_weight_gradient=True):
    """单模型对比（保留原有功能，内部调用多模型函数）"""
    plot_multi_causal_matrix(
        true_csv_path=true_csv_path,
        pred_csv_paths=[pred_csv_path],
        pred_titles=[pred_title],
        ts_data_path=ts_data_path,
        true_title=true_title,
        save_path=save_path,
        show_weight_gradient=show_weight_gradient
    )


# -------------------------- 调用示例 --------------------------
if __name__ == "__main__":

    # ============ 路径配置 ============
    TRUE_CSV = "../data/virtual/causal_nolinear.csv" # 真实因果矩阵的路径
    TS_DATA  = "../data/virtual/time_series_nolinear.csv" # 时间序列数据CSV路径（用于读取序列名称）
    SAVE_PATH = "multi_nolinear_compare.png" # 保存路径

    # 6个模型的预测结果CSV路径
    PRED_CSVS = [
        'compare_model_matrix/nolinear/TS-GC.csv',
        'compare_model_matrix/nolinear/CausalFormer.csv',
        'compare_model_matrix/nolinear/TCDF.csv',
        'compare_model_matrix/nolinear/cMLP.csv',
        'compare_model_matrix/nolinear/cLSTM.csv',
        'compare_model_matrix/nolinear/GVAR.csv',
    ]

    # 每个模型的标题
    PRED_TITLES = [
        'TS-GC',
        'CausalFormer',
        'TCDF',
        'cMLP',
        'cLSTM',
        'GVAR',
    ]

    # ============ 显示配置 ============
    # True  -> 按权重渐变显示
    # False -> 二值化显示（有因果关系统一深蓝色）
    SHOW_WEIGHT_GRADIENT = False

    # ============ 绘图 ============
    plot_multi_causal_matrix(
        true_csv_path=TRUE_CSV,
        pred_csv_paths=PRED_CSVS,
        pred_titles=PRED_TITLES,
        ts_data_path=TS_DATA,
        true_title="真实因果矩阵",
        save_path=SAVE_PATH,
        show_weight_gradient=SHOW_WEIGHT_GRADIENT
    )