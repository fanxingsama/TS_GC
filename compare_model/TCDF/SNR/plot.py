import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from pathlib import Path

# 设置中文字体
rcParams['font.family'] = 'SimHei'
rcParams['axes.unicode_minus'] = False

def normalize_name(name):
    """标准化名称，去除空格、转小写，确保匹配"""
    return str(name).strip().lower()

def plot_noise_robustness_matrix(true_csv_path, pred_csv_paths, pred_titles, ts_data_path,
                                 true_title="真实因果矩阵", save_path=None, show_weight_gradient=False, label="test",
                                 fig_size=(26, 4.5), wspace=0.3, 
                                 title_fontsize=16, label_fontsize=14, tick_fontsize=12):
    """
    绘制模型抗噪鲁棒性实验对比图。
    单行 6 列：第 1 列为真实矩阵，第 2-6 列为预测矩阵。所有矩阵保证视觉大小一致。
    
    新增参数:
        fig_size: 画布整体大小 (宽, 高)
        wspace: 子图之间的水平间距
        title_fontsize: 子图标题的字体大小
        label_fontsize: 坐标轴标签 (如'NVAR数据集') 的字体大小
        tick_fontsize: 刻度标签 (如'x0', 'x1') 的字体大小
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
        except FileNotFoundError:
            print(f"警告：未找到文件 {csv_path}，使用全零矩阵代替。")
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
    # 使用自由配置的画布大小
    fig, axes = plt.subplots(1, 6, figsize=fig_size)
    
    # --- 第 1 列：真实因果矩阵 ---
    ax_true = axes[0]
    ax_true.imshow(GC_true, cmap='Blues', aspect='equal')
    ax_true.set_title(true_title, fontsize=title_fontsize, fontweight='bold', pad=12)
    ax_true.set_xticks(np.arange(series_num))
    ax_true.set_yticks(np.arange(series_num))
    ax_true.set_xticklabels(series_names, fontsize=tick_fontsize)
    ax_true.set_yticklabels(series_names, fontsize=tick_fontsize)
    # ax_true.set_xlabel(label, fontsize=label_fontsize, labelpad=12)
    ax_true.tick_params(axis='both', which='both', length=0)

    # --- 第 2 到 6 列：不同信噪比下的预测矩阵 ---
    for idx in range(min(len(pred_data), 5)):
        ax = axes[idx + 1]

        binary, weights = pred_data[idx]
        has_w = np.any(weights > 0)
        use_grad = show_weight_gradient and has_w

        if use_grad:
            ax.imshow(weights, cmap='Blues', aspect='equal')
        else:
            ax.imshow(binary, cmap='Blues', aspect='equal')

        # 标题
        title = pred_titles[idx] if idx < len(pred_titles) else f"SNR Level {idx+1}"
        ax.set_title(title, fontsize=title_fontsize, pad=12)
        
        # 刻度设置（恢复了 Y 轴标签）
        ax.set_xticks(np.arange(series_num))
        ax.set_yticks(np.arange(series_num))
        ax.set_xticklabels(series_names, fontsize=tick_fontsize)
        ax.set_yticklabels(series_names, fontsize=tick_fontsize) # 重新启用 Y 轴标签
        ax.tick_params(axis='both', which='both', length=0)

        # 红色边框标注与真实矩阵不一致的位置
        for i in range(series_num):
            for j in range(series_num):
                if use_grad and binary[i, j] == 1 and weights[i, j] > 0:
                    weight_val = weights[i, j]
                    max_val = weights.max() if weights.max() > 0 else 1.0
                    text_color = 'white' if weight_val > (max_val * 0.5) else 'black'
                    ax.text(j, i, f"{weight_val:.2f}", ha="center", va="center",
                            color=text_color, fontsize=7, weight='bold')

                if GC_true[i, j] != binary[i, j]:
                    rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor='none',
                                         edgecolor='red', linewidth=2)
                    ax.add_patch(rect)

    # 自由调节子图之间的水平间距
    plt.subplots_adjust(wspace=wspace)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✅ 抗噪对比图已保存至：{save_path}")

    plt.show()
    plt.close()


# -------------------------- 调用示例 --------------------------
if __name__ == "__main__":

    TRUE_CSV = "causal_nolinear.csv" 
    TS_DATA  = "time_series_nolinear.csv" 
    SAVE_PATH = "TS_GC_noise_robustness.pdf" 
    
    PRED_CSVS = [
        'TS-GC.csv',
        'TS-GC_snr20.csv',
        # 'TS-GC_snr15.csv',
        'TS-GC_snr10.csv',
        'TS-GC_snr5.csv',
        'TS-GC_snr0.csv',
    ]
    label = "NVAR数据集"

    PRED_TITLES = [
        'TS-GC (无SNR)',
        'TS-GC (SNR=20dB)',
        # 'TS-GC (SNR=15dB)',
        'TS-GC (SNR=10dB)',
        'TS-GC (SNR=5dB)',
        'TS-GC (SNR=0dB)',
    ]

    SHOW_WEIGHT_GRADIENT = False

    # ================= 自定义排版参数 =================
    # 调整下面这些参数，可以自由控制图片的最终呈现效果
    FIG_SIZE = (24, 5)     # 画布整体大小：(宽度, 高度)。因为加回了Y标签，宽度建议稍微加大一点
    WSPACE = 0.2            # 矩阵之间的水平间距（调大可以让矩阵分得更开）
    TITLE_FONTSIZE = 16      # 顶部标题的字体大小（如 "真实因果矩阵", "TS-GC (SNR=20dB)"）
    LABEL_FONTSIZE = 15      # X轴底部标签的字体大小（如 "NVAR数据集"）
    TICK_FONTSIZE = 14       # XY轴刻度标签的字体大小（如 "x0", "x1"）
    # ==================================================

    plot_noise_robustness_matrix(
        true_csv_path=TRUE_CSV,
        pred_csv_paths=PRED_CSVS,
        pred_titles=PRED_TITLES,
        ts_data_path=TS_DATA,
        true_title="真实因果矩阵",
        save_path=SAVE_PATH,
        show_weight_gradient=SHOW_WEIGHT_GRADIENT,
        label=label,
        fig_size=FIG_SIZE,
        wspace=WSPACE,
        title_fontsize=TITLE_FONTSIZE,
        label_fontsize=LABEL_FONTSIZE,
        tick_fontsize=TICK_FONTSIZE
    )