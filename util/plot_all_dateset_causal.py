import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from matplotlib.gridspec import GridSpec
import os

rcParams['font.family'] = 'SimHei'
rcParams['axes.unicode_minus'] = False

# 标准化名称函数（确保匹配准确）
def normalize_name(name):
    """标准化名称，去除空格、转小写，确保匹配"""
    return str(name).strip().lower()

def load_dataset_data(true_csv_path, pred_csv_paths, ts_data_path):
    """
    加载单个数据集的名称、真实矩阵和预测矩阵数据
    """
    # -------------------------- 读取序列名称 --------------------------
    try:
        ts_header = pd.read_csv(ts_data_path, nrows=0).columns.tolist()
        series_names = [name for name in ts_header if str(name).strip() != ""]
        series_names = sorted(series_names, key=lambda s: (not s.startswith('x'), int(s[1:]) if s.startswith('x') and s[1:].isdigit() else float('inf'), s))
        if not series_names:
            raise ValueError("时间序列文件第一行未检测到有效序列名称！")
        series_num = len(series_names)
        print(f"成功读取序列，共 {series_num} 个特征。")
    except Exception as e:
        print(f"读取时间序列名称失败 ({ts_data_path})：{e}")
        return [], None, []

    name_to_idx = {normalize_name(name): idx for idx, name in enumerate(series_names)}

    # -------------------------- 加载因果矩阵 --------------------------
    def load_gc_matrix(csv_path, has_weights=True):
        try:
            df = pd.read_csv(csv_path, header=None, dtype={0: str, 1: str})
        except pd.errors.EmptyDataError:
            return np.zeros((series_num, series_num), dtype=int), np.zeros((series_num, series_num))
        except FileNotFoundError:
            print(f"警告: 未找到文件 {csv_path}，将用空矩阵填充。")
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

    return series_names, GC_true, pred_data


def plot_combined_causal_matrices(datasets_config, pred_titles, save_path=None, show_weight_gradient=False,
                                   true_title_fontsize=13, pred_title_fontsize=14,
                                   true_ticklabel_fontsize=10, pred_ticklabel_fontsize=8,
                                   dataset_label_fontsize=16,
                                   wspace=0.4, hspace=0.6):
    """
    将多个数据集的对比图绘制在同一张总图上，并在每个数据集之间添加较细的虚线分隔线。
    """
    num_datasets = len(datasets_config)
    
    # 画布高度根据数据集数量动态调整 (每个数据集约占 7 的高度)
    fig = plt.figure(figsize=(15, 7 * num_datasets))
    
    # 创建网格：总行数为 数据集数量*2，列数为 4 (1个真实矩阵 + 3个预测矩阵)
    gs = GridSpec(num_datasets * 2, 4, figure=fig, wspace=wspace, hspace=hspace,
                  width_ratios=[1.2, 1, 1, 1],
                  height_ratios=[1] * (num_datasets * 2))

    # 用于保存每个数据集的左侧大图(真实矩阵)所在的Axes，以便后续计算分隔线位置
    axes_true_list = []

    for ds_idx, config in enumerate(datasets_config):
        print(f"正在处理数据集: {config['label']}...")
        series_names, GC_true, pred_data = load_dataset_data(
            config['true_csv'], config['pred_csvs'], config['ts_data']
        )
        
        if GC_true is None:
            continue
            
        series_num = len(series_names)
        row_start = ds_idx * 2  # 当前数据集在 GridSpec 中的起始行

        # --- 左侧大图：真实因果矩阵（跨2行） ---
        ax_true = fig.add_subplot(gs[row_start:row_start+2, 0])
        axes_true_list.append(ax_true)
        ax_true.imshow(GC_true, cmap='Blues', aspect='equal')
        ax_true.set_title("真实因果图", fontsize=true_title_fontsize, fontweight='bold', pad=8)
        ax_true.set_xticks(np.arange(series_num))
        ax_true.set_yticks(np.arange(series_num))
        ax_true.set_xlabel(config['label'], fontsize=dataset_label_fontsize, labelpad=10)
        
        # 处理标签旋转 (针对特征名较长的数据集)
        rot = 45 if config.get('rotate_labels', False) else 0
        ha_val = 'right' if rot == 45 else 'center'
        
        ax_true.set_xticklabels(series_names, rotation=rot, ha=ha_val, fontsize=true_ticklabel_fontsize)
        ax_true.set_yticklabels(series_names, fontsize=true_ticklabel_fontsize)
        ax_true.tick_params(axis='both', which='both', length=0)

        # --- 右侧2×3：预测矩阵 ---
        for p_idx in range(min(len(pred_data), 6)):
            r = row_start + (p_idx // 3)  # 当前小图所在行
            c = (p_idx % 3) + 1           # 当前小图所在列 (从第2列开始，即索引1)
            ax = fig.add_subplot(gs[r, c])

            binary, weights = pred_data[p_idx]
            has_w = np.any(weights > 0)
            use_grad = show_weight_gradient and has_w

            if use_grad:
                ax.imshow(weights, cmap='Blues', aspect='equal')
            else:
                ax.imshow(binary, cmap='Blues', aspect='equal')

            # 标题
            title = pred_titles[p_idx] if p_idx < len(pred_titles) else f"Model {p_idx+1}"
            ax.set_title(title, fontsize=pred_title_fontsize, pad=6)
            ax.set_xticks(np.arange(series_num))
            ax.set_yticks(np.arange(series_num))
            ax.set_xticklabels(series_names, rotation=rot, ha=ha_val, fontsize=pred_ticklabel_fontsize)
            ax.set_yticklabels(series_names, fontsize=pred_ticklabel_fontsize)
            ax.tick_params(axis='both', which='both', length=0)

            # 红色边框标注与真实矩阵不一致的位置
            for i in range(series_num):
                for j in range(series_num):
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

    # ================= 添加数据集之间的灰色细虚线分隔线 =================
    # 先强制渲染一次画布，以便精确获取各个子图的坐标位置
    fig.canvas.draw()
    
    if len(axes_true_list) > 1:
        # 获取所有矩阵的全局左右边界，保证分隔线的长度刚好贴合整个图像的宽度
        all_x0 = min([ax.get_position().x0 for ax in fig.axes])
        all_x1 = max([ax.get_position().x1 for ax in fig.axes])
        
        # 左右两端稍微各延伸一点点（0.015个相对单位）让线头显得更舒展
        line_x0 = max(0, all_x0 - 0.015) 
        line_x1 = min(1, all_x1 + 0.015)
        
        for i in range(1, len(axes_true_list)):
            # 获取上一个数据集(大图)的底部 y 坐标
            y_above = axes_true_list[i-1].get_position().y0
            # 获取当前数据集(大图)的顶部 y 坐标
            y_below = axes_true_list[i].get_position().y1
            
            # 分隔线的 y 坐标取两个数据集垂直间隔的正中间
            y_mid = (y_above + y_below) / 2.0
            
            # 【修改处】：更改为较细的虚线，并增加一点点透明度让其柔和
            line = plt.Line2D([line_x0, line_x1], [y_mid, y_mid], 
                              transform=fig.transFigure, 
                              color='black', 
                              linestyle='--',     # 设置为虚线
                              linewidth=1.0,      # 线宽调细到 1.0
                              alpha=0.6)          # 设置透明度，使其变为深灰色，更柔和不刺眼
            fig.add_artist(line)
    # ===================================================================

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n整体多模型对比图已保存至：{save_path}")

    # plt.show()
    plt.close()


if __name__ == "__main__":

    # ============ 配置所有数据集的信息 ============
    DATASETS_CONFIG = [
        {
            "label": "VAR数据集",
            "true_csv": "./compare_model_matrix/linear/causal_linear.csv",
            "ts_data": "./compare_model_matrix/linear/time_series_linear.csv",
            "rotate_labels": False,
            "pred_csvs": [
                'compare_model_matrix/linear/TS-GC.csv',
                'compare_model_matrix/linear/CausalFormer.csv',
                'compare_model_matrix/linear/TCDF.csv',
                'compare_model_matrix/linear/cMLP.csv',
                'compare_model_matrix/linear/cLSTM.csv',
                'compare_model_matrix/linear/GVAR.csv',
            ]
        },
        {
            "label": "NVAR数据集",
            "true_csv": "./compare_model_matrix/nolinear/causal_nolinear.csv",
            "ts_data": "./compare_model_matrix/nolinear/time_series_nolinear.csv",
            "rotate_labels": False,
            "pred_csvs": [
                'compare_model_matrix/nolinear/TS-GC.csv',
                'compare_model_matrix/nolinear/CausalFormer.csv',
                'compare_model_matrix/nolinear/TCDF.csv',
                'compare_model_matrix/nolinear/cMLP.csv',
                'compare_model_matrix/nolinear/cLSTM.csv',
                'compare_model_matrix/nolinear/GVAR.csv',
            ]
        },
        {
            "label": "RRSD数据集",
            "true_csv": "./compare_model_matrix/RRP/RRP_causal_true.csv",
            "ts_data": "./compare_model_matrix/RRP/RRP_data.csv",
            "rotate_labels": False,
            "pred_csvs": [
                'compare_model_matrix/RRP/TS-GC.csv',
                'compare_model_matrix/RRP/CausalFormer.csv',
                'compare_model_matrix/RRP/TCDF.csv',
                'compare_model_matrix/RRP/cMLP.csv',
                'compare_model_matrix/RRP/cLSTM.csv',
                'compare_model_matrix/RRP/GVAR.csv',
            ]
        }
    ]

    # 每个模型的统一标题
    PRED_TITLES = [
        'TS-GC',
        'CausalFormer',
        'TCDF',
        'cMLP',
        'cLSTM',
        'GVAR',
    ]

    # ============ 显示与保存配置 ============
    SHOW_WEIGHT_GRADIENT = False
    SAVE_PATH = "all_datasets_compare.pdf"

    # ============ 字号与间距配置 ============
    TRUE_TITLE_FONTSIZE = 17       # 真实因果矩阵标题字号
    PRED_TITLE_FONTSIZE = 17       # 预测矩阵标题字号
    TRUE_TICKLABEL_FONTSIZE = 15   # 真实矩阵x/y轴标签字号
    PRED_TICKLABEL_FONTSIZE = 14    # 预测矩阵x/y轴标签字号
    DATASET_LABEL_FONTSIZE = 16    # 数据集名称字号
    WSPACE = 0.1                   # 子图水平间距 (越小越紧凑)
    HSPACE = 0.3                   # 子图垂直间距 (越小越紧凑)

    # 执行绘图
    plot_combined_causal_matrices(
        datasets_config=DATASETS_CONFIG,
        pred_titles=PRED_TITLES,
        save_path=SAVE_PATH,
        show_weight_gradient=SHOW_WEIGHT_GRADIENT,
        true_title_fontsize=TRUE_TITLE_FONTSIZE,
        pred_title_fontsize=PRED_TITLE_FONTSIZE,
        true_ticklabel_fontsize=TRUE_TICKLABEL_FONTSIZE,
        pred_ticklabel_fontsize=PRED_TICKLABEL_FONTSIZE,
        dataset_label_fontsize=DATASET_LABEL_FONTSIZE,
        wspace=WSPACE,
        hspace=HSPACE,
    )