import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

rcParams['font.family'] = 'SimHei'
rcParams['axes.unicode_minus'] = False

# 标准化名称函数（确保匹配准确）
def normalize_name(name):
    """标准化名称，去除空格、转小写，确保匹配"""
    return str(name).strip().lower()

def plot_causal_matrix_with_auto_names(true_csv_path, pred_csv_path, ts_data_path, 
                                       pred_title="模型预测", true_title="真实格兰杰因果矩阵\n(GC Ground Truth)", 
                                       save_path=None):
    """
    自动读取时间序列名称，绘制因果矩阵对比图并标注错误位置
    Args:
        true_csv_path: 真实因果矩阵CSV路径
        pred_csv_path: 模型预测因果矩阵CSV路径
        ts_data_path: 时间序列数据CSV路径（第一行为序列名称）
        pred_title: 预测矩阵的图表标题
        true_title: 真实矩阵的图表标题
        save_path: 图片保存路径
    """
    # -------------------------- 核心：自动读取序列名称 --------------------------
    # 读取时间序列数据的第一行作为序列名称（自动跳过空值/注释）
    try:
        # 读取第一行，不设表头，自动识别所有列名
        ts_header = pd.read_csv(ts_data_path, nrows=0).columns.tolist()
        # 过滤空名称和无效名称
        series_names = [name for name in ts_header if str(name).strip() != ""]
        if not series_names:
            raise ValueError("时间序列文件第一行未检测到有效序列名称！")
        series_num = len(series_names)
        print(f"自动识别到 {series_num} 个序列名称：{series_names}")
    except Exception as e:
        print(f"读取时间序列名称失败：{e}")
        return
    
    # 建立 名称 -> 索引 的映射字典（标准化后匹配）
    name_to_idx = {normalize_name(name): idx for idx, name in enumerate(series_names)}
    
    # -------------------------- 加载因果矩阵 --------------------------
    def load_gc_matrix(csv_path):
        """加载GC二值矩阵（兼容名称/数字索引）"""
        try:
            # 强制前两列为字符串，避免数字名称被解析为数值
            df = pd.read_csv(csv_path, header=None, dtype={0: str, 1: str})
        except pd.errors.EmptyDataError:
            print(f"警告：{csv_path} 文件为空，返回全零矩阵")
            return np.zeros((series_num, series_num), dtype=int)

        binary_matrix = np.zeros((series_num, series_num), dtype=int)
        
        for _, row in df.iterrows():
            # 至少需要两列（cause, effect）
            if len(row) < 2:
                continue
            
            cause_str = normalize_name(str(row.iloc[0]))
            effect_str = normalize_name(str(row.iloc[1]))
            
            try:
                # 1. 优先通过标准化名称匹配索引
                if cause_str in name_to_idx:
                    cause = name_to_idx[cause_str]
                else:
                    # 2. 名称匹配失败则尝试解析为数字索引
                    cause = int(float(cause_str))
                
                if effect_str in name_to_idx:
                    effect = name_to_idx[effect_str]
                else:
                    effect = int(float(effect_str))
            except (ValueError, KeyError):
                # 既不是有效名称也不是数字索引，跳过该行
                continue

            # 确保索引在有效范围内
            if 0 <= cause < series_num and 0 <= effect < series_num:
                # 调整为：纵轴(行)为原因，横轴(列)为受影响的序列
                binary_matrix[cause, effect] = 1
        
        return binary_matrix
    
    # 加载真实矩阵和预测矩阵
    GC_true = load_gc_matrix(true_csv_path)
    GC_pred = load_gc_matrix(pred_csv_path)
    
    # -------------------------- 绘图 + 错误标注 --------------------------
    fig, axarr = plt.subplots(1, 2, figsize=(16, 8))
    
    # 1. 绘制真实因果矩阵
    axarr[0].imshow(GC_true, cmap='Blues', aspect='auto')
    axarr[0].set_title(true_title, fontsize=18)
    axarr[0].set_ylabel('原因序列', fontsize=18)      # 纵轴改为原因
    axarr[0].set_xlabel('被影响序列', fontsize=18)  # 横轴改为被影响
    axarr[0].set_xticks(np.arange(series_num))
    axarr[0].set_yticks(np.arange(series_num))
    axarr[0].set_xticklabels(series_names, rotation=45, ha='right')
    axarr[0].set_yticklabels(series_names)
    
    # 2. 绘制预测矩阵 + 红色边框标注错误
    axarr[1].imshow(GC_pred, cmap='Blues', aspect='auto')
    axarr[1].set_title(pred_title, fontsize=18)          # 使用传入的参数作为标题
    axarr[1].set_ylabel('原因序列', fontsize=18)      # 纵轴改为原因
    axarr[1].set_xlabel('被影响序列', fontsize=18)  # 横轴改为被影响
    axarr[1].set_xticks(np.arange(series_num))
    axarr[1].set_yticks(np.arange(series_num))
    axarr[1].set_xticklabels(series_names, rotation=45, ha='right')
    axarr[1].set_yticklabels(series_names)
    
    # 核心：标注所有预测错误的位置
    for i in range(series_num):  # 行（现在是 Cause）
        for j in range(series_num):  # 列（现在是 Effect）
            if GC_true[i, j] != GC_pred[i, j]:
                # 绘制红色边框（无填充，不遮挡矩阵颜色）
                rect = plt.Rectangle(
                    (j - 0.5, i - 0.5), 1, 1,
                    facecolor='none',  # 透明填充
                    edgecolor='red',    # 红色边框
                    linewidth=2         # 边框粗细
                )
                axarr[1].add_patch(rect)
    
    # 调整布局并保存
    fig.tight_layout(pad=3.0)
    plt.show()  # 弹出窗口显示图片
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n绘图完成！图片已保存至：{save_path}")
    else:
        print("\n绘图完成！")

# -------------------------- 调用示例 --------------------------
if __name__ == "__main__":
    # 替换为你的实际文件路径
    TRUE_CSV = "../compare_model/fMRI/sim6_gt_processed.csv"    # 真实矩阵
    # PRED_CSV = "../saved/2026-03-02_02_15-19-34/GC_matrix_constrained.csv"  # TS-GC预测矩阵
    # PRED_CSV = "../compare_model/CausalFormer/csv/CausalFormer_timeseries6.csv"  # CausalFormer预测矩阵
    # PRED_CSV = "../compare_model/TCDF/TCDF_timeseries6_causal.csv"  # TCDF预测矩阵
    # PRED_CSV = "../compare_model/GVAR/GVAR_timeseries6.csv"  # GVAR预测矩阵
    PRED_CSV = "../compare_model/eSRU/eSRU_timeseries6.csv"  # GVAR预测矩阵
    TS_DATA  = "../compare_model/fMRI/timeseries6.csv"              # 时间序列数据
    SAVE_PATH = "causal_matrix_names.png"      # 保存路径
    
    # 自定义你想显示的标题
    MY_MODEL_TITLE = "预测结果"
    MY_TRUE_TITLE = "真实因果图"
    
    # 执行绘图，传入自定义标题
    plot_causal_matrix_with_auto_names(
        true_csv_path=TRUE_CSV, 
        pred_csv_path=PRED_CSV, 
        ts_data_path=TS_DATA,
        pred_title=MY_MODEL_TITLE,    # 指定右侧图表的标题
        true_title=MY_TRUE_TITLE,     # 指定左侧图表的标题 (可选)
        save_path=SAVE_PATH
    )