import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

# ============ 将 GVAR 目录加入路径 ============
sys.path.insert(0, os.path.dirname(__file__))

from training import training_procedure_trgc

# ================================================================
# ============ 1. 配置区（按需修改） ============
# ================================================================
TS_DATA_PATH  = '../../util/matrix/linear/time_series_linear.csv'
GT_PATH       = '../../util/matrix/linear/causal_linear.csv'
OUTPUT_CSV    = 'GVAR_result_linear.csv'
OUTPUT_PNG    = 'GVAR_result_linear.png'

# ---------- 模型超参数 ----------
K                  = 3       # 滞后阶数，linear数据用3阶足够
NUM_HIDDEN_LAYERS  = 2       # 隐藏层数量
HIDDEN_LAYER_SIZE  = 64      # 每层神经元数
NUM_EPOCHS         = 1000    # 训练轮数
BATCH_SIZE         = 64      # 批大小
INITIAL_LR         = 1e-3    # 学习率
LMBD               = 0.5     # 稀疏惩罚权重（越大越稀疏）
GAMMA              = 0.01    # 时间平滑惩罚权重
BETA_1             = 0.9
BETA_2             = 0.999
SEED               = 42

# ================================================================
# ============ 2. 加载并预处理时间序列数据 ============
# ================================================================
print('=' * 55)
print('  GVAR - Linear 数据集因果推断')
print('=' * 55)

ts_df        = pd.read_csv(TS_DATA_PATH)
series_names = ts_df.columns.tolist()
p            = len(series_names)
ts_data      = ts_df.values.astype(np.float64)

print(f'数据已加载: {ts_data.shape[0]} 个时间步, {p} 个变量: {series_names}')

# z-score 标准化（与 GVAR 官方预处理一致）
for j in range(p):
    ts_data[:, j] = (ts_data[:, j] - np.mean(ts_data[:, j])) / (np.std(ts_data[:, j]) + 1e-8)

print(f'标准化完成，各列均值: {ts_data.mean(axis=0).round(3)}')

# ================================================================
# ============ 3. 加载真实因果矩阵 ============
# ================================================================
gt_df        = pd.read_csv(GT_PATH, header=None)
name_to_idx  = {name.strip().lower(): idx for idx, name in enumerate(series_names)}

GC_true = np.zeros((p, p), dtype=int)
for _, row in gt_df.iterrows():
    cause_str  = str(row.iloc[0]).strip().lower()
    effect_str = str(row.iloc[1]).strip().lower()
    c = name_to_idx.get(cause_str,  -1)
    e = name_to_idx.get(effect_str, -1)
    if 0 <= c < p and 0 <= e < p:
        GC_true[e, c] = 1   # GC_true[effect, cause] = 1

print(f'真实因果边数: {np.sum(GC_true)}（含对角自因果）')

# ================================================================
# ============ 4. 运行 GVAR（TRGC 稳定性选择） ============
# ================================================================
print(f'\n开始训练 GVAR ...')
print(f'  K={K}, hidden={HIDDEN_LAYER_SIZE}x{NUM_HIDDEN_LAYERS}, '
      f'epochs={NUM_EPOCHS}, lmbd={LMBD}, gamma={GAMMA}, lr={INITIAL_LR}')
print('-' * 55)

a_hat_binary, coeffs_full = training_procedure_trgc(
    data               = ts_data,
    order              = K,
    hidden_layer_size  = HIDDEN_LAYER_SIZE,
    end_epoch          = NUM_EPOCHS,
    batch_size         = BATCH_SIZE,
    lmbd               = LMBD,
    gamma              = GAMMA,
    seed               = SEED,
    num_hidden_layers  = NUM_HIDDEN_LAYERS,
    initial_learning_rate = INITIAL_LR,
    beta_1             = BETA_1,
    beta_2             = BETA_2,
    verbose            = True,
    signed             = False
)

# ================================================================
# ============ 5. 评估 ============
# ================================================================
GC_est = a_hat_binary.astype(int)   # shape: (p, p)，GC_est[effect, cause]

tp = int(np.sum((GC_true == 1) & (GC_est == 1)))
fp = int(np.sum((GC_true == 0) & (GC_est == 1)))
fn = int(np.sum((GC_true == 1) & (GC_est == 0)))
tn = int(np.sum((GC_true == 0) & (GC_est == 0)))

accuracy  = 100.0 * (tp + tn) / (p * p)
precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
f1        = (2 * precision * recall / (precision + recall)
             if (precision + recall) > 0 else 0.0)

print('\n' + '=' * 55)
print(f'模型: GVAR (TRGC)')
print(f'真实因果边数: {np.sum(GC_true)}  |  预测因果边数: {np.sum(GC_est)}')
print(f'TP={tp}  FP={fp}  FN={fn}  TN={tn}')
print(f'Accuracy : {accuracy:.2f}%')
print(f'Precision: {precision:.4f}')
print(f'Recall   : {recall:.4f}')
print(f'F1 Score : {f1:.4f}')
print('=' * 55)

# ================================================================
# ============ 6. 保存因果矩阵为 CSV ============
# ================================================================
rows = []
for i in range(p):
    for j in range(p):
        if GC_est[i, j] == 1:
            rows.append([series_names[j], series_names[i], 1])

result_df = pd.DataFrame(rows)
result_df.to_csv(OUTPUT_CSV, index=False, header=False)
print(f'\n预测因果矩阵已保存至: {OUTPUT_CSV}')

# ================================================================
# ============ 7. 可视化对比 ============
# ================================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

titles = [
    'Ground Truth',
    f'GVAR Estimated  (F1={f1:.3f}, Acc={accuracy:.1f}%)'
]
for ax, mat, title in zip(axes, [GC_true, GC_est], titles):
    im = ax.imshow(mat, cmap='Blues', vmin=0, vmax=1, aspect='equal')
    ax.set_title(title, fontsize=13)
    ax.set_xticks(np.arange(p))
    ax.set_yticks(np.arange(p))
    ax.set_xticklabels(series_names, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(series_names, fontsize=9)
    ax.set_xlabel('Cause',  fontsize=10)
    ax.set_ylabel('Effect', fontsize=10)
    ax.tick_params(length=0)

# 在预测图上标注 TP（绿框）和错误（红框）
for i in range(p):
    for j in range(p):
        if GC_true[i, j] == 1 and GC_est[i, j] == 1:
            color = 'green'   # TP
        elif GC_true[i, j] != GC_est[i, j]:
            color = 'red'     # FP 或 FN
        else:
            continue
        axes[1].add_patch(plt.Rectangle(
            (j - 0.5, i - 0.5), 1, 1,
            facecolor='none', edgecolor=color, linewidth=2
        ))

# 添加图例
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='none', edgecolor='green', linewidth=2, label='TP (正确预测)'),
    Patch(facecolor='none', edgecolor='red',   linewidth=2, label='FP/FN (预测错误)'),
]
axes[1].legend(handles=legend_elements, loc='upper right', fontsize=8)

plt.tight_layout()
plt.savefig(OUTPUT_PNG, dpi=200, bbox_inches='tight')
plt.show()
print(f'可视化结果已保存至: {OUTPUT_PNG}')