import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

# 全局字体设置
rcParams['font.family'] = 'SimHei'
rcParams['axes.unicode_minus'] = False

# 实验设置的 SNR 水平 (dB)
snr_levels = [20, 15, 10, 5, 0]
# 实验数据
ts_gc_f1 = [0.868, 0.868, 0.868, 0.829, 0.766]          # 自研模型 TS-GC (Ours)
baseline_TCDF_f1 = [0.79, 0.79, 0.74, 0.70, 0.616] 
baseline_CausalFormer_f1 = [0.8, 0.80, 0.77, 0.73, 0.68]    
baseline_GVAR_f1 = [0.770, 0.770, 0.730, 0.68, 0.62] 
baseline_cMLP_f1 = [0.81, 0.81, 0.79, 0.75, 0.66] 
baseline_cLSTM_f1 = [0.76, 0.76, 0.71, 0.66, 0.58]   

# 创建绘图（加宽画布适配5个基准模型的图例）
plt.figure(figsize=(10, 5))

# 绘制自研模型（突出显示，层级置顶）
plt.plot(snr_levels, ts_gc_f1, marker='o', linestyle='-', linewidth=3, 
         label='TS-GC', color='#d62728', markersize=9, zorder=10)
# 绘制5个基准模型（不同颜色/标记/线型区分）
plt.plot(snr_levels, baseline_TCDF_f1, marker='s', linestyle='--', linewidth=2,
         label='TCDF', color='#1f77b4', markersize=7)
plt.plot(snr_levels, baseline_CausalFormer_f1, marker='^', linestyle='--', linewidth=2,
         label='CausalFormer', color='#2ca02c', markersize=7)
plt.plot(snr_levels, baseline_GVAR_f1, marker='d', linestyle='--', linewidth=2,
         label='GVAR', color='#ff7f0e', markersize=7)
plt.plot(snr_levels, baseline_cMLP_f1, marker='*', linestyle='--', linewidth=2,
         label='cMLP', color='#9467bd', markersize=8)
plt.plot(snr_levels, baseline_cLSTM_f1, marker='p', linestyle='--', linewidth=2,
         label='cLSTM', color='#8c564b', markersize=7)

# 设置轴标签和标题
plt.xlabel(r'信噪比大小', fontsize=16)  # 补充单位，更规范
plt.ylabel(r'$F1$ 值', fontsize=16)
# plt.title('Robustness Comparison under Different Noise Intensities', fontsize=15, pad=15)

# 反转X轴（从左到右噪声增强：20dB→0dB）
plt.gca().invert_xaxis() 

# 核心优化：调整y轴范围，剔除空白区域，放大模型差距
plt.ylim(0.45, 0.9)  # 替换原0-1，贴合数据分布
# 精细化y轴刻度，间隔0.05，对比更清晰
plt.yticks(np.arange(0.45, 0.91, 0.05), fontsize=14)

# 其他优化细节
plt.tick_params(axis='both', direction='in')  # 刻度线朝向图内
plt.grid(True, linestyle=':', alpha=0.7)  # 网格更清晰
plt.legend(loc='lower left', fontsize=14, framealpha=0.9)  # 微调图例字体，避免遮挡
plt.xticks(snr_levels, fontsize=16)  # X轴刻度匹配SNR值
plt.tight_layout()  # 自动调整布局，避免标签/图例裁剪

# 保存图片（高分辨率，适配论文排版）
plt.savefig('robustness_experiment_f1.pdf', dpi=300, bbox_inches='tight')
plt.show()