import argparse
from parse_config import ConfigParser
from datetime import datetime
import train
import interpret
import torch
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from utils import read_json
import tempfile

# python runner.py -c config/config_fMRI.json

def construct_demo():
    task_list = {}
    for i in [15]:
        task_list[f'fMRI{i}'] = {
            'dataset': f"../fMRI/timeseries{i}.csv",
            'groundtruth': f"../fMRI/sim{i}_gt_processed.csv"
        }
    return task_list

def construct_fMRI():
    task_list = {}
    for i in range(1,29):
        task_list[f'fMRI{i}'] = {
            'dataset': f"../fMRI/timeseries{i}.csv",
            'groundtruth': f"../fMRI/sim{i}_gt_processed.csv"
        }
    return task_list

def construct_linear():
    """linear 数据集任务，只有一个"""
    task_list = {
        'linear': {
            'dataset': '../../util/matrix/linear/time_series_linear.csv',
            'groundtruth': '../../util/matrix/linear/causal_linear.csv'
        }
    }
    return task_list

tasks = {
    'demo':   construct_demo,
    'fMRI':   construct_fMRI,
    'linear': construct_linear,
}    

def convert_gt_to_index(gt_path, columns):
    """
    将 causal_linear.csv（cause, effect, 1）
    转换为 evaluator.py 期望的格式（cause_idx, effect_idx, delay）
    key 用整数索引，与 columns 列表对应
    """
    name_to_idx = {name.strip().lower(): idx for idx, name in enumerate(columns)}
    gt_df = pd.read_csv(gt_path, header=None)
    
    rows = []
    for _, row in gt_df.iterrows():
        cause_str  = str(row.iloc[0]).strip().lower()
        effect_str = str(row.iloc[1]).strip().lower()
        c = name_to_idx.get(cause_str,  -1)
        e = name_to_idx.get(effect_str, -1)
        if c >= 0 and e >= 0:
            rows.append([c, e, 1])   # cause_idx, effect_idx, delay=1
    
    converted_df = pd.DataFrame(rows)
    
    # 写入临时文件
    tmp = tempfile.NamedTemporaryFile(
        mode='w', suffix='.csv', delete=False,
        dir=os.path.dirname(os.path.abspath(gt_path)),
        prefix='converted_gt_'
    )
    converted_df.to_csv(tmp.name, index=False, header=False)
    tmp.close()
    print(f'GT 已转换（字符串→整数索引），临时文件: {tmp.name}')
    return tmp.name

def runtask(label, args, dataset, ground_truth, task_name):
    # fix random seeds for reproducibility
    # fix random seeds for reproducibility
    SEED = 123
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(SEED)

    # ✅ 第一步：先读取列名，转换 GT 格式
    ts_df   = pd.read_csv(dataset)
    columns = ts_df.columns.tolist()
    
    converted_gt = None
    if ground_truth and os.path.exists(ground_truth):
        converted_gt = convert_gt_to_index(ground_truth, columns)
    
    gt_to_use = converted_gt if converted_gt else ground_truth

    # ✅ 第二步：训练
    args_dict = {
        'name':    f'Batch Runner/{label}/{task_name}',
        'config':  args.config,
        'resume':  None,
        'device':  args.device,
        'data_dir': dataset
    }
    config = ConfigParser.from_args(args=args_dict, run_id='model')
    train.main(config)
    torch.cuda.empty_cache()

    # ✅ 第三步：推断（传转换后的 GT）
    model, config, data_loader = interpret.load_model(
        f'saved/models/Batch Runner/{label}/{task_name}/model',
        args,
        f'Batch Runner/{label}/{task_name}',
        'casuality'
    )
    interpret.main(model, config, data_loader, gt_to_use)   # ← 关键：用 gt_to_use
    torch.cuda.empty_cache()
    
    # ✅ 给输出 CSV 补加第三列（全为1）
    pred_csv = os.path.join(os.getcwd(), 'csv', f'CausalFormer_{task_name}.csv')
    if os.path.exists(pred_csv):
        pred_df = pd.read_csv(pred_csv, header=None)
        if pred_df.shape[1] == 2:          # 只有2列时才补
            pred_df[2] = 1
            pred_df.to_csv(pred_csv, index=False, header=False)
            print(f'已为预测 CSV 添加第三列（全为1）: {pred_csv}')

    # ✅ 第四步：清理临时文件
    if converted_gt and os.path.exists(converted_gt):
        os.remove(converted_gt)
        print(f'临时文件已清理: {converted_gt}')

def evaluate_linear(gt_path, pred_csv_path, series_names):
    """读取预测CSV和真实矩阵，计算指标并绘图"""
    p           = len(series_names)
    name_to_idx = {n.strip().lower(): i for i, n in enumerate(series_names)}

    # ---- 读取真实矩阵（原始字符串格式）----
    GC_true = np.zeros((p, p), dtype=int)
    gt_df   = pd.read_csv(gt_path, header=None)
    for _, row in gt_df.iterrows():
        c = name_to_idx.get(str(row.iloc[0]).strip().lower(), -1)
        e = name_to_idx.get(str(row.iloc[1]).strip().lower(), -1)
        if 0 <= c < p and 0 <= e < p:
            GC_true[e, c] = 1

    # ---- 读取预测矩阵 ----
    GC_est = np.zeros((p, p), dtype=int)
    if not os.path.exists(pred_csv_path):
        print(f'⚠️  预测 CSV 未找到: {pred_csv_path}')
        return
    
    pred_df = pd.read_csv(pred_csv_path, header=None)
    for _, row in pred_df.iterrows():
        c = name_to_idx.get(str(row.iloc[0]).strip().lower(), -1)
        e = name_to_idx.get(str(row.iloc[1]).strip().lower(), -1)
        if 0 <= c < p and 0 <= e < p:
            GC_est[e, c] = 1

    # ---- 计算指标 ----
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
    print('  CausalFormer - Linear 数据集评估结果')
    print('=' * 55)
    print(f'真实因果边数: {np.sum(GC_true)}  |  预测因果边数: {np.sum(GC_est)}')
    print(f'TP={tp}  FP={fp}  FN={fn}  TN={tn}')
    print(f'Accuracy : {accuracy:.2f}%')
    print(f'Precision: {precision:.4f}')
    print(f'Recall   : {recall:.4f}')
    print(f'F1 Score : {f1:.4f}')
    print('=' * 55)

    # ---- 可视化 ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, mat, title in zip(
        axes,
        [GC_true, GC_est],
        ['Ground Truth',
         f'CausalFormer Estimated (F1={f1:.3f}, Acc={accuracy:.1f}%)']
    ):
        ax.imshow(mat, cmap='Blues', vmin=0, vmax=1, aspect='equal')
        ax.set_title(title, fontsize=13)
        ax.set_xticks(np.arange(p))
        ax.set_yticks(np.arange(p))
        ax.set_xticklabels(series_names, rotation=45, ha='right', fontsize=9)
        ax.set_yticklabels(series_names, fontsize=9)
        ax.set_xlabel('Cause',  fontsize=10)
        ax.set_ylabel('Effect', fontsize=10)
        ax.tick_params(length=0)

    for i in range(p):
        for j in range(p):
            if   GC_true[i,j] == 1 and GC_est[i,j] == 1: color = 'green'
            elif GC_true[i,j] != GC_est[i,j]:             color = 'red'
            else: continue
            axes[1].add_patch(plt.Rectangle(
                (j-0.5, i-0.5), 1, 1,
                facecolor='none', edgecolor=color, linewidth=2))

    from matplotlib.patches import Patch
    axes[1].legend(handles=[
        Patch(facecolor='none', edgecolor='green', linewidth=2, label='TP'),
        Patch(facecolor='none', edgecolor='red',   linewidth=2, label='FP/FN'),
    ], loc='upper right', fontsize=9)

    plt.tight_layout()
    out_png = os.path.join(os.path.dirname(pred_csv_path),
                           'CausalFormer_linear_result.png')
    plt.savefig(out_png, dpi=200, bbox_inches='tight')
    plt.show()
    print(f'可视化已保存至: {out_png}')

def main(args):
    import re, os

    configJSON = read_json(args.config)
    data_dir = configJSON['data_loader']['args']['data_dir']
    dataset_name = os.path.basename(data_dir).replace('.csv', '')

    label = datetime.now().strftime(r'%m%d_%H%M%S')

    # ✅ 判断是否是 linear 任务
    is_linear = (dataset_name == 'time_series_linear')

    if is_linear:
        gt_path = '../../util/matrix/linear/causal_linear.csv'
        if not os.path.exists(gt_path):
            gt_path = None
    else:
        match = re.search(r'timeseries(\d+)', dataset_name)
        if match:
            idx = match.group(1)
            gt_path = data_dir.replace(f'timeseries{idx}.csv', f'sim{idx}_gt_processed.csv')
        else:
            gt_path = None
        if gt_path and not os.path.exists(gt_path):
            gt_path = None

    print(f"===========================================================")
    print(f"🚀 开始运行 CausalFormer")
    print(f"📂 数据集: {dataset_name}")
    print(f"🎯 GroundTruth: {gt_path if gt_path else '未找到'}")
    print(f"===========================================================\n")

    runtask(label, args, data_dir, gt_path, dataset_name)

    # ✅ linear 任务：额外调用评估+可视化
    if is_linear:
        pred_csv = os.path.join(
            os.getcwd(), 'csv', f'CausalFormer_{dataset_name}.csv'
        )
        series_names = ['x0','x1','x2','x3','x4','x5','x6','x7']
        evaluate_linear(gt_path, pred_csv, series_names)

    # 读取日志摘要
    save_dir = Path(configJSON['trainer']['save_dir'])
    fname = f'{save_dir}/log/Batch Runner/{label}/{dataset_name}/casuality/info.log'
    if os.path.exists(fname):
        with open(fname, 'r') as f:
            lines = f.readlines()
            try:
                result = {
                    "Precision'": float(lines[-8].split(':')[-1][:-1]),
                    "Recall'":    float(lines[-7].split(':')[-1][:-1]),
                    "F1'":        float(lines[-6].split(':')[-1][:-1]),
                    "Precision":  float(lines[-4].split(':')[-1][:-1]),
                    "Recall":     float(lines[-3].split(':')[-1][:-1]),
                    "F1":         float(lines[-2].split(':')[-1][:-1]),
                    "PoD":        float(lines[-1].split(':')[-1][:-2]) / 100
                }
                df = pd.DataFrame([result], index=[1])
                print("\n===================Summary===================")
                print('\t' + df.to_string().replace('\n', '\n\t'))
            except Exception:
                pass


if __name__ == "__main__":
    args = argparse.ArgumentParser(description='CausalityInterpret')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-d', '--device', default="0", type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-t', '--task', default='fMRI', type=str,
                      help='task (default: fMRI)')
    args = args.parse_args()
    main(args)
