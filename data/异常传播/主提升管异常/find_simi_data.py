import pandas as pd
import numpy as np
import os
from scipy.stats import pearsonr

def find_duplicate_sensor_columns(
    input_file,
    encoding='utf-8',
    na_values=['', 'NA', 'nan'],
    similarity_method='scaled_correlation',  # 新增scaled_correlation模式
    corr_threshold=0.99,              # 相关系数阈值
    value_tolerance=1e-6,             # 数值误差容忍度
    percent_threshold=99.0,           # 相同值占比阈值
    scale_tolerance=0.05,             # 数值比例容忍度（新增，允许的相对差异）
    show_details=True
):
    """
    识别CSV中测量同一测点的重复传感器列
    
    新增参数:
        scale_tolerance (float): 数值比例容忍度（0~1），例如0.05表示允许±5%的差异
    """
    # 输入验证
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"输入文件不存在: {input_file}")
    valid_methods = ['correlation', 'value', 'percent', 'scaled_correlation']  # 新增模式
    if similarity_method not in valid_methods:
        raise ValueError(f"相似度判断方式仅支持：{', '.join(valid_methods)}")
    if not (0 <= corr_threshold <= 1):
        raise ValueError("相关系数阈值必须在0~1之间")
    if not (0 <= percent_threshold <= 100):
        raise ValueError("相同值占比阈值必须在0~100之间")
    if not (0 <= scale_tolerance <= 1):  # 新增验证
        raise ValueError("数值比例容忍度必须在0~1之间")
    
    try:
        # 读取数据（省略部分重复代码）
        print(f"正在读取文件: {input_file}")
        df = pd.read_csv(
            input_file,
            encoding=encoding,
            na_values=na_values,
            dtype='float64'
        )
        
        df = df.dropna(axis=1, how='all')
        cols = df.columns.tolist()
        total_cols = len(cols)
        print(f"有效测点数量: {total_cols}")
        
        if total_cols < 2:
            print("⚠️ 警告: 有效测点数量少于2，无需检测重复")
            return {}
        
        checked_cols = set()
        duplicate_groups = []
        
        print(f"\n正在使用 [{similarity_method}] 方法检测重复测点...")
        # 阈值信息（新增scaled_correlation的描述）
        if similarity_method == 'correlation':
            threshold_info = f'相关系数 ≥ {corr_threshold}'
        elif similarity_method == 'value':
            threshold_info = f'数值误差 ≤ {value_tolerance}'
        elif similarity_method == 'percent':
            threshold_info = f'相同值占比 ≥ {percent_threshold}%'
        else:  # scaled_correlation
            threshold_info = f'相关系数 ≥ {corr_threshold} 且 数值相对差异 ≤ {scale_tolerance*100}%'
        print(f"判断阈值: {threshold_info}")
        
        for i, col1 in enumerate(cols):
            if col1 in checked_cols:
                continue
            
            vals1 = df[col1].dropna()
            if len(vals1) == 0:
                checked_cols.add(col1)
                continue
            
            current_group = [col1]
            checked_cols.add(col1)
            
            for j, col2 in enumerate(cols):
                if i >= j or col2 in checked_cols:
                    continue
                
                vals2 = df[col2].dropna()
                if len(vals2) == 0:
                    checked_cols.add(col2)
                    continue
                
                common_idx = vals1.index.intersection(vals2.index)
                if len(common_idx) < 2:
                    continue
                
                vals1_common = vals1.loc[common_idx]
                vals2_common = vals2.loc[common_idx]
                
                is_similar = False
                similarity = 0.0
                
                if similarity_method == 'correlation':
                    # 原有逻辑：仅趋势
                    corr, _ = pearsonr(vals1_common, vals2_common)
                    similarity = corr
                    is_similar = corr >= corr_threshold
                
                elif similarity_method == 'scaled_correlation':
                    # 新增逻辑：同时判断趋势和数值量级
                    # 1. 趋势相似性（相关系数）
                    corr, _ = pearsonr(vals1_common, vals2_common)
                    # 2. 数值量级相似性（相对差异）
                    # 避免除以0，使用均值作为基准
                    mean_val = np.mean([vals1_common.mean(), vals2_common.mean()])
                    if mean_val == 0:
                        # 若均值为0，直接比较绝对差异
                        abs_diff = np.abs(vals1_common - vals2_common).mean()
                        relative_diff = abs_diff / (value_tolerance + 1e-9)  # 避免除以0
                    else:
                        # 计算平均相对差异
                        abs_diff = np.abs(vals1_common - vals2_common)
                        relative_diff = np.mean(abs_diff / mean_val)
                    # 综合判断：相关系数达标且相对差异在容忍范围内
                    is_similar = (corr >= corr_threshold) and (relative_diff <= scale_tolerance)
                    # 相似度取两者的最小值（同时满足才高）
                    similarity = min(corr, 1 - relative_diff)
                
                elif similarity_method == 'value':
                    # 原有逻辑：逐值匹配
                    abs_diff = np.abs(vals1_common - vals2_common)
                    max_diff = abs_diff.max()
                    similarity = 1.0 if max_diff <= value_tolerance else 0.0
                    is_similar = max_diff <= value_tolerance
                
                elif similarity_method == 'percent':
                    # 原有逻辑：相同值占比
                    same_vals = np.abs(vals1_common - vals2_common) <= value_tolerance
                    same_percent = (same_vals.sum() / len(same_vals)) * 100
                    similarity = same_percent
                    is_similar = same_percent >= percent_threshold
                
                if is_similar:
                    current_group.append(col2)
                    checked_cols.add(col2)
                    if show_details:
                        print(f"  ✅ {col1} ↔ {col2}: 相似度 = {similarity:.4f}")
        
            if len(current_group) >= 2:
                duplicate_groups.append(current_group)
        
        # 输出结果（省略部分重复代码）
        print("\n" + "="*60)
        if duplicate_groups:
            print(f"检测到 {len(duplicate_groups)} 组重复测点（测量同一物理量）:")
            for idx, group in enumerate(duplicate_groups, 1):
                print(f"\n第 {idx} 组重复测点 ({len(group)} 个传感器):")
                for col in group:
                    print(f"  - {col}")
        else:
            print("✅ 未检测到测量同一测点的重复传感器列")
        
        # 统计信息
        total_dup_to_remove = sum(len(group) - 1 for group in duplicate_groups)
        total_dup_in_groups = sum(len(group) for group in duplicate_groups)
        unique_point_count = total_cols - total_dup_to_remove
        
        print("\n=== 检测统计 ===")
        print(f"总有效测点数量: {total_cols}")
        print(f"重复组数量: {len(duplicate_groups)}")
        print(f"参与重复组的传感器总数: {total_dup_in_groups}")
        print(f"需要删除的重复传感器数量: {total_dup_to_remove}")
        print(f"去重后唯一测点数量: {unique_point_count}")
        
        return {
            'duplicate_groups': duplicate_groups,
            'total_columns': total_cols,
            'total_duplicate_groups': len(duplicate_groups),
            'total_duplicate_sensors_in_groups': total_dup_in_groups,
            'total_duplicate_to_remove': total_dup_to_remove,
            'unique_measure_points': unique_point_count
        }
        
    except Exception as e:
        raise RuntimeError(f"检测重复测点时出错: {str(e)}")

# 主执行逻辑
if __name__ == "__main__":
    INPUT_CSV_PATH = "提升管阀门开口全开.csv"
    FILE_ENCODING = "utf-8"
    
    # 使用新增的scaled_correlation模式
    SIMILARITY_METHOD = "scaled_correlation"
    CORR_THRESHOLD = 0.999          # 趋势相似度阈值
    SCALE_TOLERANCE = 0.03          # 允许±3%的数值差异（可根据需求调整）
    VALUE_TOLERANCE = 1e-6
    PERCENT_THRESHOLD = 99.0
    
    try:
        result = find_duplicate_sensor_columns(
            input_file=INPUT_CSV_PATH,
            encoding=FILE_ENCODING,
            similarity_method=SIMILARITY_METHOD,
            corr_threshold=CORR_THRESHOLD,
            scale_tolerance=SCALE_TOLERANCE,  # 新增参数
            value_tolerance=VALUE_TOLERANCE,
            percent_threshold=PERCENT_THRESHOLD,
            show_details=True
        )
    except Exception as e:
        print(f"错误: {e}")