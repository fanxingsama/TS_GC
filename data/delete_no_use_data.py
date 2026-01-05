import pandas as pd
import numpy as np

def remove_low_fluctuation_sensors(csv_path, output_path, fluctuation_threshold=50, ignore_na=True):
    """
    从时序CSV文件中删除波动次数极小的测点（列），同时删除.out后缀测点、同名测点中的SV后缀测点
    
    参数说明：
    csv_path: str - 输入CSV文件路径（如："input_data.csv"）
    output_path: str - 输出筛选后CSV文件路径（如："filtered_data.csv"）
    fluctuation_threshold: int - 波动次数阈值，低于该值的测点将被删除（默认50次，可按需调整）
    ignore_na: bool - 是否忽略缺失值（True=忽略，False=将NA视为波动）
    """
    # 1. 读取CSV文件，第一行作为列名（测点名）
    # 注意：确保CSV文件编码正确，若出现乱码可添加 encoding='gbk' 或 encoding='utf-8-sig'
    try:
        df = pd.read_csv(csv_path)
        print(f"成功读取CSV文件，共包含 {df.shape[1]} 个测点，{df.shape[0]} 条时序数据")
    except Exception as e:
        print(f"读取CSV文件失败：{e}")
        return
    
    # ===== 第一步：删除.out后缀的测点 =====
    # 筛选出不包含.out后缀的列
    non_out_columns = [col for col in df.columns if not col.endswith('.OUT')]
    out_columns = [col for col in df.columns if col.endswith('.OUT')]
    
    # 打印.out后缀测点的删除信息
    if out_columns:
        print(f"\n1. 检测到 {len(out_columns)} 个.out后缀的测点，已删除：")
        for col in out_columns:
            print(f"  - {col}")
    else:
        print("\n1. 未检测到.out后缀的测点")
    
    # 过滤.out后缀列
    df = df[non_out_columns]
    print(f"   过滤.out后缀后剩余 {df.shape[1]} 个测点")

    # ===== 第二步：删除同名测点中的SV后缀测点 =====
    # 构建测点名称映射：基名 -> [所有后缀列]
    sensor_base_map = {}
    for col in df.columns:
        # 分割基名和后缀（按最后一个.分割，如TIC_1101.PV → 基名TIC_1101，后缀PV）
        if '.' in col:
            base_name, suffix = col.rsplit('.', 1)
        else:
            base_name = col
            suffix = ""
        
        if base_name not in sensor_base_map:
            sensor_base_map[base_name] = []
        sensor_base_map[base_name].append((col, suffix))
    
    # 筛选保留的列：同名测点中只保留非SV的（优先保留PV，无PV则保留其他）
    non_sv_columns = []
    sv_removed_columns = []
    for base_name, col_suffix_list in sensor_base_map.items():
        # 分离SV和非SV列
        sv_cols = [col for col, suffix in col_suffix_list if suffix.upper() == 'SV']
        non_sv_cols = [col for col, suffix in col_suffix_list if suffix.upper() != 'SV']
        
        # 如果有同名的非SV列，删除SV列；如果只有SV列，则保留（避免误删）
        if non_sv_cols:
            non_sv_columns.extend(non_sv_cols)
            sv_removed_columns.extend(sv_cols)
        else:
            non_sv_columns.extend(sv_cols)  # 无其他同名列时保留SV列
    
    # 打印SV测点删除信息
    if sv_removed_columns:
        print(f"\n2. 检测到 {len(sv_removed_columns)} 个SV后缀的重复测点，已删除：")
        for col in sv_removed_columns:
            print(f"  - {col}")
    else:
        print("\n2. 未检测到需要删除的SV后缀重复测点")
    
    # 过滤SV后缀重复列
    df = df[non_sv_columns]
    print(f"   过滤SV后缀重复测点后剩余 {df.shape[1]} 个测点")
    
    # 2. 定义波动判断逻辑：遍历每个测点（列），计算其波动次数
    # 波动定义：当前行数据与上一行数据不相等（即序列值发生变化）
    kept_columns = []  # 存储需要保留的测点名称
    for col in df.columns:
        # 获取当前测点的时序数据
        sensor_data = df[col].values
        
        # 计算相邻数据的差异（判断是否波动）
        # np.diff 计算后一个元素减前一个元素，差值非0即为波动
        if ignore_na:
            # 忽略NA值：先去除NaN，再计算波动
            valid_data = sensor_data[~pd.isna(sensor_data)]
            if len(valid_data) < 2:
                # 有效数据不足2条，视为无波动，直接跳过
                print(f"\n测点 {col} 有效数据不足，波动次数为0，已删除")
                continue
            # 计算波动次数
            fluctuations = np.sum(np.diff(valid_data) != 0)
        else:
            # 不忽略NA值：将NaN与非NaN的变化也视为波动
            # 先将数据转为可比较的格式，再计算差异
            diff_array = np.diff(sensor_data)
            # 差值非0 或 存在NaN（相邻数据类型不一致，即出现NA波动）
            fluctuation_mask = (diff_array != 0) | pd.isna(diff_array)
            fluctuations = np.sum(fluctuation_mask)
        
        # 3. 判断是否保留该测点：波动次数 >= 阈值则保留
        if fluctuations >= fluctuation_threshold:
            kept_columns.append(col)
            print(f"\n测点 {col} 波动次数：{fluctuations}，保留")
        else:
            print(f"\n测点 {col} 波动次数：{fluctuations}（低于阈值 {fluctuation_threshold}），已删除")
    
    # 4. 筛选保留的测点，生成新的DataFrame
    filtered_df = df[kept_columns]
    print(f"\n==================== 预处理完成 ====================")
    print(f"原始测点总数：{len(non_out_columns) + len(out_columns)}")
    print(f"删除.out后缀测点：{len(out_columns)}")
    print(f"删除SV后缀重复测点：{len(sv_removed_columns)}")
    print(f"删除低波动测点：{len(non_sv_columns) - len(kept_columns)}")
    print(f"最终保留测点：{len(kept_columns)}")
    
    # 5. 保存筛选后的数据到新CSV文件
    try:
        filtered_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\n筛选后的数据已保存至：{output_path}")
    except Exception as e:
        print(f"保存CSV文件失败：{e}")
        return

# -------------------------- 主程序调用 --------------------------
if __name__ == "__main__":
    # 配置参数（按需修改！）
    INPUT_CSV_PATH = "TIC_1101异常低波动2.csv"  # 你的输入CSV文件路径
    OUTPUT_CSV_PATH = "TIC_1101异常低波动3.csv"  # 输出文件路径
    FLUCTUATION_THRESHOLD = 300  # 波动次数阈值（比如你说的几十次，这里默认50）
    
    # 执行低波动测点删除操作
    remove_low_fluctuation_sensors(
        csv_path=INPUT_CSV_PATH,
        output_path=OUTPUT_CSV_PATH,
        fluctuation_threshold=FLUCTUATION_THRESHOLD,
        ignore_na=True
    )