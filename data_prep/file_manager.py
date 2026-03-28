"""
SWAN数据集文件管理工具
用于查找、解压和组织数据文件路径。
"""
import os
import glob
import re
import tarfile
from datetime import datetime

def parse_timestamp_from_filename(filename):
    """
    从文件名中解析时间戳。
    支持多种格式。
    """
    # 尝试模式 sYYYY-MM-DDTHH:MM:SS
    match_iso_t = re.search(r"_s(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})", filename)
    if match_iso_t:
        try:
            return datetime.strptime(match_iso_t.group(1), "%Y-%m-%dT%H:%M:%S")
        except ValueError:
            pass

    # 尝试模式 YYYYMMDD_HHMMSS
    match_simple = re.search(r"(\d{8})_(\d{6})", filename)
    if match_simple:
        try:
            return datetime.strptime(match_simple.group(1) + match_simple.group(2), "%Y%m%d%H%M%S")
        except ValueError:
            pass

    # 尝试解析文件名中的其他时间格式
    # ... (可以根据需要添加更多解析逻辑) ...

    return None


def get_ar_files(data_dir, partition, include_nf_data=True):
    """
    获取指定分区下所有AR记录的CSV文件路径和时间戳。
    
    Args:
        data_dir (str): 数据集根目录。
        partition (int): 分区号 (e.g., 1, 2, 3, 4, 5)。
        include_nf_data (bool): 是否包含NF目录的数据。
    
    Returns:
        dict: 一个字典，键是AR记录ID (例如 '@12345')，
              值是与该记录相关的 (文件路径, 开始时间) 元组的列表。
              列表按开始时间排序。
    """
    partition_dir = os.path.join(data_dir, f"partition{partition}")
    
    sub_dirs_to_check = []
    fl_dir = os.path.join(partition_dir, "FL")
    nf_dir = os.path.join(partition_dir, "NF")

    if os.path.exists(fl_dir) and os.path.isdir(fl_dir):
        sub_dirs_to_check.append(fl_dir)
    else:
        print(f"警告: 目录 {fl_dir} 不存在或不是一个目录。")

    if include_nf_data:
        if os.path.exists(nf_dir) and os.path.isdir(nf_dir):
            sub_dirs_to_check.append(nf_dir)
        else:
            print(f"警告: (当include_nf_data=True时)目录 {nf_dir} 不存在或不是一个目录。")
    
    if not sub_dirs_to_check:
        print(f"错误: 在 partition{partition}下，FL 和 NF 目录均未找到或无效。")
        return {}

    all_csv_files = []
    for sub_dir_path in sub_dirs_to_check:
        general_pattern = os.path.join(sub_dir_path, "*.csv")
        current_subdir_files = glob.glob(general_pattern)
        all_csv_files.extend(current_subdir_files)

    if not all_csv_files:
        print(f"在 partition{partition} 的 FL/NF 目录中没有找到匹配的CSV文件。")
        return {}

    ar_files_map = {}
    parsed_count = 0
    failed_parse_count = 0

    for file_path in all_csv_files:
        filename = os.path.basename(file_path)
        
        record_id_str = None
        timestamp_obj = None

        # 优先使用详细的正则表达式匹配
        match1 = re.match(r"([A-Z0-9.]+)@(\d+):(?:Primary|Secondary)_ar(\d+)_s(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})_e.*\.csv", filename)
        if match1:
            sharp_id = match1.group(2)
            ar_num = match1.group(3)
            timestamp_str = match1.group(4)
            record_id_str = f"@{sharp_id}_ar{ar_num}"
            try:
                timestamp_obj = datetime.strptime(timestamp_str, "%Y-%m-%dT%H:%M:%S")
            except ValueError:
                failed_parse_count += 1
                continue
        else:
            match2 = re.match(r"FQ_ar(\d+)_s(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})_e.*\.csv", filename)
            if match2:
                ar_num = match2.group(1)
                timestamp_str = match2.group(2)
                record_id_str = f"@ar{ar_num}"
                try:
                    timestamp_obj = datetime.strptime(timestamp_str, "%Y-%m-%dT%H:%M:%S")
                except ValueError:
                    failed_parse_count += 1
                    continue
        
        # 如果详细匹配失败，则进行通用匹配
        if not record_id_str:
            generic_id_match = re.search(r"@(\d+)", filename)
            if not generic_id_match:
                generic_ar_id_match = re.search(r"_ar(\d+)", filename)
                if generic_ar_id_match:
                    record_id_str = f"@ar{generic_ar_id_match.group(1)}"
            else:
                record_id_str = f"@{generic_id_match.group(1)}"
        
        if not timestamp_obj:
            timestamp_obj = parse_timestamp_from_filename(filename)

        if record_id_str and timestamp_obj:
            if record_id_str not in ar_files_map:
                ar_files_map[record_id_str] = []
            ar_files_map[record_id_str].append((file_path, timestamp_obj))
            parsed_count += 1
        else:
            failed_parse_count += 1
            print(f"警告: 文件名未能解析ID或时间戳: {filename}")

    if parsed_count == 0 and len(all_csv_files) > 0:
        print(f"错误: 所有 {len(all_csv_files)} 个文件都未能成功解析记录ID或时间戳。")
    elif failed_parse_count > 0:
        print(f"信息: 成功解析 {parsed_count} 个文件，未能解析 {failed_parse_count} 个文件。")

    for record_id in ar_files_map:
        ar_files_map[record_id].sort(key=lambda x: x[1])

    print(f"找到{len(ar_files_map)}个唯一的记录ID，共包含{parsed_count}个有效文件")
    return ar_files_map


def extract_data_if_needed(data_dir, partition=1):
    """
    如果需要，解压数据集分区。
    
    Args:
        data_dir (str): 数据集根目录。
        partition (int): 分区编号。
            
    Returns:
        str: 解压后的分区目录。
    """
    partition_dir = os.path.join(data_dir, f"partition{partition}")
    
    if not os.path.exists(partition_dir):
        print(f"解压分区{partition}数据...")
        tar_file = os.path.join(data_dir, f"partition{partition}_instances.tar.gz")
        
        if os.path.exists(tar_file):
            with tarfile.open(tar_file) as tar:
                tar.extractall(path=data_dir)
            print(f"成功解压: {tar_file}")
        else:
            raise FileNotFoundError(f"找不到数据文件: {tar_file}")
            
    return partition_dir


def extract_all_partitions(data_dir, max_partition=5):
    """
    解压所有分区数据。
    
    Args:
        data_dir (str): 数据集根目录。
        max_partition (int): 最大分区编号。
    """
    print(f"正在检查并解压所有SWAN数据分区(1-{max_partition})...")
    
    for partition in range(1, max_partition + 1):
        partition_dir = os.path.join(data_dir, f"partition{partition}")
        
        if not os.path.exists(partition_dir):
            tar_file = os.path.join(data_dir, f"partition{partition}_instances.tar.gz")
            
            if os.path.exists(tar_file):
                print(f"正在解压分区{partition}...")
                with tarfile.open(tar_file) as tar:
                    tar.extractall(path=data_dir)
                print(f"成功解压: {tar_file}")
            else:
                print(f"找不到数据文件: {tar_file}")
        else:
            print(f"分区{partition}已解压")
            
    print("所有分区处理完成") 