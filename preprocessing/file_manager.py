import os
import re
import logging
from collections import defaultdict
import glob
import tarfile
from datetime import datetime

logger = logging.getLogger(__name__)

class FileManager:
    """
    负责发现、解压和管理数据集文件。
    参考 swan_data_processor.py 中的文件处理逻辑进行重构。
    """
    def __init__(self, data_directory, include_nf_data=False):
        """
        初始化文件管理器。

        Args:
            data_directory (str): 数据集的根目录。
            include_nf_data (bool): 是否也包括 'NF' 文件夹中的数据。
        """
        if not os.path.isdir(data_directory):
            raise ValueError(f"提供的数据目录不存在: {data_directory}")
        self.data_directory = data_directory
        self.include_nf_data = include_nf_data
        logger.info(f"FileManager 初始化于目录: {self.data_directory}")
        logger.info(f"是否包含NF数据: {self.include_nf_data}")

    def extract_data_if_needed(self, partition):
        """如果需要，解压数据集分区。"""
        partition_dir = os.path.join(self.data_directory, f"partition{partition}")
        if not os.path.exists(partition_dir):
            tar_file = os.path.join(self.data_directory, f"partition{partition}_instances.tar.gz")
            if os.path.exists(tar_file):
                logger.info(f"正在为分区 {partition} 解压数据 from {tar_file}...")
                with tarfile.open(tar_file) as tar:
                    tar.extractall(path=self.data_directory)
                logger.info(f"成功解压: {tar_file}")
            else:
                logger.warning(f"数据压缩包未找到: {tar_file}，将继续执行。")
        return partition_dir

    def get_ar_files(self, partition):
        """
        获取指定分区下所有AR记录的CSV文件路径和时间戳。
        该实现基于 swan_data_processor.py 中的 get_ar_files 方法。
        """
        partition_dir_template = os.path.join(self.data_directory, f"partition{partition}")
        
        sub_dirs_to_check = []
        fl_dir = os.path.join(partition_dir_template, "FL")
        nf_dir = os.path.join(partition_dir_template, "NF")

        if os.path.exists(fl_dir) and os.path.isdir(fl_dir):
            logger.info(f"从 {fl_dir} 加载数据...")
            sub_dirs_to_check.append(fl_dir)

        if self.include_nf_data and os.path.exists(nf_dir) and os.path.isdir(nf_dir):
            logger.info(f"从 {nf_dir} 加载数据...")
            sub_dirs_to_check.append(nf_dir)

        if not sub_dirs_to_check:
            logger.error(f"在 partition{partition}下，FL 和/或 NF 目录均未找到。")
            return {}

        all_csv_files = []
        for sub_dir_path in sub_dirs_to_check:
            general_pattern = os.path.join(sub_dir_path, "*.csv")
            all_csv_files.extend(glob.glob(general_pattern))

        if not all_csv_files:
            logger.warning(f"在 partition{partition} 的目录中没有找到CSV文件。")
            return {}

        ar_files_map = defaultdict(list)
        parsed_count = 0
        failed_parse_count = 0

        for file_path in all_csv_files:
            filename = os.path.basename(file_path)
            record_id_str, timestamp_obj = self._parse_filename_robust(filename)
            
            if record_id_str and timestamp_obj:
                ar_files_map[record_id_str].append((file_path, timestamp_obj))
                parsed_count += 1
            else:
                failed_parse_count += 1
        
        if failed_parse_count > 0:
            logger.info(f"成功解析 {parsed_count} 个文件名, 未能解析 {failed_parse_count} 个。")

        for record_id in ar_files_map:
            ar_files_map[record_id].sort(key=lambda x: x[1])

        logger.info(f"在分区 {partition} 中找到 {len(ar_files_map)} 个唯一的记录ID。")
        return ar_files_map

    def _parse_filename_robust(self, filename):
        """
        从文件名稳健地解析记录ID和时间戳，尝试多种格式。
        该实现基于 swan_data_processor.py 中的 get_ar_files 内部逻辑。
        """
        record_id_str = None
        timestamp_obj = None

        # 格式1: M1.7@3189:Primary_ar1321_s...
        match1 = re.match(r"([A-Z0-9.]+)@(\d+):(?:Primary|Secondary)_ar(\d+)_s(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})_e.*\.csv", filename)
        if match1:
            sharp_id = match1.group(2)
            ar_num = match1.group(3)
            timestamp_str = match1.group(4)
            record_id_str = f"@{sharp_id}_ar{ar_num}"
            try:
                timestamp_obj = datetime.strptime(timestamp_str, "%Y-%m-%dT%H:%M:%S")
                return record_id_str, timestamp_obj
            except ValueError as e:
                logger.warning(f"解析时间戳失败 (格式1) {timestamp_str} from {filename}: {e}")
                return None, None
        
        # 格式2: FQ_ar610_s...
        match2 = re.match(r"FQ_ar(\d+)_s(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})_e.*\.csv", filename)
        if match2:
            ar_num = match2.group(1)
            timestamp_str = match2.group(2)
            record_id_str = f"@ar{ar_num}"
            try:
                timestamp_obj = datetime.strptime(timestamp_str, "%Y-%m-%dT%H:%M:%S")
                return record_id_str, timestamp_obj
            except ValueError as e:
                logger.warning(f"解析时间戳失败 (格式2) {timestamp_str} from {filename}: {e}")
                return None, None
        
        # 通用格式 (作为后备)
        match_generic = re.search(r"_ar(\d+)_s(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})", filename)
        if match_generic:
            ar_num = match_generic.group(1)
            timestamp_str = match_generic.group(2)
            record_id_str = f"@ar{ar_num}"
            try:
                timestamp_obj = datetime.strptime(timestamp_str, "%Y-%m-%dT%H:%M:%S")
                return record_id_str, timestamp_obj
            except ValueError as e:
                logger.warning(f"解析时间戳失败 (通用格式) {timestamp_str} from {filename}: {e}")
                return None, None

        logger.debug(f"文件名 {filename} 不符合任何预期的格式。")
        return None, None