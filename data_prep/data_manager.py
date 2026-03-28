"""
数据管理器
负责从原始文件中加载和合并数据。
"""
import logging
import pandas as pd
import numpy as np
from . import file_manager

class DataManager:
    """
    一个用于管理数据加载和合并的类。
    """
    def __init__(self, config):
        """
        初始化数据管理器。

        Args:
            config (dict): 包含数据相关参数的配置字典。
        """
        self.config = config
        self.specified_features = config.get('specified_features', [])
        
    def load_and_merge_data(self):
        """
        从分区加载并合并所有AR记录文件。
        """
        logging.info("开始数据加载和合并流程...")
        
        partitions = self.config.get('partitions_to_process', [])
        if not partitions:
            logging.warning("配置文件中未指定要处理的分区。")
            return []
            
        for p in partitions:
            file_manager.extract_data_if_needed(self.config['data_dir'], p)
            
        all_merged_samples = []
        for p in partitions:
            logging.info(f"正在处理分区 {p}...")
            ar_files_map = file_manager.get_ar_files(
                self.config['data_dir'], 
                p, 
                self.config.get('include_nf_data', True)
            )
            for record_id, files in ar_files_map.items():
                merged_sample = self._merge_ar_record_files(record_id, files)
                if merged_sample:
                    all_merged_samples.append(merged_sample)
        
        logging.info(f"数据加载和合并完成，共获得 {len(all_merged_samples)} 个长时序样本。")
        return all_merged_samples 

    def _merge_ar_record_files(self, record_id, files):
        """
        合并单个AR记录的所有CSV文件。
        """
        if not files:
            return None

        files.sort(key=lambda x: x[1])
        
        all_dfs = []
        for file_path, _ in files:
            try:
                # 假设分隔符是制表符，这在一些文件中可以看到。
                # 第一列也可能是索引或带有'#'。
                df = pd.read_csv(file_path, sep='\\t', engine='python', comment='#')

                # 动态查找并重命名时间戳列
                if 'timestamp' not in df.columns:
                    if 'Timestamp' in df.columns:
                        df.rename(columns={'Timestamp': 'timestamp'}, inplace=True)
                    elif df.columns[0] == '#': # 处理 '# timestamp' 格式
                        df.rename(columns={'#': 'timestamp'}, inplace=True)

                if not df.empty:
                    all_dfs.append(df)
            except Exception as e:
                logging.warning(f"读取或处理文件失败 {file_path}: {e}")
                continue

        if not all_dfs:
            return None
            
        try:
            merged_df = pd.concat(all_dfs, ignore_index=True)
        except Exception as e:
            logging.error(f"合并记录失败 {record_id} due to: {e}")
            return None

        if 'timestamp' not in merged_df.columns:
            logging.error(f"在记录 {record_id} 的文件中找不到 'timestamp' 列。可用列: {merged_df.columns.tolist()}")
            return None

        # --- 正确的事件、持续时间和时间戳确定逻辑 ---
        merged_df['timestamp_dt'] = pd.to_datetime(merged_df['timestamp'], errors='coerce')
        merged_df.dropna(subset=['timestamp_dt'], inplace=True)
        merged_df.sort_values('timestamp_dt', inplace=True)
        merged_df.drop_duplicates(subset=['timestamp_dt'], keep='first', inplace=True)


        if merged_df.empty:
            return None

        # 确保特征列顺序一致且为数值类型
        features_df = pd.DataFrame(columns=self.specified_features)
        available_features = [f for f in self.specified_features if f in merged_df.columns]
        if available_features:
            features_df[available_features] = merged_df[available_features]
        features_df = features_df.apply(pd.to_numeric, errors='coerce').fillna(0)
        features = features_df.values.astype(np.float32)

        # 基于第一个事件计算持续时间和标签
        timestamps_list = merged_df['timestamp'].astype(str).tolist()
        first_ts_dt = merged_df['timestamp_dt'].iloc[0]
        last_ts_dt = merged_df['timestamp_dt'].iloc[-1]

        label = 0
        event_timestamp_raw = None
        # 默认持续时间是整个观测窗口（删失情况）
        duration_hours = (last_ts_dt - first_ts_dt).total_seconds() / 3600.0

        # 使用标准列名检查耀斑事件
        flare_columns = ['MFLARE', 'XFLARE'] 
        
        for col in flare_columns:
            if col in merged_df.columns:
                numeric_col = pd.to_numeric(merged_df[col], errors='coerce').fillna(0)
                if numeric_col.any():
                    label = 1
                    try:
                        # 找到第一个耀斑事件的时间戳
                        first_flare_idx = numeric_col[numeric_col > 0].index[0]
                        event_ts_dt = merged_df.loc[first_flare_idx, 'timestamp_dt']
                        event_timestamp_raw = merged_df.loc[first_flare_idx, 'timestamp']
                        
                        # 如果事件发生，持续时间是到事件发生的时间
                        if pd.notna(first_ts_dt) and pd.notna(event_ts_dt) and event_ts_dt >= first_ts_dt:
                            duration_hours = (event_ts_dt - first_ts_dt).total_seconds() / 3600.0
                        
                    except (IndexError, KeyError):
                        label = 0 # 如果找不到时间戳则恢复为删失
                        event_timestamp_raw = None
                    break # 找到一种耀斑类型后停止
        
        if duration_hours < 0: 
            duration_hours = 0.0

        # 返回时确保包含标准化的持续时间字段（以小时为单位），并保留旧键 'time' 以兼容历史代码
        return {
            'record_id': record_id,
            'features': features_df,
            'timestamps_list': timestamps_list,
            'event': label,
            'time': float(duration_hours),  # 旧键，保留兼容性
            'duration': float(duration_hours),
            'duration_hours': float(duration_hours),
            'duration_units': 'hours',
            'event_time_raw': event_timestamp_raw,
            'start_time': first_ts_dt,
            'end_time': last_ts_dt
        }