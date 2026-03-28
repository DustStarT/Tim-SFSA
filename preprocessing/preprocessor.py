"""
数据预处理器
负责编排整个数据准备流程，包括加载、合并、特征选择、数据划分和批处理生成。
"""
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
import os
import json


def sanitize_numeric_df(df, max_abs=1e6):
    """Sanitize numeric DataFrame: replace inf with nan, clip extreme values, fill na with column median."""
    df_clean = df.copy()
    # replace inf/ -inf with nan
    df_clean.replace([np.inf, -np.inf], np.nan, inplace=True)
    # clip extreme values
    try:
        df_clean = df_clean.clip(lower=-max_abs, upper=max_abs)
    except Exception:
        pass
    # fill nan with column median (or 0 if median is nan)
    for col in df_clean.columns:
        col_vals = df_clean[col]
        median = col_vals.median(skipna=True)
        if pd.isna(median):
            median = 0.0
        df_clean[col].fillna(median, inplace=True)
    return df_clean

# 从项目模块导入
from data_prep import subsequence_generator
from data_prep.data_manager import DataManager
from . import feature_selector
from .outlier_handler import OutlierHandler
from .feature_transformer import FeatureTransformer

# 1. 定义flare相关特征名
flare_related_features = [
    'TOTUSJH', 'TOTBSQ', 'TOTPOT', 'TOTUSJZ', 'ABSNJZH',
    'USFLUX', 'TOTFZ', 'MEANPOT', 'EPSZ', 'MEANSHR', 'SHRGT45'
]

# 2. 在归一化前，对flare特征做log1p变换
def log_transform_flare_features(X_df, feature_names):
    X_new = X_df.copy()
    for fname in flare_related_features:
        if fname in feature_names:
            X_new[fname] = np.sign(X_new[fname]) * np.log1p(np.abs(X_new[fname]))
    return X_new


class DataPreprocessor:
    """
    一个高级处理器，用于协调从原始文件到模型就绪数据的整个流程。
    """
    def __init__(self, config):
        """
        初始化数据预处理器。
        """
        self.config = config
        self.all_merged_samples = []
        self.train_samples = []
        self.test_samples = []
        self.scaler = None
        self.imputer = None
        self.outlier_handler = None
        self.feature_transformer = None
        self.active_feature_names = None
        self.specified_features = self.config.data.specified_features
        self.logger = logging.getLogger(__name__)

        # 初始化处理器
        self.outlier_handler = OutlierHandler(**self.config.data.outlier_config)
        
        # 从配置中获取feature_transformer的设置
        self.feature_transformer = FeatureTransformer(enabled=self.config.data.feature_transformation.enabled)

    def execute(self):
        """
        执行完整的预处理流水线。
        1. 加载数据
        2. 划分训练/测试父样本
        3. 在训练父样本上进行特征选择和拟合转换器
        """
        self.logger.info("开始执行数据预处理流水线...")
        
        # 1. 加载和合并数据
        data_manager = DataManager(self.config.data)
        self.all_merged_samples = data_manager.load_and_merge_data()
        self._log_missing_values()
        self.logger.info(f"数据加载和合并完成，共获得 {len(self.all_merged_samples)} 个长时序样本。")

        # 2. 划分父样本集
        # 三种划分模式：
        # - subsequence_independent: 先展开所有父样本为子序列，再按子序列独立划分
        # - group_balanced: 按归一化 record_id 做 group 划分，同时尝试平衡事件率
        # - group: 仅按 group 划分
        split_mode = getattr(self.config.data, 'split_mode', 'group')
        use_group = bool(getattr(self.config.data, 'use_group_split', True))

        # 若启用严格复现并提供了强制测试 record_ids，则优先按照该名单划分父样本
        strict_cfg = getattr(self.config.evaluation, 'strict_reval', None)
        enforced_done = False
        if strict_cfg and bool(getattr(strict_cfg, 'enabled', False)):
            force_ids = getattr(strict_cfg, 'force_test_record_ids', None)
            if force_ids:
                self.logger.info(f"严格复现启用：根据旧 run 的 record_ids 强制构建测试父样本集合（{len(force_ids)} 条子序列标识）")

                def _norm_group(obj):
                    # 统一不同类型的 record_id 表达
                    try:
                        import re
                        if obj is None:
                            return None
                        if isinstance(obj, dict):
                            raw = obj.get('raw')
                            if raw:
                                return str(raw).lower()
                            ar = obj.get('ar')
                            if ar:
                                return str(ar).lower()
                            # 回退：从任何值里提取 ar 编号
                            for v in obj.values():
                                s = str(v)
                                m = re.search(r'\b[aA][rR]?-?(\d+)\b', s)
                                if m:
                                    return f"ar{m.group(1)}".lower()
                            return str(obj).lower()
                        # 字符串：优先返回原样（lower）
                        s = str(obj)
                        if s:
                            return s.lower()
                    except Exception:
                        pass
                    return str(obj).lower()

                forced_groups = set()
                try:
                    for rid in force_ids:
                        # npz 中可能是 numpy.object_ -> 转 python
                        try:
                            if hasattr(rid, 'item'):
                                rid = rid.item()
                        except Exception:
                            pass
                        forced_groups.add(_norm_group(rid))
                except Exception:
                    forced_groups = set()

                if forced_groups:
                    # 将所有父样本按 group 归类（使用与常规 group split 相同的归一化方式）
                    train_list, test_list = [], []
                    for s in self.all_merged_samples:
                        rid = s.get('record_id')
                        g = _norm_group(rid)
                        # 兼容：若父样本的 rid 是 dict 且含 'raw'，尝试直接匹配其 raw
                        if g in forced_groups:
                            test_list.append(s)
                        else:
                            train_list.append(s)
                    self.test_samples = test_list
                    self.train_samples = train_list
                    self.train_is_subsequences = False
                    enforced_done = True
                    self.logger.info(f"严格复现划分完成: train={len(self.train_samples)}, test={len(self.test_samples)} (依据旧 run record_ids)")
                else:
                    self.logger.warning("严格复现启用但未能解析到有效的 forced record_ids；回退到常规划分逻辑。")

        if enforced_done:
            # 跳过常规划分逻辑
            pass
        elif split_mode == 'subsequence_independent':
            # 生成所有父样本的子序列，并按子序列级别做 stratified split
            self.logger.info('split_mode=subsequence_independent: 将父样本展开为子序列并在子序列级别执行划分。')
            X_all_3d, y_all, record_ids_all = self._generate_and_shape_features(self.all_merged_samples)
            if X_all_3d is None:
                raise ValueError('无法从父样本生成子序列，subsequence_independent 模式失败。')
            # 构建子序列级别的列表对象以兼容后续处理
            subseq_list = []
            for idx in range(len(record_ids_all)):
                subseq_list.append({
                    'features': X_all_3d[idx],
                    'duration': float(y_all[idx][0]) if y_all.ndim>1 else float(y_all[idx]),
                    'event': int(y_all[idx][1]) if y_all.ndim>1 else int(y_all[idx]),
                    'record_id': record_ids_all[idx]
                })
            # stratify by event label
            strata = [s['event'] for s in subseq_list]
            try:
                train_subseq, test_subseq = train_test_split(
                    subseq_list,
                    train_size=self.config.data.train_test_split_ratio,
                    random_state=self.config.data.random_seed,
                    stratify=strata
                )
            except Exception:
                # fallback to no stratify
                train_subseq, test_subseq = train_test_split(
                    subseq_list,
                    train_size=self.config.data.train_test_split_ratio,
                    random_state=self.config.data.random_seed
                )
            self.train_samples = train_subseq
            self.test_samples = test_subseq
            self.train_is_subsequences = True
            self.logger.info(f"subsequence_independent 划分完成: {len(self.train_samples)} 个训练子序列, {len(self.test_samples)} 个测试子序列。")

        elif split_mode == 'subsequence_temporal_per_parent':
            # Improved strategy: try per-parent temporal splits with varying train fractions to
            # find a global train/test partition whose overall event-rate diff is within tolerance.
            self.logger.info('split_mode=subsequence_temporal_per_parent: 以父内时间切分子序列，并尝试在全局事件率上进行平衡。')
            seq_gen = self.config.data.sequence_generation

            # Build subsequence lists per parent first
            parent_to_subseqs = []
            for sample in self.all_merged_samples:
                sample_with_cfg = sample.copy()
                sample_with_cfg['config'] = {'sequence_generation': {'include_aggregated_features': bool(seq_gen.get('include_aggregated_features', False))}}
                subseqs = subsequence_generator.generate_subsequences(
                    sample_with_cfg,
                    seq_gen['num_covariate_timesteps'],
                    seq_gen['prediction_window_hours'],
                    seq_gen['sub_sequence_step'],
                    self.active_feature_names
                )
                if subseqs:
                    try:
                        subseqs_sorted = sorted(subseqs, key=lambda s: (s.get('record_id') or {}).get('subseq_start') or 0)
                    except Exception:
                        subseqs_sorted = subseqs
                    parent_to_subseqs.append((sample.get('record_id'), subseqs_sorted))

            if not parent_to_subseqs:
                self.logger.warning('No subsequences generated for any parent in subsequence_temporal_per_parent mode.')
                self.train_samples, self.test_samples = [], []
                return

            # Candidate fractions to try (centered at configured temporal_parent_train_fraction)
            base_frac = float(getattr(self.config.data, 'temporal_parent_train_fraction', self.config.data.train_test_split_ratio))
            tol = float(getattr(self.config.data, 'temporal_parent_balance_tolerance', 0.05))
            max_tries = int(getattr(self.config.data, 'temporal_parent_balance_max_tries', 10))

            # create candidate fractions: base +/- small deltas
            deltas = [0.0]
            step = 0.05
            for i in range(1, max_tries):
                d = step * i
                deltas.append(-d)
                deltas.append(d)
            # limit candidates to [0.01,0.99]
            candidates = []
            for d in deltas:
                f = min(0.99, max(0.01, base_frac + d))
                if f not in candidates:
                    candidates.append(f)

            best = None
            for frac in candidates:
                per_parent_train = []
                per_parent_test = []
                for (rid, subseqs_sorted) in parent_to_subseqs:
                    n = len(subseqs_sorted)
                    split_idx = int(np.floor(n * frac))
                    if split_idx == 0 and n > 1 and frac > 0:
                        split_idx = 1
                    per_parent_train.append(subseqs_sorted[:split_idx])
                    per_parent_test.append(subseqs_sorted[split_idx:])

                # Ensure parents that contain positive subsequences contribute at least one positive to test
                ensure_per_parent_test_pos = True
                if ensure_per_parent_test_pos:
                    for i, (rid, subseqs_sorted) in enumerate(parent_to_subseqs):
                        train_part = per_parent_train[i]
                        test_part = per_parent_test[i]
                        # count positives
                        total_pos = sum(int(s.get('event', 0)) for s in subseqs_sorted)
                        test_pos = sum(int(s.get('event', 0)) for s in test_part)
                        # if parent has positives but none are in test, and test_part is non-empty,
                        # move the last positive from train_part to the front of test_part if possible
                        if total_pos > 0 and test_pos == 0 and len(test_part) > 0 and len(train_part) > 0:
                            # find last positive in train_part (prefer later ones)
                            moved = False
                            for j in range(len(train_part)-1, -1, -1):
                                if int(train_part[j].get('event', 0)) == 1:
                                    # move this subsequence
                                    item = train_part.pop(j)
                                    test_part.insert(0, item)
                                    moved = True
                                    break
                            # if no positive in train_part but positives exist earlier (edge case), try moving first positive from entire seqs
                            if not moved:
                                # look in full subseqs_sorted for a positive that currently lies in train range
                                for j, s in enumerate(subseqs_sorted):
                                    if int(s.get('event', 0)) == 1:
                                        # determine if it's currently assigned to train_part (index < split_idx)
                                        if j < len(per_parent_train[i]) + len(per_parent_test[i]):
                                            # remove from train_part if present
                                            try:
                                                per_parent_train[i].remove(s)
                                            except Exception:
                                                pass
                                            per_parent_test[i].insert(0, s)
                                            break

                # flatten per-parent lists
                train_list = [s for part in per_parent_train for s in part]
                test_list = [s for part in per_parent_test for s in part]

                # compute global event rates
                def rate(samples):
                    total = len(samples)
                    ev = sum(int(s.get('event', 0)) for s in samples)
                    return ev / max(1, total), ev, total

                train_rate, train_ev, train_total = rate(train_list)
                test_rate, test_ev, test_total = rate(test_list)
                diff = abs(train_rate - test_rate)
                self.logger.info(f'trial frac={frac:.3f} -> train {train_total} ({train_ev} ev, rate={train_rate:.3f}), test {test_total} ({test_ev} ev, rate={test_rate:.3f}), diff={diff:.4f}')
                if diff <= tol:
                    best = (frac, train_list, test_list, train_rate, test_rate, diff)
                    break
                if best is None or diff < best[5]:
                    best = (frac, train_list, test_list, train_rate, test_rate, diff)

            # choose best candidate
            chosen = best
            frac_chosen = chosen[0]
            self.train_samples = chosen[1]
            self.test_samples = chosen[2]
            self.train_is_subsequences = True
            self.logger.info(f'subsequence_temporal_per_parent 选用 frac={frac_chosen:.3f}, train={len(self.train_samples)}, test={len(self.test_samples)}, event_rate_diff={chosen[5]:.4f}')

        elif split_mode == 'group_balanced' and use_group:
            # 尝试在多个随机重试中找到 train/test，使得事件率差异在容忍范围内
            self.logger.info('split_mode=group_balanced: 在 group 级别尝试平衡事件率。')
            import re
            groups = []
            for s in self.all_merged_samples:
                rid = s.get('record_id')
                g = None
                try:
                    if isinstance(rid, dict):
                        g = rid.get('ar') or rid.get('raw') or None
                    if g is None and isinstance(rid, str):
                        m_ar_all = re.findall(r'ar\D*?(\d+)', rid, flags=re.IGNORECASE)
                        if m_ar_all:
                            g = f"ar{int(m_ar_all[-1])}"
                        else:
                            m_ar = re.search(r'\b([aA][rR]?\d+)\b', rid)
                            if m_ar:
                                g = m_ar.group(1).lower()
                    if g is None:
                        g = str(rid)
                except Exception:
                    g = str(rid)
                groups.append(str(g))

            from sklearn.model_selection import GroupShuffleSplit
            tol = float(getattr(self.config.data, 'group_balance_tolerance', 0.02))
            max_tries = int(getattr(self.config.data, 'group_balance_max_tries', 50))
            best = None
            import collections, random
            idx_all = list(range(len(self.all_merged_samples)))
            for attempt in range(max_tries):
                rs = self.config.data.random_seed + attempt
                gss = GroupShuffleSplit(n_splits=1, test_size=1.0 - self.config.data.train_test_split_ratio, random_state=rs)
                train_idx, test_idx = next(gss.split(idx_all, groups=groups))
                train_events = sum(1 for i in train_idx if self.all_merged_samples[i].get('event',0)==1)
                test_events = sum(1 for i in test_idx if self.all_merged_samples[i].get('event',0)==1)
                train_rate = train_events / max(1, len(train_idx))
                test_rate = test_events / max(1, len(test_idx))
                diff = abs(train_rate - test_rate)
                if diff <= tol:
                    best = (train_idx, test_idx, train_rate, test_rate, diff)
                    break
                # keep best so far
                if best is None or diff < best[4]:
                    best = (train_idx, test_idx, train_rate, test_rate, diff)

            if best is None:
                # fallback to simple group split (should not happen)
                self.logger.warning('group_balanced 未找到任何候选划分，回退为普通 group 划分')
                gss = GroupShuffleSplit(n_splits=1, test_size=1.0 - self.config.data.train_test_split_ratio, random_state=self.config.data.random_seed)
                train_idx, test_idx = next(gss.split(idx_all, groups=groups))
            else:
                train_idx, test_idx = best[0], best[1]
                self.logger.info(f'group_balanced 选用划分 (event rate diff={best[4]:.4f}, train_rate={best[2]:.4f}, test_rate={best[3]:.4f})')

            self.train_samples = [self.all_merged_samples[i] for i in train_idx]
            self.test_samples = [self.all_merged_samples[i] for i in test_idx]
            self.logger.info(f"按 group_balanced 划分完成: {len(self.train_samples)} 个训练样本, {len(self.test_samples)} 个测试样本。")

        else:
            # 默认或 'group' 模式：如果 use_group=False 则做按事件 stratified 划分，否则按 group 划分（遇错回退到 stratified）
            if not use_group:
                self.logger.info("use_group_split=False，使用按事件的 stratified train/test 划分以保持原始行为。")
                strata = [s['event'] for s in self.all_merged_samples]
                self.train_samples, self.test_samples = train_test_split(
                    self.all_merged_samples,
                    train_size=self.config.data.train_test_split_ratio,
                    random_state=self.config.data.random_seed,
                    stratify=strata
                )
                self.logger.info(f"划分完成: {len(self.train_samples)} 个训练样本, {len(self.test_samples)} 个测试样本。")
            else:
                self.logger.info("使用按 group 的默认划分（split_mode=group）。")
                try:
                    import re
                    groups = []
                    for s in self.all_merged_samples:
                        rid = s.get('record_id')
                        g = None
                        try:
                            if isinstance(rid, dict):
                                g = rid.get('ar') or rid.get('raw') or None
                            if g is None and isinstance(rid, str):
                                m_ar_all = re.findall(r'ar\D*?(\d+)', rid, flags=re.IGNORECASE)
                                if m_ar_all:
                                    g = f"ar{int(m_ar_all[-1])}"
                                else:
                                    m_ar = re.search(r'\b([aA][rR]?\d+)\b', rid)
                                    if m_ar:
                                        g = m_ar.group(1).lower()
                            if g is None:
                                g = str(rid)
                        except Exception:
                            g = str(rid)
                        groups.append(str(g))

                    from sklearn.model_selection import GroupShuffleSplit
                    gss = GroupShuffleSplit(n_splits=1, test_size=1.0 - self.config.data.train_test_split_ratio, random_state=self.config.data.random_seed)
                    idx_all = list(range(len(self.all_merged_samples)))
                    train_idx, test_idx = next(gss.split(idx_all, groups=groups))
                    self.train_samples = [self.all_merged_samples[i] for i in train_idx]
                    self.test_samples = [self.all_merged_samples[i] for i in test_idx]
                    self.logger.info(f"按 group (归一化record_id) 划分完成: {len(self.train_samples)} 个训练样本, {len(self.test_samples)} 个测试样本。")
                    # 诊断性统计：共有多少不同 group 被分配到每侧
                    try:
                        import collections
                        g_train = collections.Counter([groups[i] for i in train_idx])
                        g_test = collections.Counter([groups[i] for i in test_idx])
                        self.logger.info(f"Train groups: {len(g_train)}, Test groups: {len(g_test)}")
                    except Exception:
                        pass
                except Exception as e:
                    # 回退到按事件做 stratified split（原有行为）
                    self.logger.warning(f"Group-based split failed ({e}), 回退到 stratified train_test_split")
                    strata = [s['event'] for s in self.all_merged_samples]
                    self.train_samples, self.test_samples = train_test_split(
                        self.all_merged_samples,
                        train_size=self.config.data.train_test_split_ratio,
                        random_state=self.config.data.random_seed,
                        stratify=strata
                    )
                    self.logger.info(f"划分完成: {len(self.train_samples)} 个训练样本, {len(self.test_samples)} 个测试样本。")

        # 3. 特征选择和拟合转换器 (仅在训练集上)
        self._perform_feature_selection_and_fit_scalers()
        
        self.logger.info("数据预处理流水线执行完毕。")

    def _log_missing_values(self):
        """遍历所有样本并记录缺失值的统计信息。"""
        total_missing_count = 0
        columns_with_missing = set()
        for sample in self.all_merged_samples:
            if 'features' in sample and sample['features'] is not None:
                features_df = pd.DataFrame(sample['features'], columns=self.specified_features)
                missing_in_sample = features_df.isnull().sum().sum()
                if missing_in_sample > 0:
                    total_missing_count += missing_in_sample
                    columns_with_missing.update(features_df.columns[features_df.isnull().any()].tolist())

        self.logger.info(f"数据加载完成。在所有样本中发现总计 {total_missing_count} 个缺失值。")
        if columns_with_missing:
            self.logger.info(f"包含缺失值的列: {sorted(list(columns_with_missing))}")

    def _perform_feature_selection_and_fit_scalers(self):
        """在训练集上执行完整的特征选择和缩放器/变换器拟合流程。"""
        self.logger.info("准备用于特征选择的数据 (仅训练集)...")
        
        all_train_features_list = []
        all_train_labels_list = []
        for sample in self.train_samples:
            if 'features' in sample and sample['features'] is not None:
                features_df = pd.DataFrame(sample['features'], columns=self.specified_features)
                num_timesteps = len(features_df)
                
                # 为每个时间点创建对应的生存标签
                event = sample['event']
                duration = sample.get('duration_hours', 0) # 使用 .get() 避免  KeyError
                labels_df = pd.DataFrame({
                    'event': [event] * num_timesteps,
                    'duration': [duration] * num_timesteps
                })

                all_train_features_list.append(features_df)
                all_train_labels_list.append(labels_df)
        
        if not all_train_features_list:
            self.logger.error("训练样本中没有可用的特征数据。")
            return

        X_for_selection = pd.concat(all_train_features_list, ignore_index=True)
        y_for_selection = pd.concat(all_train_labels_list, ignore_index=True)

        # Pipeline: Outliers -> Log Transform -> Scaling -> Feature Selection
        X_clean = self.outlier_handler.fit_transform(X_for_selection)
        X_transformed = self.feature_transformer.fit_transform(X_clean)
        # flare特征log变换
        X_transformed = log_transform_flare_features(X_transformed, X_transformed.columns)

        # 自动剔除共线性和常数特征
        X_transformed_np = X_transformed.values
        feature_names = list(X_transformed.columns)
        X_transformed_np, feature_names = remove_collinear_and_constant_features(X_transformed_np, feature_names)
        X_transformed = pd.DataFrame(X_transformed_np, columns=feature_names)

        # --- 可选：基于训练/测试分布差异的特征过滤 ---
        try:
            fs_cfg = getattr(self.config.data, 'feature_shift', None)
        except Exception:
            fs_cfg = None
        if fs_cfg and bool(fs_cfg.get('enabled', False)):
            try:
                min_ts = int(fs_cfg.get('min_test_samples', 50))
                # 构建用于比较的样本集合（可选：使用 train 内部切出的 validation 而非全量 test，避免泄露）
                compare_target = str(fs_cfg.get('compare_to', 'val')).lower()
                all_test_features_list = []
                source_for_compare = None
                if compare_target == 'test':
                    source_for_compare = self.test_samples
                elif compare_target == 'val':
                    # 从训练父样本中切出一个验证子集（group split），仅用于分布比较，不改变 self.train_samples
                    try:
                        from sklearn.model_selection import GroupShuffleSplit
                        import re
                        groups = []
                        for s in self.train_samples:
                            rid = s.get('record_id')
                            g = None
                            try:
                                if isinstance(rid, dict):
                                    g = rid.get('ar') or rid.get('raw') or None
                                if g is None and isinstance(rid, str):
                                    m_ar_all = re.findall(r'ar\D*?(\d+)', rid, flags=re.IGNORECASE)
                                    if m_ar_all:
                                        g = f"ar{int(m_ar_all[-1])}"
                                    else:
                                        m_ar = re.search(r'\b([aA][rR]?\d+)\b', rid)
                                        if m_ar:
                                            g = m_ar.group(1).lower()
                                if g is None:
                                    g = str(rid)
                            except Exception:
                                g = str(rid)
                            groups.append(str(g))

                        idx_all = list(range(len(self.train_samples)))
                        val_frac = float(getattr(self.config.data, 'validation_ratio', 0.2))
                        if val_frac <= 0 or val_frac >= 1.0:
                            val_frac = 0.2
                        gss = GroupShuffleSplit(n_splits=1, test_size=val_frac, random_state=self.config.data.random_seed)
                        _, val_idx = next(gss.split(idx_all, groups=groups))
                        source_for_compare = [self.train_samples[i] for i in val_idx]
                    except Exception:
                        # fallback to using full test_samples if val split fails
                        source_for_compare = self.test_samples
                else:
                    # unknown mode: fallback to test
                    source_for_compare = self.test_samples

                for sample in (source_for_compare or []):
                    if 'features' in sample and sample['features'] is not None:
                        fdf = pd.DataFrame(sample['features'], columns=self.specified_features)
                        all_test_features_list.append(fdf)

                if len(all_test_features_list) >= 1 and sum(len(df) for df in all_test_features_list) >= min_ts:
                    X_test_for_selection = pd.concat(all_test_features_list, ignore_index=True)
                    # apply same outlier transform and feature transforms (use transform, not fit)
                    try:
                        X_test_clean = self.outlier_handler.transform(X_test_for_selection)
                    except Exception:
                        X_test_clean = X_test_for_selection.copy()
                    try:
                        X_test_transformed = self.feature_transformer.transform(X_test_clean)
                    except Exception:
                        X_test_transformed = X_test_clean.copy()
                    X_test_transformed = log_transform_flare_features(X_test_transformed, X_test_transformed.columns)

                    # Align columns: only compare上在 train 中还存在的列
                    common_cols = [c for c in X_transformed.columns if c in X_test_transformed.columns]
                    if common_cols:
                        repeats = int(fs_cfg.get('repeats', 1))
                        drop_fraction = float(fs_cfg.get('drop_fraction', 0.6))
                        action = str(fs_cfg.get('action', 'drop')).lower()
                        per_feature_scores = {c: [] for c in common_cols}

                        # 如果 repeats>1，则在 train_samples 上多次切分 val 并统计 score
                        metric = str(fs_cfg.get('metric', 'median_mad')).lower()
                        for rep in range(max(1, repeats)):
                            # For rep>0 when compare_to='val', re-split train->val with different seed
                            try:
                                if compare_target == 'val' and repeats > 1:
                                    from sklearn.model_selection import GroupShuffleSplit
                                    import random
                                    import re as _re
                                    groups = []
                                    for s in self.train_samples:
                                        rid = s.get('record_id')
                                        g = None
                                        try:
                                            if isinstance(rid, dict):
                                                g = rid.get('ar') or rid.get('raw') or None
                                            if g is None and isinstance(rid, str):
                                                m_ar_all = _re.findall(r'ar\D*?(\d+)', rid, flags=_re.IGNORECASE)
                                                if m_ar_all:
                                                    g = f"ar{int(m_ar_all[-1])}"
                                                else:
                                                    m_ar = _re.search(r'\b([aA][rR]?\d+)\b', rid)
                                                    if m_ar:
                                                        g = m_ar.group(1).lower()
                                            if g is None:
                                                g = str(rid)
                                        except Exception:
                                            g = str(rid)
                                        groups.append(str(g))

                                    idx_all = list(range(len(self.train_samples)))
                                    val_frac = float(getattr(self.config.data, 'validation_ratio', 0.2))
                                    gss = GroupShuffleSplit(n_splits=1, test_size=val_frac, random_state=self.config.data.random_seed + rep)
                                    _, val_idx = next(gss.split(idx_all, groups=groups))
                                    source_cmp = [self.train_samples[i] for i in val_idx]
                                else:
                                    source_cmp = source_for_compare

                                # build 2D df for this repetition
                                lst = []
                                for sample in (source_cmp or []):
                                    if 'features' in sample and sample['features'] is not None:
                                        lst.append(pd.DataFrame(sample['features'], columns=self.specified_features))
                                if not lst:
                                    continue
                                X_cmp = pd.concat(lst, ignore_index=True)
                                try:
                                    X_cmp = self.outlier_handler.transform(X_cmp)
                                except Exception:
                                    pass
                                try:
                                    X_cmp = self.feature_transformer.transform(X_cmp)
                                except Exception:
                                    pass
                                X_cmp = log_transform_flare_features(X_cmp, X_cmp.columns)

                                # prepare arrays per-column
                                train_vals_all = X_transformed[common_cols].values
                                cmp_vals_all = X_cmp[common_cols].values
                                eps = 1e-9
                                # two supported metrics: 'median_mad' (default) and 'ks'
                                if metric == 'ks':
                                    # try to use scipy if available for ks_2samp, otherwise approximate
                                    try:
                                        from scipy.stats import ks_2samp
                                        for i, col in enumerate(common_cols):
                                            a = train_vals_all[:, i]
                                            b = cmp_vals_all[:, i]
                                            try:
                                                ks = ks_2samp(a, b)
                                                score = float(ks.statistic)
                                            except Exception:
                                                # fallback approximate
                                                vals = np.sort(np.unique(np.concatenate([a, b])))
                                                cdf_a = np.searchsorted(np.sort(a), vals, side='right') / float(max(1, len(a)))
                                                cdf_b = np.searchsorted(np.sort(b), vals, side='right') / float(max(1, len(b)))
                                                score = float(np.max(np.abs(cdf_a - cdf_b)))
                                            per_feature_scores[col].append(score)
                                    except Exception:
                                        # scipy not available; approximate KS for each column
                                        for i, col in enumerate(common_cols):
                                            a = train_vals_all[:, i]
                                            b = cmp_vals_all[:, i]
                                            vals = np.sort(np.unique(np.concatenate([a, b])))
                                            cdf_a = np.searchsorted(np.sort(a), vals, side='right') / float(max(1, len(a)))
                                            cdf_b = np.searchsorted(np.sort(b), vals, side='right') / float(max(1, len(b)))
                                            score = float(np.max(np.abs(cdf_a - cdf_b)))
                                            per_feature_scores[col].append(score)
                                else:
                                    # default: median difference normalized by train MAD
                                    med_train = np.nanmedian(train_vals_all, axis=0)
                                    med_cmp = np.nanmedian(cmp_vals_all, axis=0)
                                    mad_train = np.nanmedian(np.abs(train_vals_all - med_train), axis=0)
                                    for i, col in enumerate(common_cols):
                                        denom = mad_train[i] + eps
                                        score = float(abs(med_train[i] - med_cmp[i]) / denom) if np.isfinite(denom) and denom > 0 else float('inf')
                                        per_feature_scores[col].append(score)
                            except Exception:
                                # proceed to next repetition
                                continue

                        # summarize per-feature across repetitions
                        drop_candidates = []
                        for col in common_cols:
                            scores = per_feature_scores.get(col, [])
                            if not scores:
                                continue
                            mean_score = float(np.nanmean(scores))
                            frac_above = float(sum(1 for s in scores if s > float(fs_cfg.get('threshold', 3.0))) / max(1, len(scores)))
                            # decide based on fraction
                            if frac_above >= float(drop_fraction):
                                # collect representative med/mad from full train vs last cmp
                                try:
                                    med_t = float(np.nanmedian(X_transformed[col].values))
                                except Exception:
                                    med_t = None
                                try:
                                    med_c = float(np.nanmedian(X_cmp[col].values))
                                except Exception:
                                    med_c = None
                                try:
                                    mad_t = float(np.nanmedian(np.abs(X_transformed[col].values - np.nanmedian(X_transformed[col].values))))
                                except Exception:
                                    mad_t = None
                                drop_candidates.append({'feature': col, 'mean_score': mean_score, 'frac_above': frac_above, 'med_train': med_t, 'med_cmp': med_c, 'mad_train': mad_t})

                        # 如果有需要处理的特征，执行处理并写入报告
                        if drop_candidates:
                            dropped = [d['feature'] for d in drop_candidates]
                            self.logger.info(f"feature_shift: {action} {len(dropped)} features due to large train/compare shift: {dropped}")
                            # apply action
                            if action == 'drop':
                                X_transformed.drop(columns=dropped, inplace=True, errors='ignore')
                                feature_names = [c for c in feature_names if c not in set(dropped)]
                            elif action == 'mask':
                                # set values in these columns to NaN so imputer will fill them
                                try:
                                    X_transformed.loc[:, dropped] = np.nan
                                except Exception:
                                    for c in dropped:
                                        X_transformed[c] = np.nan
                            elif action == 'winsorize' or action == 'winsorize':
                                # Clamp training values to configured quantiles per-feature
                                try:
                                    low_q = float(fs_cfg.get('winsorize_lower_pct', 0.01))
                                    high_q = float(fs_cfg.get('winsorize_upper_pct', 0.99))
                                except Exception:
                                    low_q, high_q = 0.01, 0.99
                                # create mapping from feature name to its report entry
                                feat_map = {d['feature']: d for d in drop_candidates}
                                for feat in dropped:
                                    try:
                                        vals = X_transformed[feat].values.astype(float)
                                        # compute quantiles ignoring NaNs
                                        qlow = float(np.nanquantile(vals, low_q)) if np.nanquantile is not None else float(np.percentile(vals[~np.isnan(vals)], low_q*100))
                                        qhigh = float(np.nanquantile(vals, high_q)) if np.nanquantile is not None else float(np.percentile(vals[~np.isnan(vals)], high_q*100))
                                        # apply clipping
                                        X_transformed[feat] = np.clip(X_transformed[feat].astype(float), qlow, qhigh)
                                        # record bounds in report entry if present
                                        if feat in feat_map:
                                            feat_map[feat]['winsor_lower'] = qlow
                                            feat_map[feat]['winsor_upper'] = qhigh
                                    except Exception:
                                        # if anything fails, fallback to masking this feature
                                        try:
                                            X_transformed.loc[:, feat] = np.nan
                                        except Exception:
                                            X_transformed[feat] = np.nan
                            else:
                                # unknown action: fallback to drop
                                X_transformed.drop(columns=dropped, inplace=True, errors='ignore')
                                feature_names = [c for c in feature_names if c not in set(dropped)]

                            # 将报告写入 preproc_reports
                            report_path = None
                            try:
                                report_dir = getattr(self.config, 'results_dir', 'results')
                                report_dir = os.path.join(report_dir, 'preproc_reports')
                                os.makedirs(report_dir, exist_ok=True)
                                report_out = {'features': drop_candidates, 'action': action, 'repeats': repeats}
                                report_path = os.path.join(report_dir, 'feature_shift_report.json')
                                with open(report_path, 'w') as _fo:
                                    json.dump(report_out, _fo)
                                self.logger.info(f"写入 feature_shift_report.json 到 {report_dir}")
                            except Exception:
                                pass

                            # 额外：在日志中输出简明摘要（brief）
                            try:
                                considered = len(common_cols)
                                flagged = len(dropped)
                                # 排名前3的示例（按 mean_score 降序）
                                top_features = sorted(drop_candidates, key=lambda x: x.get('mean_score', 0), reverse=True)[:3]
                                top_example = ', '.join([f"{t['feature']}({t.get('mean_score'):.2f}, frac={t.get('frac_above'):.2f})" for t in top_features]) if top_features else 'None'
                                brief = (f"feature_shift brief: compare_to={compare_target}, repeats={repeats}, action={action}, "
                                         f"considered={considered}, flagged={flagged}, top_examples=[{top_example}], report={report_path}")
                                self.logger.info(brief)
                            except Exception:
                                pass
                        else:
                            # 未检测到需处理的特征，输出简明日志
                            try:
                                considered = len(common_cols)
                                brief = (f"feature_shift brief: compare_to={compare_target}, repeats={int(fs_cfg.get('repeats',1))}, "
                                         f"action={str(fs_cfg.get('action','drop'))}, considered={considered}, flagged=0, report=None")
                                self.logger.info(brief)
                            except Exception:
                                pass
                else:
                    self.logger.info('feature_shift: 测试样本过少，跳过分布差异检测。')
            except Exception as e:
                self.logger.warning(f'feature_shift: 检测失败，跳过该步骤: {e}')
        
        # 使用RobustScaler
        scaler_for_selection = RobustScaler()
        X_scaled = pd.DataFrame(scaler_for_selection.fit_transform(X_transformed), columns=X_transformed.columns)

        # --- 特征选择 ---
        if self.config.data.feature_selection.enabled:
            self.logger.info("开始特征选择...")
            selector = feature_selector.FeatureSelector(
                n_features=self.config.data.feature_selection.n_features,
                vif_threshold=self.config.data.feature_selection.vif_threshold
            )
            self.active_feature_names = selector.select_features(X_scaled, y_for_selection)
            if not self.active_feature_names:
                self.logger.warning("特征选择未返回任何特征，将使用所有指定的特征作为后备。")
                self.active_feature_names = feature_names
        else:
            self.active_feature_names = feature_names
            self.logger.info("特征选择被禁用，使用所有指定的特征。")
        
        self.logger.info(f"激活的特征: {self.active_feature_names}")

        # 在完整训练数据上拟合最终的缩放器
        self.logger.info("在完整训练数据上拟合最终的缩放器...")
        
        # 关键修复: 在与评估时相同形状的数据上拟合缩放器
        # 如果 train_samples 已经是子序列列表（在 subsequence_independent 模式下），
        # 则直接从这些子序列构建 3D 数组，而不是再次尝试作为父样本生成子序列。
        X_train_unscaled_3d = None
        try:
            first_train = self.train_samples[0]
            if isinstance(first_train, dict) and 'features' in first_train and not isinstance(first_train['features'], pd.DataFrame):
                # assume features are ndarray
                feature_list = [s['features'] for s in self.train_samples]
                X_train_unscaled_3d = np.array(feature_list, dtype=np.float32)
        except Exception:
            X_train_unscaled_3d = None

        if X_train_unscaled_3d is None:
            X_train_unscaled_3d, _, _ = self._generate_and_shape_features(self.train_samples)
        
        if X_train_unscaled_3d is None:
            raise ValueError("无法从训练样本生成子序列，无法拟合缩放器。")
        
        # 为了拟合缩放器，需要将数据暂时变为2D
        n_samples, n_timesteps, n_features = X_train_unscaled_3d.shape
        X_train_unscaled_2d = X_train_unscaled_3d.reshape(-1, n_features)
        # 如果子序列生成器附加了序列级聚合特征，n_features 可能大于 self.active_feature_names
        # 自动扩展列名以匹配附加的聚合特征
        # 新版（7x）：每个基础特征被扩展为 [orig, last, mean, var, std, max, min]
        # 旧版（5x）：每个基础特征被扩展为 [orig, last, mean, std, slope]
        if n_features != len(self.active_feature_names):
            base_feats = list(self.active_feature_names)
            k = len(base_feats)
            if k > 0 and n_features == k * 7:
                # 新版：orig + 6 个聚合
                extra_names = []
                for f in base_feats:
                    extra_names.extend([f + '_last', f + '_mean', f + '_var', f + '_std', f + '_max', f + '_min'])
                cols_for_scaler = base_feats + extra_names
                self.logger.info("Detected 7x aggregated features (last/mean/var/std/max/min); expanded columns for scaler fitting.")
            elif k > 0 and n_features == k * 5:
                # 兼容旧版：orig + 4 个聚合（last/mean/std/slope）
                extra_names = []
                for f in base_feats:
                    extra_names.extend([f + '_last', f + '_mean', f + '_std', f + '_slope'])
                cols_for_scaler = base_feats + extra_names
                self.logger.info("Detected legacy 5x aggregated features (last/mean/std/slope); expanded columns for scaler fitting.")
            else:
                # 若不符合预期，生成通用额外列名以避免形状不匹配错误
                extra_count = n_features - len(self.active_feature_names)
                extra_names = [f'extra_feat_{i}' for i in range(extra_count)]
                cols_for_scaler = list(self.active_feature_names) + extra_names
                self.logger.warning(f"Feature count mismatch when fitting scaler: expected {len(self.active_feature_names)}, got {n_features}. Appending {extra_count} generic names for scaling.")
        else:
            cols_for_scaler = self.active_feature_names

        # 持久化用于缩放器的列名，后续 transform 时应优先使用这一列表以保证一致性
        try:
            self.scaler_feature_columns = list(cols_for_scaler)
        except Exception:
            self.scaler_feature_columns = None

        # flare特征log变换
        X_train_unscaled_2d_df = pd.DataFrame(X_train_unscaled_2d, columns=cols_for_scaler)
        X_train_unscaled_2d_df = log_transform_flare_features(X_train_unscaled_2d_df, self.active_feature_names)
        # 清理数值（去除 inf / 极端值并填充 NaN）以避免 scaler 出错
        X_train_unscaled_2d_df = sanitize_numeric_df(X_train_unscaled_2d_df)
        # --- 新增：处理极端 sentinel/Inf -> 转为 NaN，并保存列级报表 ---
        try:
            SENTINEL = getattr(self.config.data, 'sentinel_threshold', 1e6)
        except Exception:
            SENTINEL = 1e6

        # 将 ±inf 替换为 NaN
        X_train_unscaled_2d_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        # 将明显的 sentinel/极端值标记为 NaN
        try:
            extreme_mask = (X_train_unscaled_2d_df.abs() >= float(SENTINEL))
            if extreme_mask.any().any():
                X_train_unscaled_2d_df[extreme_mask] = np.nan
        except Exception:
            pass
        # 若存在全为 NaN 的列，用 0 进行占位填充，避免下游 nan 函数告警
        try:
            all_nan_cols = X_train_unscaled_2d_df.columns[X_train_unscaled_2d_df.isna().all()].tolist()
            if all_nan_cols:
                X_train_unscaled_2d_df[all_nan_cols] = 0.0
                self.logger.info(f"Detected {len(all_nan_cols)} all-NaN columns during scaler fitting; filled with 0.0 as placeholders.")
        except Exception:
            pass

        # 列级报表
        try:
            report_dir = getattr(self.config, 'results_dir', 'results')
            report_dir = os.path.join(report_dir, 'preproc_reports')
            os.makedirs(report_dir, exist_ok=True)
            nan_counts = X_train_unscaled_2d_df.isna().sum()
            extreme_counts = (X_train_unscaled_2d_df.abs() >= float(SENTINEL)).sum()
            report_df = pd.DataFrame({'nan_count': nan_counts, 'extreme_count': extreme_counts})
            report_path = os.path.join(report_dir, 'scaler_input_column_report.csv')
            report_df.to_csv(report_path)
            self.logger.info(f'写入预处理列级报告: {report_path}')
            # Persist scaler_feature_columns if available
            try:
                if getattr(self, 'scaler_feature_columns', None) is not None:
                    import json as _json
                    cols_path = os.path.join(report_dir, 'scaler_feature_columns.json')
                    with open(cols_path, 'w') as _f:
                        _json.dump(self.scaler_feature_columns, _f)
                    self.logger.info(f'写入 scaler 列名到: {cols_path}')
            except Exception as _e:
                self.logger.warning(f'无法写入 scaler_feature_columns: {_e}')
        except Exception as e:
            self.logger.warning(f'无法写入列级报告: {e}')

        # 使用 SimpleImputer(strategy='median') 填充后再拟合 RobustScaler
        try:
            self.imputer = SimpleImputer(strategy='median')
            X_imputed = self.imputer.fit_transform(X_train_unscaled_2d_df.values)
        except Exception as e:
            # 回退：使用 sanitize_numeric_df 的填充逻辑
            self.logger.warning(f'Imputer 失败，回退到 sanitize_numeric_df: {e}')
            X_train_unscaled_2d_df = sanitize_numeric_df(X_train_unscaled_2d_df)
            X_imputed = X_train_unscaled_2d_df.values

        # 使用RobustScaler
        self.scaler = RobustScaler()
        self.scaler.fit(X_imputed)

        # 记录标准化统计信息
        try:
            self.logger.info(f"RobustScaler统计 - 中位数: {self.scaler.center_[:5]}..., IQR: {self.scaler.scale_[:5]}...")
        except Exception:
            self.logger.info("RobustScaler拟合完成（无法打印部分统计信息）。")
        self.logger.info("最终的RobustScaler拟合完成。")
        # Persist scaler stats for debugging / reproducibility
        try:
            report_dir = getattr(self.config, 'results_dir', 'results')
            report_dir = os.path.join(report_dir, 'preproc_reports')
            os.makedirs(report_dir, exist_ok=True)
            import json as _json
            stats = {
                'center': None,
                'scale': None
            }
            try:
                stats['center'] = self.scaler.center_.tolist()
                stats['scale'] = self.scaler.scale_.tolist()
            except Exception:
                pass
            with open(os.path.join(report_dir, 'scaler_stats.json'), 'w') as _sf:
                _json.dump(stats, _sf)
            self.logger.info(f'写入 scaler_stats.json 到 {report_dir}')
        except Exception:
            pass
        # --- 新增: 写入 preproc manifest，用于后续评估的严格校验和审计 ---
        try:
            manifest = {}
            import time as _time, json as _json
            manifest['timestamp'] = float(_time.time())
            try:
                # number of features scaler was fit on
                manifest['scaler_feature_columns_count'] = len(self.scaler_feature_columns) if getattr(self, 'scaler_feature_columns', None) is not None else None
            except Exception:
                manifest['scaler_feature_columns_count'] = None
            try:
                manifest['scaler_feature_columns_preview'] = list(self.scaler_feature_columns)[:50] if getattr(self, 'scaler_feature_columns', None) is not None else None
            except Exception:
                manifest['scaler_feature_columns_preview'] = None
            try:
                manifest['n_features_in_fit'] = int(n_features)
            except Exception:
                manifest['n_features_in_fit'] = None
            try:
                seq_cfg = getattr(self.config.data, 'sequence_generation', {})
                manifest['include_aggregated_features'] = bool(seq_cfg.get('include_aggregated_features', False)) if seq_cfg is not None else None
            except Exception:
                manifest['include_aggregated_features'] = None
            try:
                # scaler stats summary
                manifest['scaler_stats_summary'] = {
                    'center_len': len(stats.get('center')) if stats.get('center') is not None else None,
                    'scale_len': len(stats.get('scale')) if stats.get('scale') is not None else None
                }
            except Exception:
                manifest['scaler_stats_summary'] = None
            manifest_path = os.path.join(report_dir, 'preproc_manifest.json')
            with open(manifest_path, 'w') as _mf:
                _json.dump(manifest, _mf)
            self.logger.info(f'写入 preproc_manifest.json 到 {manifest_path}')
        except Exception:
            pass
        # 持久化 scaler 与 imputer（joblib），便于评估和复现
        try:
            import joblib
            scaler_path = os.path.join(report_dir, 'scaler.joblib')
            joblib.dump(self.scaler, scaler_path)
            self.logger.info(f'已持久化 scaler 到 {scaler_path}')
            imputer_path = os.path.join(report_dir, 'imputer.joblib')
            try:
                joblib.dump(self.imputer, imputer_path)
                self.logger.info(f'已持久化 imputer 到 {imputer_path}')
            except Exception:
                # imputer 可选，失败不致命
                pass
        except Exception:
            pass
        self.logger.info("数据预处理流水线执行完毕。")

    def get_train_test_split(self):
        """返回顶层的训练/测试父样本集。"""
        return self.train_samples, self.test_samples

    def get_cross_val_split(self, split_indices):
        """
        根据交叉验证提供的索引，返回对应的训练/验证父样本集。
        """
        train_parent_samples = [self.train_samples[i] for i in split_indices[0]]
        val_parent_samples = [self.train_samples[i] for i in split_indices[1]]
        return train_parent_samples, val_parent_samples

    def _generate_and_shape_features(self, source_samples):
        """内部辅助函数：仅生成和重塑特征，不进行缩放。"""
        all_subsequences = []
        seq_gen_config = self.config.data.sequence_generation
        for sample in source_samples:
            # 标准化时间戳字段：接受多种可能的源列名并规范为 'timestamps_list'
            if 'timestamps_list' not in sample:
                # 支持原始数据中使用 'Timestamp'（单列）或 'timestamps' 等字段
                if 'Timestamp' in sample:
                    try:
                        sample['timestamps_list'] = sample['Timestamp']
                    except Exception:
                        sample['timestamps_list'] = sample.get('timestamps') or sample.get('timestamps_list')
                elif 'timestamps' in sample:
                    sample['timestamps_list'] = sample['timestamps']
                elif 'time_index' in sample:
                    sample['timestamps_list'] = sample['time_index']
                else:
                    # last-resort: try to find any key that looks like timestamp series
                    for k, v in list(sample.items()):
                        if k.lower() in ('timestamp', 'timestamps', 'time', 'time_index'):
                            sample['timestamps_list'] = v
                            break
            # 将序列生成相关配置注入到 sample 的副本里，以便 subsequence_generator 可以选择是否附加聚合特征
            sample_with_cfg = sample.copy()
            sample_with_cfg['config'] = {'sequence_generation': {'include_aggregated_features': bool(seq_gen_config.get('include_aggregated_features', False))}}
            subsequences = subsequence_generator.generate_subsequences(
                sample_with_cfg,
                seq_gen_config['num_covariate_timesteps'],
                seq_gen_config['prediction_window_hours'],
                seq_gen_config['sub_sequence_step'],
                self.active_feature_names
            )
            all_subsequences.extend(subsequences)

        if not all_subsequences:
            self.logger.warning("未能从源样本生成任何有效的子序列。")
            return None, None, None

        feature_list = [s['features'] for s in all_subsequences]
        # Diagnostic: 检查所有子序列的列宽是否一致
        try:
            widths = [f.shape[1] for f in feature_list]
            unique_widths = sorted(set(widths))
            if len(unique_widths) > 1:
                self.logger.warning(f"Detected heterogeneous subsequence feature widths: {unique_widths}. Showing sample counts per width.")
                from collections import Counter
                cnt = Counter(widths)
                self.logger.warning(f"Per-width counts: {dict(cnt)}")
                # 打印最多两个示例用以诊断
                examples = {}
                for w in unique_widths[:2]:
                    for idx, f in enumerate(feature_list):
                        if f.shape[1] == w:
                            examples[w] = {
                                'index': idx,
                                'record_id': all_subsequences[idx].get('record_id')
                            }
                            break
                self.logger.warning(f"Width examples: {examples}")
        except Exception:
            pass
        labels_list = [(s['duration'], s['event']) for s in all_subsequences]
        # Ensure durations are canonical hours for downstream consumers
        try:
            from evaluation.plotting import _ensure_durations_in_hours
            durations_raw = np.array([l[0] for l in labels_list], dtype=float)
            durations_hours = _ensure_durations_in_hours(durations_raw, cfg=self.config, name='preprocessor_labels')
            # rebuild labels_list with converted durations while preserving events
            labels_list = [(float(durations_hours[i]), int(labels_list[i][1])) for i in range(len(labels_list))]
        except Exception:
            # fallback: keep original labels_list
            pass
        record_id_list = [s.get('record_id', None) for s in all_subsequences]
        
        X_3d = np.array(feature_list, dtype=np.float32)
        y = np.array(labels_list, dtype=np.float32)
        record_ids = np.array(record_id_list)

        # 此函数现在总是返回3D数据，塑形将在get_evaluation_subsequences中进行
        return X_3d, y, record_ids

    def get_evaluation_subsequences(self, source_samples):
        """从给定的样本集生成用于评估的子序列（包括缩放）。"""
        self.logger.info(f"正从 {len(source_samples)} 个指定的源样本生成子序列...")

        # 如果 fit 时持久化了 scaler_feature_columns，但当前对象没有，尝试从磁盘加载
        if getattr(self, 'scaler_feature_columns', None) is None:
            try:
                report_dir = getattr(self.config, 'results_dir', 'results')
                cols_path = os.path.join(report_dir, 'preproc_reports', 'scaler_feature_columns.json')
                if os.path.exists(cols_path):
                    import json as _json
                    with open(cols_path, 'r') as _f:
                        self.scaler_feature_columns = _json.load(_f)
                    self.logger.info(f'Loaded persisted scaler_feature_columns from {cols_path}')
            except Exception:
                pass

        # 如果 source_samples 已经是子序列列表（subsequence_independent 情况），则直接使用而不重新生成
        X_unscaled_3d = None
        try:
            first = source_samples[0]
            if isinstance(first, dict) and 'features' in first and not isinstance(first['features'], pd.DataFrame):
                # assume features is ndarray
                self.logger.info('检测到 source_samples 为已生成的子序列列表，直接使用而不重新生成。')
                feature_list = [s['features'] for s in source_samples]
                labels_list = [(s.get('duration', 0.0), s.get('event', 0)) for s in source_samples]
                record_id_list = [s.get('record_id', None) for s in source_samples]
                X_unscaled_3d = np.array(feature_list, dtype=np.float32)
                y = np.array(labels_list, dtype=np.float32)
                record_ids = np.array(record_id_list)
        except Exception:
            X_unscaled_3d = None

        if X_unscaled_3d is None:
            X_unscaled_3d, y, record_ids = self._generate_and_shape_features(source_samples)

        if X_unscaled_3d is None:
            return None, None, None
        
        # 对最终的特征矩阵应用缩放器
        # 1. 保存原始3D形状
        n_samples, n_timesteps, n_features = X_unscaled_3d.shape
        # 在变形为2D之前，若已有已拟合的 scaler，优先让特征数与其期望一致
        try:
            expected_by_scaler = getattr(self.scaler, 'n_features_in_', None)
        except Exception:
            expected_by_scaler = None
        if expected_by_scaler is not None and expected_by_scaler != n_features:
            self.logger.info(f"Aligning generated features to scaler.n_features_in_={expected_by_scaler} from {n_features}.")
            if expected_by_scaler < n_features:
                # 截断多余的列（通常为未在拟合时包含的聚合附加列）
                X_unscaled_3d = X_unscaled_3d[:, :, :expected_by_scaler]
            else:
                # 若生成的列少于 scaler 期望，则在末尾用零填充
                pad = expected_by_scaler - n_features
                X_unscaled_3d = np.pad(X_unscaled_3d, ((0,0),(0,0),(0,pad)), mode='constant', constant_values=0.0)
            n_features = expected_by_scaler
        # 2. 变形为2D以进行缩放
        X_unscaled_2d = X_unscaled_3d.reshape(-1, n_features)
        # flare特征log变换
        # 优先使用拟合时保存的列名列表以保证一致性；若不存在则按原有逻辑推断
        if getattr(self, 'scaler_feature_columns', None) is not None:
            cols = list(self.scaler_feature_columns)
            # 如果实际生成的 n_features 与拟合时列数不一致，记录并尝试容错地进行修复，而不是直接抛出异常
            if len(cols) != n_features:
                self.logger.warning(
                    f"Fitted scaler columns ({len(cols)}) != generated n_features ({n_features}). Attempting to reconcile automatically."
                )
                # 尝试使用 active_feature_names 重建聚合列名（如果可能）
                try:
                    k = len(self.active_feature_names) if getattr(self, 'active_feature_names', None) is not None else 0
                except Exception:
                    k = 0

                reconciled = None
                if k > 0 and n_features == k * 7:
                    # 新版：每个基础特征被扩展为 6 个聚合（last, mean, var, std, max, min）
                    extra_names = []
                    for f in self.active_feature_names:
                        extra_names.extend([f + '_last', f + '_mean', f + '_var', f + '_std', f + '_max', f + '_min'])
                    reconciled = list(self.active_feature_names) + extra_names
                    self.logger.info("Reconstructed aggregated feature names assuming 7x expansion to match n_features.")
                elif k > 0 and n_features == k * 5:
                    # 旧版：每个基础特征被扩展为 4 个聚合（last, mean, std, slope）
                    extra_names = []
                    for f in self.active_feature_names:
                        extra_names.extend([f + '_last', f + '_mean', f + '_std', f + '_slope'])
                    reconciled = list(self.active_feature_names) + extra_names
                    self.logger.info("Reconstructed aggregated feature names assuming legacy 5x expansion to match n_features.")
                else:
                    # 如果目标列数大于已保存列数，追加通用占位列名；若小于则截断
                    if n_features > len(cols):
                        extra_count = n_features - len(cols)
                        reconciled = cols + [f'extra_feat_{i}' for i in range(extra_count)]
                        self.logger.info(f"Appended {extra_count} generic extra feature names to saved scaler columns to match n_features.")
                    else:
                        reconciled = cols[:n_features]
                        self.logger.info(f"Truncated saved scaler_feature_columns from {len(cols)} to match n_features={n_features}.")

                # 将修复后的列名回写到实例，以便后续处理使用一致的列名
                cols = reconciled
                try:
                    self.scaler_feature_columns = list(cols)
                except Exception:
                    pass
        else:
            # 和拟合缩放器时一样，容错地扩展 active_feature_names 以匹配 n_features
            if n_features != len(self.active_feature_names):
                base_feats = list(self.active_feature_names)
                k = len(base_feats)
                if k > 0 and n_features == k * 7:
                    extra_names = []
                    for f in base_feats:
                        extra_names.extend([f + '_last', f + '_mean', f + '_var', f + '_std', f + '_max', f + '_min'])
                    cols = base_feats + extra_names
                elif k > 0 and n_features == k * 5:
                    extra_names = []
                    for f in base_feats:
                        extra_names.extend([f + '_last', f + '_mean', f + '_std', f + '_slope'])
                    cols = base_feats + extra_names
                else:
                    extra_count = n_features - len(self.active_feature_names)
                    cols = list(self.active_feature_names) + [f'extra_feat_{i}' for i in range(extra_count)]
                cols = cols
            else:
                cols = self.active_feature_names

        # 确保列名与当前特征数一致
        if len(cols) != n_features:
            # 这里直接将 n_features 与列名数量对齐，避免后续 reshape 告警
            if len(cols) > n_features:
                cols = cols[:n_features]
            else:
                cols = cols + [f'extra_feat_{i}' for i in range(n_features - len(cols))]
        X_unscaled_2d_df = pd.DataFrame(X_unscaled_2d, columns=cols)
        # 对于log变换，使用 cols（可能包含扩展的聚合名）进行判断
        X_unscaled_2d_df = log_transform_flare_features(X_unscaled_2d_df, cols)
        # 清理数值并保证没有 Inf/NaN
        X_unscaled_2d_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        try:
            # 将明显的 sentinel 标记为 NaN（与 fit 时一致）
            SENTINEL = getattr(self.config.data, 'sentinel_threshold', 1e6)
            extreme_mask = (X_unscaled_2d_df.abs() >= float(SENTINEL))
            if extreme_mask.any().any():
                X_unscaled_2d_df[extreme_mask] = np.nan
        except Exception:
            pass
        # 与拟合阶段一致：若出现全为 NaN 的列，先占位填 0，避免下游告警
        try:
            all_nan_cols = X_unscaled_2d_df.columns[X_unscaled_2d_df.isna().all()].tolist()
            if all_nan_cols:
                X_unscaled_2d_df[all_nan_cols] = 0.0
                self.logger.debug(f"During evaluation transform, filled {len(all_nan_cols)} all-NaN columns with 0.0 placeholders.")
        except Exception:
            pass

        # 首选使用 fit 时训练好的 imputer
        if getattr(self, 'imputer', None) is not None:
            try:
                # Ensure pre-cleaning: replace Inf and clip extreme values BEFORE imputation
                try:
                    SENTINEL = float(getattr(self.config.data, 'sentinel_threshold', 1e6))
                except Exception:
                    SENTINEL = 1e6
                X_unscaled_2d_df.replace([np.inf, -np.inf], np.nan, inplace=True)
                try:
                    X_unscaled_2d_df = X_unscaled_2d_df.clip(lower=-SENTINEL, upper=SENTINEL)
                except Exception:
                    pass
                X_imputed = self.imputer.transform(X_unscaled_2d_df.values)
            except Exception:
                X_imputed = sanitize_numeric_df(X_unscaled_2d_df).values
        else:
            # perform cleaning even without imputer
            try:
                SENTINEL = float(getattr(self.config.data, 'sentinel_threshold', 1e6))
            except Exception:
                SENTINEL = 1e6
            X_unscaled_2d_df.replace([np.inf, -np.inf], np.nan, inplace=True)
            try:
                X_unscaled_2d_df = X_unscaled_2d_df.clip(lower=-SENTINEL, upper=SENTINEL)
            except Exception:
                pass
            X_imputed = sanitize_numeric_df(X_unscaled_2d_df).values

        # Try transform; if fails, write detailed problematic rows and re-raise
        try:
            X_scaled_2d = self.scaler.transform(X_imputed)
            # 数值稳定性：将 transform 后的非有限值清理为 0，并裁剪极端值
            try:
                SENTINEL = float(getattr(self.config.data, 'sentinel_threshold', 1e6))
            except Exception:
                SENTINEL = 1e6
            # 用 0 替换 NaN/Inf
            X_scaled_2d = np.nan_to_num(X_scaled_2d, nan=0.0, posinf=0.0, neginf=0.0)
            # 再做一次合理范围裁剪
            try:
                X_scaled_2d = np.clip(X_scaled_2d, -SENTINEL, SENTINEL)
            except Exception:
                pass
        except Exception as e:
            self.logger.warning(f"scaler.transform failed: {e}; attempting to record problematic subsequences and abort.")
            try:
                report_dir = getattr(self.config, 'results_dir', 'results')
                os.makedirs(os.path.join(report_dir, 'preproc_reports'), exist_ok=True)
                orig = X_unscaled_2d
                # Identify problematic rows: Inf or exceeding sentinel
                mask_inf = np.isinf(orig).any(axis=1)
                try:
                    SENTINEL = float(getattr(self.config.data, 'sentinel_threshold', 1e6))
                except Exception:
                    SENTINEL = 1e6
                mask_extreme = (np.abs(orig) >= SENTINEL).any(axis=1)
                problematic_rows = np.where(mask_inf | mask_extreme)[0]
                # Limit number of reported rows to avoid huge files
                max_report = 50
                problematic_rows = problematic_rows[:max_report]
                detailed = []
                import json as _json, csv
                for r in problematic_rows:
                    rec_idx = int(r // n_timesteps)
                    try:
                        rec_id = record_ids[rec_idx]
                    except Exception:
                        rec_id = None
                    row_vals = orig[r, :]
                    num_infs = int(np.isinf(row_vals).sum())
                    row_max = float(np.nanmax(np.abs(row_vals))) if row_vals.size else None
                    # sample up to first 10 values for inspection
                    sample_vals = [float(x) if np.isfinite(x) else None for x in row_vals[:10]]
                    detailed.append({'row': int(r), 'record_index': rec_idx, 'record_id': rec_id, 'num_infs': num_infs, 'row_max_abs': row_max, 'sample_values_first10': sample_vals})

                bad_path = os.path.join(report_dir, 'preproc_reports', 'problematic_subsequences_detailed.json')
                with open(bad_path, 'w') as _f:
                    _json.dump(detailed, _f)
                self.logger.error(f'Wrote detailed problematic subsequences to {bad_path} (count={len(detailed)})')
                # Also write a compact CSV
                csv_path = os.path.join(report_dir, 'preproc_reports', 'problematic_subsequences.csv')
                with open(csv_path, 'w', newline='') as _cf:
                    writer = csv.DictWriter(_cf, fieldnames=['row','record_index','record_id','num_infs','row_max_abs'])
                    writer.writeheader()
                    for it in detailed:
                        writer.writerow({k: it[k] for k in ['row','record_index','record_id','num_infs','row_max_abs']})
                self.logger.error(f'Wrote compact problematic subsequences CSV to {csv_path}')
            except Exception as _ee:
                self.logger.error(f'Failed to write problematic subsequences report: {_ee}')
            # re-raise original exception
            raise
        # 4. 恢复为3D形状 —— 增加防护性检查，避免因行/列不匹配导致的难以诊断的reshape错误
        rows, cols = X_scaled_2d.shape
        expected_rows = n_samples * n_timesteps
        # 记录中间形状以便诊断
        self.logger.debug(f"Reshape diagnostics: X_unscaled_3d.shape={X_unscaled_3d.shape}, X_unscaled_2d.shape={X_unscaled_2d.shape}, X_unscaled_2d_df.shape={(X_unscaled_2d_df.shape if 'X_unscaled_2d_df' in locals() else 'NA')}, X_imputed.shape={X_imputed.shape}, X_scaled_2d.shape={X_scaled_2d.shape}")
        if cols != n_features:
            # 不立即抛出，尝试容错处理：调整 n_features 以匹配 scaler 的输出列数
            self.logger.warning(f"Feature dimension mismatch during reshape: scaler output has {cols} cols but expected n_features={n_features}. Attempting to reconcile.")
            # 如果可用，基于已持久化的 scaler_feature_columns 调整 active_feature_names
            try:
                saved_cols = getattr(self, 'scaler_feature_columns', None)
                if saved_cols is not None:
                    # 如果 saved_cols 长度 != cols，优先信任实际 scaler 输出列数
                    if len(saved_cols) != cols:
                        self.logger.info(f"Saved scaler_feature_columns length ({len(saved_cols)}) != actual scaler output cols ({cols}). Truncating/expanding saved list to match actual columns.")
                        if len(saved_cols) < cols:
                            saved_cols = saved_cols + [f'extra_feat_{i}' for i in range(cols - len(saved_cols))]
                        else:
                            saved_cols = saved_cols[:cols]
                    # 更新实例字段
                    self.scaler_feature_columns = saved_cols
                    # 尝试更新 active_feature_names if possible (strip aggregated suffixes)
                    try:
                        base = [c for c in saved_cols if not any(suffix in c for suffix in ['_last','_mean','_var','_std','_max','_min','_slope','extra_feat_'])]
                        if base:
                            self.active_feature_names = base
                    except Exception:
                        pass
            except Exception:
                pass
            # 让 n_features 与 scaler 输出列数一致以继续reshape
            n_features = cols
        if rows != expected_rows:
            # 如果行数可以被 n_timesteps 整除，则尝试推断出实际样本数并调整
            if n_timesteps > 0 and rows % n_timesteps == 0:
                inferred_n_samples = rows // n_timesteps
                self.logger.warning(f"Scaled rows ({rows}) != expected ({expected_rows}). Inferred n_samples={inferred_n_samples} (was {n_samples}). Adjusting n_samples for reshape. Please verify subsequence generator consistency.")
                n_samples = inferred_n_samples
            else:
                raise ValueError(f"Cannot reshape scaler output of size {rows}x{cols} into ({n_samples},{n_timesteps},{n_features}). Expected rows={expected_rows}.")
        X_scaled_3d = X_scaled_2d.reshape(n_samples, n_timesteps, n_features)

        self.logger.info(f"生成了 {X_scaled_3d.shape[0]} 个评估子序列。特征维度: {X_scaled_3d.shape}, 标签维度: {y.shape}")
        
        return X_scaled_3d, y, record_ids

    def save_preprocessor(self, path):
        """保存预处理器状态到指定目录：scaler 与 active_feature_names。"""
        os.makedirs(path, exist_ok=True)
        try:
            import joblib
            scaler_path = os.path.join(path, 'scaler.joblib')
            joblib.dump(self.scaler, scaler_path)
            features_path = os.path.join(path, 'active_features.json')
            with open(features_path, 'w') as f:
                json.dump(self.active_feature_names, f)
            self.logger.info(f"Preprocessor saved: scaler -> {scaler_path}, features -> {features_path}")
        except Exception as e:
            self.logger.warning(f"Failed to save preprocessor: {e}")

def remove_collinear_and_constant_features(X, feature_names, threshold=0.98):
    logger = logging.getLogger(__name__)
    X_df = pd.DataFrame(X.reshape(-1, X.shape[-1]), columns=feature_names)
    # 1. 剔除常数特征
    nunique = X_df.nunique()
    constant_features = nunique[nunique == 1].index.tolist()
    if constant_features:
        logger.info(f"Remove constant features: {constant_features}")
        X_df = X_df.drop(columns=constant_features)
    # 2. 剔除高度共线性特征
    corr_matrix = X_df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = set()
    for col in upper.columns:
        for row in upper.index:
            if upper.loc[row, col] > threshold:
                # 剔除方差较小的特征
                var_row = X_df[row].var()
                var_col = X_df[col].var()
                drop = row if var_row < var_col else col
                to_drop.add(drop)
    if to_drop:
        logger.info(f"Remove collinear features (corr>{threshold}): {list(to_drop)}")
        X_df = X_df.drop(columns=list(to_drop))
    # 返回剔除后的数据和特征名
    X_new = X_df.values.reshape(X.shape[0], X.shape[1], -1) if X.ndim == 3 else X_df.values
    new_feature_names = X_df.columns.tolist()
    return X_new, new_feature_names
