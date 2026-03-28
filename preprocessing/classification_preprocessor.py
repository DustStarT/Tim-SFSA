"""
分类任务数据预处理器
为LSTM分类模型准备数据，将生存分析问题转换为二分类问题
"""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import logging
from typing import Tuple, List, Dict, Any, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class ClassificationDataset(Dataset):
    """
    分类任务数据集
    """
    
    def __init__(self, X: np.ndarray, y: np.ndarray, feature_names: List[str] = None):
        """
        初始化数据集
        
        Args:
            X: 特征数据，形状为 (n_samples, seq_len, n_features)
            y: 标签数据，形状为 (n_samples,) - 二分类标签 (0/1)
            feature_names: 特征名称列表
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        self.feature_names = feature_names
        
        logger.info(f"创建分类数据集: {len(self.X)} 个样本, 特征维度: {self.X.shape}")
        logger.info(f"正样本比例: {np.mean(y):.3f}")
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class ClassificationDataPreprocessor:
    """
    分类任务数据预处理器
    将生存分析数据转换为二分类数据
    """
    
    def __init__(self, config: Any):
        """
        初始化预处理器
        
        Args:
            config: 配置对象
        """
        self.config = config
        self.feature_names = None
        self.scaler = StandardScaler()
        
    def prepare_classification_data(self, 
                                  X: np.ndarray, 
                                  y: np.ndarray, 
                                  record_ids: List[Any] = None) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        准备分类数据
        
        Args:
            X: 输入特征，形状为 (n_samples, seq_len, n_features)
            y: 生存分析标签，形状为 (n_samples, 2) - [duration, event]
            record_ids: 记录ID列表
            
        Returns:
            (X_classification, y_classification, feature_names)
        """
        logger.info("开始准备分类数据...")
        
        # 提取事件标签 (y[:, 1] 表示是否发生事件)
        event_labels = y[:, 1].astype(int)
        
        # 记录数据统计
        n_samples = len(X)
        n_events = np.sum(event_labels)
        event_ratio = n_events / n_samples
        
        logger.info(f"分类数据统计:")
        logger.info(f"  总样本数: {n_samples}")
        logger.info(f"  事件样本数: {n_events}")
        logger.info(f"  事件比例: {event_ratio:.3f}")
        
        # 特征标准化
        X_reshaped = X.reshape(-1, X.shape[-1])  # (n_samples * seq_len, n_features)
        X_scaled = self.scaler.fit_transform(X_reshaped)
        X_scaled = X_scaled.reshape(X.shape)  # 恢复原始形状
        
        # 生成特征名称
        if self.feature_names is None:
            self.feature_names = [f"feature_{i}" for i in range(X.shape[-1])]
        
        logger.info(f"特征标准化完成，特征维度: {X_scaled.shape}")
        
        return X_scaled, event_labels, self.feature_names
    
    def create_data_loaders(self, 
                           X: np.ndarray, 
                           y: np.ndarray,
                           test_size: float = 0.2,
                           val_size: float = 0.2,
                           batch_size: int = 64,
                           random_state: int = 42,
                           balance_train_data: bool = False,
                           balance_method: str = 'oversample',
                           undersample_ratio: float = 1.0,
                           undersample_random_state: Optional[int] = None) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        创建数据加载器
        
        Args:
            X: 特征数据
            y: 标签数据
            test_size: 测试集比例
            val_size: 验证集比例（从训练集中划分）
            batch_size: 批次大小
            random_state: 随机种子
            balance_train_data: 是否对训练集进行数据平衡（默认False，不改变验证集和测试集）
            balance_method: 平衡方法 ('undersample', 'oversample', 'smote')
            undersample_ratio: 欠采样时的非事件:事件目标比例（仅当 balance_method='undersample' 时生效）
            undersample_random_state: 欠采样的随机种子
            
        Returns:
            (train_loader, val_loader, test_loader)
        """
        logger.info("创建数据加载器...")
        
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # 从训练集中划分验证集
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=val_size, random_state=random_state, stratify=y_train
        )
        
        # 只对训练集进行数据平衡（不改变验证集和测试集）
        if balance_train_data and balance_method:
            logger.info(f"对训练集进行数据平衡，方法: {balance_method}")
            original_train_size = len(X_train)
            X_train, y_train = self.balance_dataset(
                X_train,
                y_train,
                method=balance_method,
                undersample_ratio=undersample_ratio,
                random_state=undersample_random_state if undersample_random_state is not None else random_state
            )
            logger.info(f"训练集数据平衡完成: {original_train_size} -> {len(X_train)} 样本")
        
        # 创建数据集
        train_dataset = ClassificationDataset(X_train, y_train, self.feature_names)
        val_dataset = ClassificationDataset(X_val, y_val, self.feature_names)
        test_dataset = ClassificationDataset(X_test, y_test, self.feature_names)
        
        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=0,  # 避免多进程问题
            pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )
        
        logger.info(f"数据加载器创建完成:")
        logger.info(f"  训练集: {len(train_dataset)} 样本")
        logger.info(f"  验证集: {len(val_dataset)} 样本")
        logger.info(f"  测试集: {len(test_dataset)} 样本")
        
        return train_loader, val_loader, test_loader
    
    def balance_dataset(self, X: np.ndarray, y: np.ndarray, 
                       method: str = 'undersample',
                       undersample_ratio: float = 1.0,
                       random_state: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        平衡数据集
        
        Args:
            X: 特征数据
            y: 标签数据
            method: 平衡方法 ('undersample', 'oversample', 'smote')
            
        Returns:
            (X_balanced, y_balanced)
        """
        logger.info(f"使用 {method} 方法平衡数据集...")
        
        if method == 'undersample':
            return self._undersample(X, y, ratio=undersample_ratio, random_state=random_state)
        elif method == 'oversample':
            return self._oversample(X, y)
        elif method == 'smote':
            return self._smote_balance(X, y)
        else:
            logger.warning(f"未知的平衡方法: {method}，返回原始数据")
            return X, y
    
    def _undersample(self, X: np.ndarray, y: np.ndarray,
                     ratio: float = 1.0,
                     random_state: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """下采样多数类"""
        from collections import Counter

        class_counts = Counter(y)
        if len(class_counts) <= 1:
            logger.warning("欠采样未执行：仅检测到单一类别")
            return X, y

        try:
            ratio = float(ratio) if ratio is not None else 1.0
        except Exception:
            ratio = 1.0
        if not np.isfinite(ratio) or ratio <= 0:
            logger.warning(f"欠采样比例 {ratio} 无效，回退为 1.0")
            ratio = 1.0

        min_class_count = min(class_counts.values())
        rng = np.random.default_rng(random_state)

        class_indices = {cls: np.where(y == cls)[0] for cls in class_counts.keys()}
        selected_indices = []
        for cls, indices in class_indices.items():
            count = len(indices)
            if count == min_class_count:
                selected = indices
            else:
                target_count = int(round(min_class_count * ratio))
                target_count = max(1, target_count)
                target_count = min(target_count, count)
                if target_count >= count:
                    selected = indices
                else:
                    selected = rng.choice(indices, size=target_count, replace=False)
            selected_indices.extend(selected)

        selected_indices = np.array(selected_indices)
        rng.shuffle(selected_indices)

        logger.info(f"下采样后样本数: {len(selected_indices)} (ratio={ratio:.3f})")
        return X[selected_indices], y[selected_indices]
    
    def _oversample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """上采样少数类"""
        from collections import Counter
        
        class_counts = Counter(y)
        max_class_count = max(class_counts.values())
        
        # 获取每个类的索引
        class_indices = {cls: np.where(y == cls)[0] for cls in class_counts.keys()}
        
        # 对少数类进行重复采样
        balanced_indices = []
        for cls, indices in class_indices.items():
            current_count = len(indices)
            if current_count < max_class_count:
                # 重复采样到目标数量
                repeat_times = max_class_count // current_count
                remainder = max_class_count % current_count
                
                repeated_indices = np.tile(indices, repeat_times)
                if remainder > 0:
                    additional_indices = np.random.choice(indices, remainder, replace=False)
                    repeated_indices = np.concatenate([repeated_indices, additional_indices])
                
                balanced_indices.extend(repeated_indices)
            else:
                balanced_indices.extend(indices)
        
        balanced_indices = np.array(balanced_indices)
        np.random.shuffle(balanced_indices)
        
        logger.info(f"上采样后样本数: {len(balanced_indices)}")
        return X[balanced_indices], y[balanced_indices]
    
    def _smote_balance(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """使用SMOTE平衡数据"""
        try:
            from imblearn.over_sampling import SMOTE
            
            # 将3D数据reshape为2D进行SMOTE
            original_shape = X.shape
            X_2d = X.reshape(X.shape[0], -1)
            
            smote = SMOTE(random_state=42)
            X_balanced_2d, y_balanced = smote.fit_resample(X_2d, y)
            
            # 恢复原始形状
            X_balanced = X_balanced_2d.reshape(-1, original_shape[1], original_shape[2])
            
            logger.info(f"SMOTE平衡后样本数: {len(X_balanced)}")
            return X_balanced, y_balanced
            
        except ImportError:
            logger.warning("imbalanced-learn未安装，使用上采样代替SMOTE")
            return self._oversample(X, y)
    
    def get_class_weights(self, y: np.ndarray) -> torch.Tensor:
        """
        计算类别权重用于损失函数
        
        Args:
            y: 标签数据
            
        Returns:
            类别权重张量
        """
        from collections import Counter
        
        class_counts = Counter(y)
        total_samples = len(y)
        
        # 计算权重：总样本数 / (类别数 * 该类样本数)
        weights = []
        for i in range(len(class_counts)):
            weight = total_samples / (len(class_counts) * class_counts[i])
            weights.append(weight)
        
        return torch.FloatTensor(weights)
    
    def save_preprocessor(self, path: str):
        """保存预处理器"""
        import pickle
        
        preprocessor_data = {
            'scaler': self.scaler,
            'feature_names': self.feature_names
        }
        
        with open(path, 'wb') as f:
            pickle.dump(preprocessor_data, f)
        
        logger.info(f"预处理器已保存到: {path}")
    
    def load_preprocessor(self, path: str):
        """加载预处理器"""
        import pickle
        
        with open(path, 'rb') as f:
            preprocessor_data = pickle.load(f)
        
        self.scaler = preprocessor_data['scaler']
        self.feature_names = preprocessor_data['feature_names']
        
        logger.info(f"预处理器已从 {path} 加载")
