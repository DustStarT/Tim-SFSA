"""
分类模型训练器
整合LSTM分类模型的训练、评估和可视化
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import logging
import os
from typing import Dict, Any, Tuple, Optional
import json
from datetime import datetime

from .lstm_classifier import LSTMClassifier, LSTMClassifierTrainer
from .transformer_classifier import TransformerClassifier, TransformerClassifierTrainer
from preprocessing.classification_preprocessor import ClassificationDataPreprocessor
from evaluation.classification_metrics import ClassificationEvaluator

logger = logging.getLogger(__name__)


class ClassificationModelManager:
    """
    分类模型管理器
    负责分类模型的训练、评估和权重管理
    """
    
    def __init__(self, config: Any, device: torch.device):
        """
        初始化管理器
        
        Args:
            config: 配置对象
            device: 设备
        """
        self.config = config
        self.device = device
        self.model = None
        self.trainer = None
        self.preprocessor = None
        self.evaluator = None
        self.class_weights = None  # 类别权重
        
        # 创建输出目录
        self.output_dir = os.path.join(
            config.results_dir, 
            f"classification_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        os.makedirs(self.output_dir, exist_ok=True)
        
        logger.info(f"分类模型管理器初始化完成，输出目录: {self.output_dir}")
    
    def prepare_data(self, X: np.ndarray, y: np.ndarray, 
                    record_ids: list = None) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        准备分类数据
        
        Args:
            X: 输入特征，形状为 (n_samples, seq_len, n_features)
            y: 生存分析标签，形状为 (n_samples, 2)
            record_ids: 记录ID列表
            
        Returns:
            (train_loader, val_loader, test_loader)
        """
        logger.info("开始准备分类数据...")
        
        # 创建预处理器
        self.preprocessor = ClassificationDataPreprocessor(self.config)
        
        # 准备分类数据
        X_classification, y_classification, feature_names = self.preprocessor.prepare_classification_data(
            X, y, record_ids
        )
        
        cls_stage_cfg = getattr(self.config.model.two_stage, 'classification_stage', {})
        use_class_weights = bool(getattr(cls_stage_cfg, 'use_class_weights', True))

        if use_class_weights:
            try:
                self.class_weights = self.preprocessor.get_class_weights(y_classification)
                logger.info(f"使用类别权重: {self.class_weights.numpy()}")
            except Exception as e:
                logger.warning(f"计算类别权重失败，将继续不使用权重: {e}")
                self.class_weights = None
        else:
            self.class_weights = None
            logger.info("按配置禁用类别权重，分类损失将不使用 class weights")

        # 创建数据加载器（先划分数据，然后只对训练集进行平衡）
        sampling_mode = str(getattr(self.config.model.two_stage.classification_stage, 'sampling_mode', 'none') or 'none').lower()
        valid_modes = {'none', 'undersample', 'oversample', 'smote'}
        if sampling_mode not in valid_modes:
            logger.warning(f"未知的分类采样模式: {sampling_mode}，回退为 'none'")
            sampling_mode = 'none'

        balance_data = sampling_mode != 'none'
        balance_method = sampling_mode if balance_data else None

        sampling_params = getattr(self.config.model.two_stage.classification_stage, 'sampling_params', {}) or {}
        balance_undersample_ratio = float(getattr(sampling_params, 'undersample_ratio', 1.0)) if balance_data else 1.0
        balance_random_state = getattr(
            sampling_params,
            'random_state',
            getattr(self.config, 'seed', None)
        ) if balance_data else getattr(self.config, 'seed', None)
        
        train_loader, val_loader, test_loader = self.preprocessor.create_data_loaders(
            X_classification, y_classification,
            test_size=0.2,
            val_size=0.2,
            batch_size=self.config.model.two_stage.classification_stage.batch_size,
            random_state=self.config.data.random_seed,
            balance_train_data=balance_data,
            balance_method=balance_method,
            undersample_ratio=balance_undersample_ratio,
            undersample_random_state=balance_random_state
        )
        
        # 保存预处理器
        preprocessor_path = os.path.join(self.output_dir, "classification_preprocessor.pkl")
        self.preprocessor.save_preprocessor(preprocessor_path)
        
        logger.info("分类数据准备完成")
        return train_loader, val_loader, test_loader
    
    def create_model(self, input_size: int):
        """
        按 encoder.type 创建分类模型：'lstm' 或 'transformer'
        """
        enc_type = str(getattr(getattr(self.config.model, 'encoder', {}), 'type', 'lstm')).lower()
        if enc_type == 'transformer':
            enc_cfg = getattr(getattr(self.config.model, 'encoder', {}), 'transformer', {})
            logger.info(f"创建Transformer分类模型，输入维度: {input_size}")
            d_model = int(getattr(enc_cfg, 'd_model', 128))
            nhead = int(getattr(enc_cfg, 'nhead', 4))
            if d_model % max(1, nhead) != 0:
                raise ValueError(
                    f"Transformer 配置无效: d_model={d_model} 必须能被 nhead={nhead} 整除。"
                )
            self.model = TransformerClassifier(
                input_size=input_size,
                d_model=d_model,
                nhead=nhead,
                num_layers=int(getattr(enc_cfg, 'num_layers', 4)),
                dim_feedforward=int(getattr(enc_cfg, 'dim_feedforward', 256)),
                dropout=float(getattr(enc_cfg, 'dropout', 0.2)),
                activation=str(getattr(enc_cfg, 'activation', 'gelu')),
                norm=str(getattr(enc_cfg, 'norm', 'layernorm')),
                num_classes=2,
                use_cls_token=bool(getattr(enc_cfg, 'use_cls_token', True)),
                pooling=str(getattr(enc_cfg, 'pooling', 'cls'))
            )
            self.trainer = TransformerClassifierTrainer(self.model, self.config, self.device, class_weights=self.class_weights)
            self.evaluator = ClassificationEvaluator(
                model_name=f"{self.config.model.name}_TransformerClassifier",
                output_dir=self.output_dir
            )
            logger.info("Transformer分类模型创建完成")
        else:
            logger.info(f"创建LSTM分类模型，输入维度: {input_size}")
            lstm_config = self.config.model.lstm
            self.model = LSTMClassifier(
                input_size=input_size,
                hidden_size=lstm_config.hidden_size,
                num_lstm_layers=lstm_config.num_lstm_layers,
                dropout_rate=lstm_config.dropout_rate,
                bidirectional=lstm_config.bidirectional,
                use_attention=lstm_config.use_attention,
                num_classes=2
            )
            self.trainer = LSTMClassifierTrainer(self.model, self.config, self.device, class_weights=self.class_weights)
            self.evaluator = ClassificationEvaluator(
                model_name=f"{self.config.model.name}_Classifier",
                output_dir=self.output_dir
            )
            logger.info("LSTM分类模型创建完成")
        return self.model
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict[str, Any]:
        """
        训练分类模型
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            
        Returns:
            训练结果字典
        """
        enc_type = str(getattr(getattr(self.config.model, 'encoder', {}), 'type', 'lstm')).lower()
        logger.info(f"开始训练{('Transformer' if enc_type=='transformer' else 'LSTM')}分类模型...")
        
        # 训练模型
        self.trainer.train(train_loader, val_loader, self.output_dir)
        
        # 收集训练历史
        training_history = {
            'train_losses': self.trainer.train_losses,
            'val_losses': self.trainer.val_losses,
            'train_accuracies': self.trainer.train_accuracies,
            'val_accuracies': self.trainer.val_accuracies,
            'val_f1_scores': self.trainer.val_f1_scores,
            'val_auc_scores': self.trainer.val_auc_scores,
            'best_threshold': getattr(self.trainer, 'best_threshold', 0.5),
            'best_threshold_f1': getattr(self.trainer, 'best_threshold_f1', None)
        }
        
        # 保存训练历史
        history_path = os.path.join(self.output_dir, "training_history.json")
        with open(history_path, 'w') as f:
            json.dump(training_history, f, indent=2)

        threshold_info = {
            'best_threshold': getattr(self.trainer, 'best_threshold', 0.5),
            'best_threshold_f1': getattr(self.trainer, 'best_threshold_f1', None)
        }
        threshold_path = os.path.join(self.output_dir, "best_threshold.json")
        with open(threshold_path, 'w') as f:
            json.dump(threshold_info, f, indent=2)
        
        logger.info(f"{('Transformer' if enc_type=='transformer' else 'LSTM')}分类模型训练完成")
        return training_history
    
    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """
        评估分类模型
        
        Args:
            test_loader: 测试数据加载器
            
        Returns:
            评估指标字典
        """
        enc_type = str(getattr(getattr(self.config.model, 'encoder', {}), 'type', 'lstm')).lower()
        logger.info(f"开始评估{('Transformer' if enc_type=='transformer' else 'LSTM')}分类模型...")
        
        # 评估模型
        metrics = self.trainer.evaluate(test_loader)
        
        # 获取预测结果用于可视化
        self.model.eval()
        all_preds = []
        all_targets = []
        all_probs = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                probs = torch.softmax(output, dim=1)
                threshold = getattr(self.trainer, 'best_threshold', 0.5)
                pred = (probs[:, 1] >= threshold).long()
                
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())  # 正类概率
        
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        all_probs = np.array(all_probs)
        
        # 生成可视化
        self.evaluator.plot_all_metrics(
            all_targets, all_preds, all_probs, 
            self.trainer.__dict__  # 传递训练历史
        )
        
        # 保存评估结果
        results_path = os.path.join(self.output_dir, "evaluation_results.json")
        with open(results_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"{('Transformer' if enc_type=='transformer' else 'LSTM')}分类模型评估完成")
        return metrics
    
    def save_model(self, path: str = None) -> str:
        """
        保存模型
        
        Args:
            path: 保存路径
            
        Returns:
            实际保存路径
        """
        if path is None:
            path = os.path.join(self.output_dir, "classification_model.pth")
        
        # 保存模型状态（兼容 LSTM / Transformer）
        enc_type = str(getattr(getattr(self.config.model, 'encoder', {}), 'type', 'lstm')).lower()
        state = {
            'model_state_dict': self.model.state_dict(),
            'encoder_type': enc_type,
            'training_config': {
                'num_epochs': self.config.model.two_stage.classification_stage.num_epochs,
                'learning_rate': self.config.model.two_stage.classification_stage.learning_rate,
                'batch_size': self.config.model.two_stage.classification_stage.batch_size
            },
            'decision_threshold': getattr(self.trainer, 'best_threshold', 0.5)
        }
        if enc_type == 'transformer':
            state['model_config'] = {
                'input_size': getattr(self.model, 'input_size', None),
                'd_model': getattr(self.model, 'd_model', None),
                'nhead': getattr(self.model, 'nhead', None),
                'num_layers': getattr(self.model, 'num_layers', None),
                'dim_feedforward': getattr(self.model, 'dim_feedforward', None),
                'dropout': getattr(self.model, 'dropout', None),
                'activation': getattr(self.model, 'activation', None),
                'norm': getattr(self.model, 'norm', None),
                'num_classes': getattr(self.model, 'num_classes', None)
            }
        else:
            state['model_config'] = {
                'input_size': self.model.input_size,
                'hidden_size': self.model.hidden_size,
                'num_lstm_layers': self.model.num_lstm_layers,
                'dropout_rate': self.model.dropout_rate,
                'bidirectional': self.model.bidirectional,
                'use_attention': self.model.use_attention,
                'num_classes': self.model.num_classes
            }
        torch.save(state, path)
        
        logger.info(f"分类模型已保存到: {path}")
        return path
    
    def load_model(self, path: str):
        """
        加载模型
        
        Args:
            path: 模型路径
            
        Returns:
            加载的模型
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        # 检查是否是直接的state_dict格式（没有元数据）
        has_meta_keys = isinstance(checkpoint, dict) and any(k in checkpoint for k in ['model_state_dict', 'state_dict', 'model_config', 'encoder_type'])
        
        if not has_meta_keys and isinstance(checkpoint, dict):
            # 这是一个直接的state_dict，从键名推断编码器类型
            logger.info("检测到checkpoint为直接的state_dict格式")
            
            # 从state_dict键推断编码器类型
            enc_type = 'lstm'  # 默认
            if any('lstm' in k.lower() for k in checkpoint.keys()):
                enc_type = 'lstm'
                logger.info("从state_dict推断为LSTM分类模型")
            elif any('transformer' in k.lower() or 'encoder.layers' in k for k in checkpoint.keys()):
                enc_type = 'transformer'
                logger.info("从state_dict推断为Transformer分类模型")
            else:
                # 回退到config
                enc_type = str(getattr(getattr(self.config.model, 'encoder', {}), 'type', 'lstm')).lower()
                logger.info(f"无法从state_dict推断类型，使用config中的{enc_type}")
            
            # 从state_dict键推断模型配置
            input_size = 24  # 默认值
            hidden_size = 64  # 默认
            lstm_layers = 7  # 默认
            bidirectional = True  # 默认
            use_attention = True  # 默认
            
            # 首先从LSTM权重推断input_size和hidden_size
            for key in checkpoint.keys():
                if 'lstm.weight_ih_l0' in key and '_reverse' not in key:
                    weight_shape = checkpoint[key].shape
                    if len(weight_shape) == 2:
                        # shape: [4*hidden_size, input_size]
                        lstm_weight_dim = weight_shape[0]  # 4 * hidden_size
                        input_size_from_checkpoint = weight_shape[1]  # input_size
                        
                        if lstm_weight_dim % 4 == 0:
                            hidden_size = lstm_weight_dim // 4
                            input_size = input_size_from_checkpoint
                            logger.info(f"从checkpoint推断: input_size={input_size}, hidden_size={hidden_size}")
                            break
            
            # 从checkpoint推断LSTM配置
            if enc_type == 'lstm':
                # 计算LSTM层数
                layer_keys = [k for k in checkpoint.keys() if 'lstm.weight_ih_l' in k]
                if layer_keys:
                    max_layer_idx = max(int(k.split('_l')[1].split('_')[0]) for k in layer_keys)
                    lstm_layers = max_layer_idx + 1
                    logger.info(f"从checkpoint推断LSTM层数={lstm_layers}")
                
                # 检查是否是双向（从checkpoint键名推断）
                if any('_reverse' in k for k in checkpoint.keys()):
                    bidirectional = True
                    logger.info("从checkpoint推断双向LSTM=True")
                else:
                    bidirectional = False
                    logger.info("从checkpoint推断单向LSTM")
                
                # hidden_size和input_size已经在前面推断过了
                
                # 检查是否有attention
                if any('attention' in k for k in checkpoint.keys()):
                    use_attention = True
                    logger.info("从checkpoint推断使用注意力机制")
            
            state_dict = checkpoint  # checkpoint本身就是state_dict
            
            # 创建模型 - 使用推断出的配置
            if enc_type == 'transformer':
                enc_cfg = getattr(getattr(self.config.model, 'encoder', {}), 'transformer', {})
                model_config = {
                    'input_size': input_size,
                    'd_model': int(getattr(enc_cfg, 'd_model', 64)),
                    'nhead': int(getattr(enc_cfg, 'nhead', 2)),
                    'num_layers': int(getattr(enc_cfg, 'num_layers', 2)),
                    'dim_feedforward': int(getattr(enc_cfg, 'dim_feedforward', 128)),
                    'dropout': float(getattr(enc_cfg, 'dropout', 0.5)),
                    'activation': str(getattr(enc_cfg, 'activation', 'gelu')),
                    'norm': str(getattr(enc_cfg, 'norm', 'layernorm')),
                    'num_classes': 2
                }
                self.model = TransformerClassifier(**model_config)
                self.trainer = TransformerClassifierTrainer(self.model, self.config, self.device)
                model_name = f"{self.config.model.name}_TransformerClassifier"
            else:
                # 使用从checkpoint推断出的配置
                lstm_config = self.config.model.lstm
                model_config = {
                    'input_size': input_size,
                    'hidden_size': hidden_size,  # 使用推断出的维度
                    'num_lstm_layers': lstm_layers,  # 使用推断出的层数
                    'dropout_rate': lstm_config.dropout_rate,
                    'bidirectional': bidirectional,  # 使用推断出的方向
                    'use_attention': use_attention,  # 使用推断出的注意力
                    'num_classes': 2
                }
                self.model = LSTMClassifier(**model_config)
                self.trainer = LSTMClassifierTrainer(self.model, self.config, self.device, class_weights=self.class_weights)
                model_name = f"{self.config.model.name}_Classifier"
        else:
            # 有元数据的checkpoint格式
            enc_type = str(checkpoint.get('encoder_type', 'lstm')).lower()
            model_config = checkpoint.get('model_config', {})
            
            # 过滤掉None值
            model_config_clean = {}
            for k, v in model_config.items():
                if v is not None:
                    model_config_clean[k] = v
            
            # 如果model_config_clean为空，从config中获取
            if not model_config_clean:
                logger.warning("model_config为空，使用config中的默认值")
                if enc_type == 'transformer':
                    enc_cfg = getattr(getattr(self.config.model, 'encoder', {}), 'transformer', {})
                    model_config_clean = {
                        'input_size': 24,
                        'd_model': int(getattr(enc_cfg, 'd_model', 128)),
                        'nhead': int(getattr(enc_cfg, 'nhead', 4)),
                        'num_layers': int(getattr(enc_cfg, 'num_layers', 4)),
                        'dim_feedforward': int(getattr(enc_cfg, 'dim_feedforward', 256)),
                        'dropout': float(getattr(enc_cfg, 'dropout', 0.2)),
                        'activation': str(getattr(enc_cfg, 'activation', 'gelu')),
                        'norm': str(getattr(enc_cfg, 'norm', 'layernorm')),
                        'num_classes': 2
                    }
                else:
                    lstm_config = self.config.model.lstm
                    model_config_clean = {
                        'input_size': 24,
                        'hidden_size': lstm_config.hidden_size,
                        'num_lstm_layers': lstm_config.num_lstm_layers,
                        'dropout_rate': lstm_config.dropout_rate,
                        'bidirectional': lstm_config.bidirectional,
                        'use_attention': lstm_config.use_attention,
                        'num_classes': 2
                    }
            
            # 创建模型
            if enc_type == 'transformer':
                self.model = TransformerClassifier(**model_config_clean)
                self.trainer = TransformerClassifierTrainer(self.model, self.config, self.device)
                model_name = f"{self.config.model.name}_TransformerClassifier"
            else:
                self.model = LSTMClassifier(**model_config_clean)
                self.trainer = LSTMClassifierTrainer(self.model, self.config, self.device, class_weights=self.class_weights)
                model_name = f"{self.config.model.name}_Classifier"
            
            # 从checkpoint获取state_dict
            state_dict = checkpoint.get('model_state_dict') or checkpoint.get('state_dict') or checkpoint
        
        # 加载权重
        try:
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
        except Exception as e:
            logger.error(f"加载state_dict失败: {e}")
            raise
        
        # 创建评估器
        self.evaluator = ClassificationEvaluator(
            model_name=model_name,
            output_dir=self.output_dir
        )
        
        logger.info(f"分类模型已从 {path} 加载")
        return self.model
    
    def get_lstm_weights(self) -> Dict[str, torch.Tensor]:
        """
        获取LSTM/Transformer权重用于后续生存分析模型
        
        Returns:
            LSTM/Transformer权重字典
        """
        if self.model is None:
            raise ValueError("模型未初始化")
        
        enc_type = str(getattr(getattr(self.config.model, 'encoder', {}), 'type', 'lstm')).lower()
        if enc_type == 'transformer':
            # 返回Transformer权重
            return self.model.get_transformer_weights()
        else:
            # 返回LSTM权重
            return self.model.get_lstm_weights()
    
    def run_full_training_pipeline(self, X: np.ndarray, y: np.ndarray, 
                                  record_ids: list = None) -> Dict[str, Any]:
        """
        运行完整的训练流程
        
        Args:
            X: 输入特征
            y: 生存分析标签
            record_ids: 记录ID列表
            
        Returns:
            完整结果字典
        """
        logger.info("开始运行完整的分类训练流程...")
        
        # 1. 准备数据
        train_loader, val_loader, test_loader = self.prepare_data(X, y, record_ids)
        
        # 2. 创建模型
        input_size = X.shape[-1]
        self.create_model(input_size)
        
        # 3. 训练模型
        training_history = self.train(train_loader, val_loader)
        
        # 4. 评估模型
        evaluation_metrics = self.evaluate(test_loader)
        
        # 5. 保存模型
        model_path = self.save_model()
        
        # 6. 收集所有结果
        results = {
            'training_history': training_history,
            'evaluation_metrics': evaluation_metrics,
            'model_path': model_path,
            'output_dir': self.output_dir,
            'best_val_auc': self.trainer.best_val_auc
        }
        
        # 保存完整结果
        results_path = os.path.join(self.output_dir, "complete_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info("完整的分类训练流程完成")
        return results


def run_classification_only(config: Any, X: np.ndarray, y: np.ndarray, 
                           record_ids: list = None) -> Dict[str, Any]:
    """
    仅运行分类任务的便捷函数
    
    Args:
        config: 配置对象
        X: 输入特征
        y: 生存分析标签
        record_ids: 记录ID列表
        
    Returns:
        分类结果字典
    """
    device = torch.device(config.training.device)
    
    # 创建管理器
    manager = ClassificationModelManager(config, device)
    
    # 运行完整流程
    results = manager.run_full_training_pipeline(X, y, record_ids)
    
    return results
