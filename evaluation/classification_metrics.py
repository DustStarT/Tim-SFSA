"""
分类任务评估指标和可视化
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report, f1_score, accuracy_score,
    precision_score, recall_score, roc_auc_score
)
import logging
from typing import List, Dict, Any, Tuple, Optional
import os

logger = logging.getLogger(__name__)


class ClassificationEvaluator:
    """
    分类任务评估器
    """
    
    def __init__(self, model_name: str = "LSTM_Classifier", output_dir: str = "results"):
        """
        初始化评估器
        
        Args:
            model_name: 模型名称
            output_dir: 输出目录
        """
        self.model_name = model_name
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 设置绘图样式
        plt.style.use('default')
        sns.set_palette("husl")
    
    def evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray, 
                      y_probs: np.ndarray = None) -> Dict[str, float]:
        """
        评估模型性能
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            y_probs: 预测概率（用于计算AUC等）
            
        Returns:
            评估指标字典
        """
        metrics = {}
        
        # 基本指标
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, average='binary')
        metrics['recall'] = recall_score(y_true, y_pred, average='binary')
        metrics['f1_score'] = f1_score(y_true, y_pred, average='binary')
        
        # AUC相关指标
        if y_probs is not None:
            metrics['roc_auc'] = roc_auc_score(y_true, y_probs)
            metrics['pr_auc'] = average_precision_score(y_true, y_probs)
        else:
            metrics['roc_auc'] = 0.0
            metrics['pr_auc'] = 0.0
        
        # 打印结果
        logger.info("分类模型评估结果:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        return metrics
    
    def plot_roc_curve(self, y_true: np.ndarray, y_probs: np.ndarray, 
                      save_path: str = None) -> None:
        """
        绘制ROC曲线
        
        Args:
            y_true: 真实标签
            y_probs: 预测概率
            save_path: 保存路径
        """
        fpr, tpr, _ = roc_curve(y_true, y_probs)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                label='Random classifier')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.title(f'{self.model_name} - ROC curve')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        if save_path is None:
            save_path = os.path.join(self.output_dir, f'{self.model_name}_roc_curve.png')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"ROC曲线已保存到: {save_path}")
    
    def plot_precision_recall_curve(self, y_true: np.ndarray, y_probs: np.ndarray,
                                   save_path: str = None) -> None:
        """
        绘制精确率-召回率曲线
        
        Args:
            y_true: 真实标签
            y_probs: 预测概率
            save_path: 保存路径
        """
        precision, recall, _ = precision_recall_curve(y_true, y_probs)
        pr_auc = average_precision_score(y_true, y_probs)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='darkorange', lw=2,
                label=f'PR curve (AUC = {pr_auc:.3f})')
        
        # 添加基线（随机分类器）
        baseline = np.sum(y_true) / len(y_true)
        plt.axhline(y=baseline, color='navy', linestyle='--', 
                   label=f'Random classifier (AP = {baseline:.3f})')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'{self.model_name} - Precision-Recall')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        
        if save_path is None:
            save_path = os.path.join(self.output_dir, f'{self.model_name}_pr_curve.png')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"PR曲线已保存到: {save_path}")
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                             save_path: str = None) -> None:
        """
        绘制混淆矩阵
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            save_path: 保存路径
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['No flare', 'flare'],
                   yticklabels=['No flare', 'flare'])
        
        plt.title(f'{self.model_name} - Confusion matrix')
        plt.xlabel('Predicting labels')
        plt.ylabel('Real label')
        
        if save_path is None:
            save_path = os.path.join(self.output_dir, f'{self.model_name}_confusion_matrix.png')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"混淆矩阵已保存到: {save_path}")
    
    def plot_training_history(self, history: Dict[str, List[float]], 
                             save_path: str = None) -> None:
        """
        绘制训练历史
        
        Args:
            history: 训练历史字典
            save_path: 保存路径
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 损失曲线
        axes[0, 0].plot(history.get('train_losses', []), label='Training loss', color='blue')
        axes[0, 0].plot(history.get('val_losses', []), label='Validation loss', color='red')
        axes[0, 0].set_title('Training and validation loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 准确率曲线
        axes[0, 1].plot(history.get('train_accuracies', []), label='Training accuracy', color='blue')
        axes[0, 1].plot(history.get('val_accuracies', []), label='Validation accuracy', color='red')
        axes[0, 1].set_title('Train and validate accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # F1分数曲线
        axes[1, 0].plot(history.get('val_f1_scores', []), label='Validating F1 score', color='green')
        axes[1, 0].set_title('Validating F1 score')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('F1 score')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # AUC曲线
        axes[1, 1].plot(history.get('val_auc_scores', []), label='Validate AUC', color='purple')
        axes[1, 1].set_title('Validate AUC')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('AUC')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(f'{self.model_name} - Training History', fontsize=16)
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.output_dir, f'{self.model_name}_training_history.png')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"训练历史图已保存到: {save_path}")
    
    def plot_class_distribution(self, y_true: np.ndarray, y_pred: np.ndarray = None,
                               save_path: str = None) -> None:
        """
        绘制类别分布
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签（可选）
            save_path: 保存路径
        """
        fig, axes = plt.subplots(1, 2 if y_pred is not None else 1, figsize=(12, 5))
        
        if y_pred is None:
            axes = [axes]
        
        # 真实标签分布
        unique, counts = np.unique(y_true, return_counts=True)
        axes[0].bar(['No flare', 'flare'], counts, color=['skyblue', 'orange'])
        axes[0].set_title('True label distribution')
        axes[0].set_ylabel('Number of samples')
        
        # 预测标签分布
        if y_pred is not None:
            unique_pred, counts_pred = np.unique(y_pred, return_counts=True)
            axes[1].bar(['No flare', 'flare'], counts_pred, color=['lightgreen', 'red'])
            axes[1].set_title('Predicting label distribution')
            axes[1].set_ylabel('Number of samples')
        
        plt.suptitle(f'{self.model_name} - Class distribution', fontsize=14)
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.output_dir, f'{self.model_name}_class_distribution.png')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"类别分布图已保存到: {save_path}")
    
    def generate_classification_report(self, y_true: np.ndarray, y_pred: np.ndarray,
                                     save_path: str = None) -> str:
        """
        生成分类报告
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            save_path: 保存路径
            
        Returns:
            分类报告字符串
        """
        report = classification_report(y_true, y_pred, 
                                     target_names=['No flare', 'flare'],
                                     output_dict=False)
        
        if save_path is None:
            save_path = os.path.join(self.output_dir, f'{self.model_name}_classification_report.txt')
        
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(f"{self.model_name} 分类报告\n")
            f.write("=" * 50 + "\n")
            f.write(report)
        
        logger.info(f"分类报告已保存到: {save_path}")
        return report
    
    def plot_all_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                        y_probs: np.ndarray = None, history: Dict[str, List[float]] = None) -> None:
        """
        绘制所有评估指标
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            y_probs: 预测概率
            history: 训练历史
        """
        logger.info("生成所有评估图表...")
        
        # 基本评估
        metrics = self.evaluate_model(y_true, y_pred, y_probs)
        
        # 绘制各种图表
        if y_probs is not None:
            self.plot_roc_curve(y_true, y_probs)
            self.plot_precision_recall_curve(y_true, y_probs)
        
        self.plot_confusion_matrix(y_true, y_pred)
        self.plot_class_distribution(y_true, y_pred)
        
        if history is not None:
            self.plot_training_history(history)
        
        # 生成报告
        self.generate_classification_report(y_true, y_pred)
        
        logger.info("所有评估图表生成完成")
