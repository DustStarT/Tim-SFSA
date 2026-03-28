"""
LSTM分类模型
用于两阶段训练的第一阶段：预测窗口内是否有耀斑发生
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import numpy as np
from typing import Optional, Tuple, Dict, Any

from models.losses import FocalLoss

logger = logging.getLogger(__name__)


class LSTMClassifier(nn.Module):
    """
    LSTM分类模型，用于预测时间序列中是否会发生耀斑事件
    
    该模型包含：
    1. LSTM层用于序列建模
    2. 注意力机制（可选）
    3. 全连接层用于分类
    """
    
    def __init__(self, 
                 input_size: int,
                 hidden_size: int = 64,
                 num_lstm_layers: int = 2,
                 dropout_rate: float = 0.3,
                 bidirectional: bool = False,
                 use_attention: bool = True,
                 num_classes: int = 2):
        """
        初始化LSTM分类模型
        
        Args:
            input_size: 输入特征维度
            hidden_size: LSTM隐藏层大小
            num_lstm_layers: LSTM层数
            dropout_rate: Dropout比率
            bidirectional: 是否使用双向LSTM
            use_attention: 是否使用注意力机制
            num_classes: 分类类别数（默认2：有/无耀斑）
        """
        super(LSTMClassifier, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_lstm_layers = num_lstm_layers
        self.dropout_rate = dropout_rate
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        self.num_classes = num_classes
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            dropout=dropout_rate if num_lstm_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # 注意力机制
        if use_attention:
            lstm_output_size = hidden_size * (2 if bidirectional else 1)
            self.attention = nn.MultiheadAttention(
                embed_dim=lstm_output_size,
                num_heads=4,
                dropout=dropout_rate,
                batch_first=True
            )
            self.attention_norm = nn.LayerNorm(lstm_output_size)
        
        # 分类头
        lstm_output_size = hidden_size * (2 if bidirectional else 1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(lstm_output_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, num_classes)
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化模型权重"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'lstm' in name:
                    nn.init.xavier_uniform_(param)
                elif 'classifier' in name:
                    nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 (batch_size, seq_len, input_size)
            
        Returns:
            分类logits，形状为 (batch_size, num_classes)
        """
        batch_size, seq_len, _ = x.shape
        
        # LSTM前向传播
        lstm_out, (hidden, cell) = self.lstm(x)  # (batch_size, seq_len, hidden_size * directions)
        
        # 注意力机制
        if self.use_attention:
            # 自注意力
            attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
            # 残差连接和层归一化
            lstm_out = self.attention_norm(lstm_out + attn_out)
        
        # 全局平均池化 + 最大池化
        avg_pool = torch.mean(lstm_out, dim=1)  # (batch_size, hidden_size * directions)
        # max_pool, _ = torch.max(lstm_out, dim=1)  # (batch_size, hidden_size * directions)
        
        # print(avg_pool.shape(), max_pool.shape())

        # 拼接两种池化结果
        # pooled = torch.cat(avg_pool, dim=1)  # (batch_size, 2 * hidden_size * directions)
        
        # 分类
        logits = self.classifier(avg_pool)
        
        return logits
    
    def get_lstm_weights(self) -> Dict[str, torch.Tensor]:
        """
        获取LSTM层的权重，用于后续生存分析模型的初始化
        
        Returns:
            包含LSTM权重的字典
        """
        lstm_weights = {}
        for name, param in self.lstm.named_parameters():
            lstm_weights[f"lstm.{name}"] = param.data.clone()
        return lstm_weights
    
    def load_lstm_weights(self, weights: Dict[str, torch.Tensor]):
        """
        加载LSTM权重
        
        Args:
            weights: 包含LSTM权重的字典
        """
        lstm_state_dict = {}
        for name, weight in weights.items():
            if name.startswith("lstm."):
                lstm_state_dict[name[5:]] = weight  # 移除"lstm."前缀
        
        self.lstm.load_state_dict(lstm_state_dict)
        logger.info("LSTM权重加载成功")
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        预测概率
        
        Args:
            x: 输入张量
            
        Returns:
            预测概率，形状为 (batch_size, num_classes)
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=1)
        return probs
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        预测类别
        
        Args:
            x: 输入张量
            
        Returns:
            预测类别，形状为 (batch_size,)
        """
        probs = self.predict_proba(x)
        return torch.argmax(probs, dim=1)


class LSTMClassifierTrainer:
    """
    LSTM分类模型训练器
    """
    
    def __init__(self, 
                 model: LSTMClassifier,
                 config: Any,
                 device: torch.device,
                 class_weights: torch.Tensor = None):
        """
        初始化训练器
        
        Args:
            model: LSTM分类模型
            config: 配置对象
            device: 设备
            class_weights: 类别权重张量
        """
        self.model = model
        self.config = config
        self.device = device
        self.model.to(device)
        
        # 训练配置
        self.num_epochs = config.model.two_stage.classification_stage.num_epochs
        self.learning_rate = config.model.two_stage.classification_stage.learning_rate
        self.batch_size = config.model.two_stage.classification_stage.batch_size
        self.patience = config.model.two_stage.classification_stage.early_stopping_patience
        
        # 优化器和损失函数
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-4
        )
        
        ls = float(getattr(config.model.two_stage.classification_stage, 'label_smoothing', 0.0))
        loss_cfg = getattr(config.model.two_stage.classification_stage, 'loss', {}) or {}
        loss_type = str(getattr(loss_cfg, 'type', 'cross_entropy')).lower()
        self.loss_type = loss_type

        if loss_type == 'focal':
            alpha = float(getattr(loss_cfg, 'alpha', 0.25))
            gamma = float(getattr(loss_cfg, 'gamma', 2.0))
            self.criterion = FocalLoss(alpha=alpha, gamma=gamma)
            if class_weights is not None:
                logger.warning("FocalLoss 当前不支持类别权重，传入的 class_weights 将被忽略。")
            if ls > 0:
                logger.warning("FocalLoss 与 label_smoothing 不兼容，label_smoothing 参数将被忽略。")
            logger.info(f"使用 FocalLoss(alpha={alpha}, gamma={gamma}) 作为分类损失")
        else:
            if class_weights is not None:
                class_weights = class_weights.to(device)
                if ls > 0:
                    self.criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=ls)
                    logger.info(f"使用类别权重损失函数（带label smoothing={ls}）: {class_weights.cpu().numpy()}")
                else:
                    self.criterion = nn.CrossEntropyLoss(weight=class_weights)
                    logger.info(f"使用类别权重损失函数（无label smoothing）: {class_weights.cpu().numpy()}")
            else:
                if ls > 0:
                    self.criterion = nn.CrossEntropyLoss(label_smoothing=ls)
                    logger.info(f"使用未加权损失函数（带label smoothing={ls}）")
                else:
                    self.criterion = nn.CrossEntropyLoss()
                    logger.info("使用未加权损失函数（无label smoothing）")
        
        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=10,
            min_lr=1e-6,
            verbose=True
        )
        
        # 训练历史
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.val_f1_scores = []
        self.val_auc_scores = []
        
        # 早停
        self.best_val_auc = 0.0
        self.epochs_no_improve = 0
        self.best_model_state = None
        self.best_threshold = 0.5
        self.best_threshold_f1 = 0.0
    
    def train_epoch(self, train_loader) -> Tuple[float, float]:
        """
        训练一个epoch
        
        Args:
            train_loader: 训练数据加载器
            
        Returns:
            (平均损失, 准确率)
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # 统计
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100.0 * correct / total
        
        return avg_loss, accuracy
    
    def validate_epoch(self, val_loader) -> Tuple[float, float, float, float]:
        """
        验证一个epoch
        
        Args:
            val_loader: 验证数据加载器
            
        Returns:
            (平均损失, 准确率, F1分数, AUC分数)
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []
        all_probs = []
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
                
                # 收集预测结果用于计算指标
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                probs = F.softmax(output, dim=1)
                all_probs.extend(probs[:, 1].cpu().numpy())  # 正类概率
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100.0 * correct / total
        
        # 计算F1和AUC
        f1_score = self._calculate_f1_score(all_targets, all_preds)
        auc_score = self._calculate_auc_score(all_targets, all_probs)
        
        return avg_loss, accuracy, f1_score, auc_score
    
    def _calculate_f1_score(self, y_true, y_pred):
        """计算F1分数"""
        try:
            from sklearn.metrics import f1_score
            return f1_score(y_true, y_pred, average='binary')
        except ImportError:
            logger.warning("sklearn未安装，无法计算F1分数")
            return 0.0
    
    def _calculate_auc_score(self, y_true, y_probs):
        """计算AUC分数"""
        try:
            from sklearn.metrics import roc_auc_score
            return roc_auc_score(y_true, y_probs)
        except ImportError:
            logger.warning("sklearn未安装，无法计算AUC分数")
            return 0.0
    
    def train(self, train_loader, val_loader, output_dir: str = None):
        """
        训练模型
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            output_dir: 输出目录
        """
        logger.info(f"开始训练LSTM分类模型，共{self.num_epochs}个epoch")
        
        for epoch in range(self.num_epochs):
            # 训练
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # 验证
            val_loss, val_acc, val_f1, val_auc = self.validate_epoch(val_loader)
            
            # 记录历史
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            self.val_f1_scores.append(val_f1)
            self.val_auc_scores.append(val_auc)
            
            # 学习率调度
            self.scheduler.step(val_auc)
            
            # 早停检查
            if val_auc > self.best_val_auc:
                self.best_val_auc = val_auc
                self.epochs_no_improve = 0
                self.best_model_state = self.model.state_dict().copy()
                
                # 保存最佳模型
                if output_dir and self.config.model.two_stage.classification_stage.save_best_model:
                    model_path = f"{output_dir}/classification_best_model.pth"
                    torch.save(self.best_model_state, model_path)
                    logger.info(f"保存最佳模型到: {model_path}")
            else:
                self.epochs_no_improve += 1
            
            # 打印进度
            if epoch % 10 == 0 or epoch == self.num_epochs - 1:
                logger.info(
                    f"Epoch {epoch+1}/{self.num_epochs}: "
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, "
                    f"Val F1: {val_f1:.4f}, Val AUC: {val_auc:.4f}"
                )
            
            # 早停
            if self.epochs_no_improve >= self.patience:
                logger.info(f"早停：验证AUC在{self.patience}个epoch内未提升")
                break
        
        # 加载最佳模型
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            logger.info("已加载最佳模型权重")
        
        logger.info(f"训练完成，最佳验证AUC: {self.best_val_auc:.4f}")

        # 基于验证集选择最佳概率阈值
        self.best_threshold, self.best_threshold_f1 = self._determine_best_threshold(val_loader)
        logger.info(
            "基于验证集的最佳阈值: %.4f (F1=%.4f)",
            self.best_threshold,
            self.best_threshold_f1
        )
    
    def evaluate(self, test_loader) -> Dict[str, float]:
        """
        评估模型
        
        Args:
            test_loader: 测试数据加载器
            
        Returns:
            评估指标字典
        """
        self.model.eval()
        all_preds = []
        all_targets = []
        all_probs = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                probs = F.softmax(output, dim=1)
                threshold = getattr(self, 'best_threshold', 0.5)
                pred = (probs[:, 1] >= threshold).long()
                
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())
        
        # 计算指标
        accuracy = 100.0 * sum(p == t for p, t in zip(all_preds, all_targets)) / len(all_targets)
        f1_score = self._calculate_f1_score(all_targets, all_preds)
        auc_score = self._calculate_auc_score(all_targets, all_probs)
        
        metrics = {
            'accuracy': accuracy,
            'f1_score': f1_score,
            'auc_score': auc_score,
            'precision': self._calculate_precision(all_targets, all_preds),
            'recall': self._calculate_recall(all_targets, all_preds),
            'threshold': getattr(self, 'best_threshold', 0.5)
        }
        
        return metrics

    def _determine_best_threshold(self, val_loader) -> Tuple[float, float]:
        """在验证集上搜索最佳概率阈值以最大化F1。"""
        try:
            from sklearn.metrics import precision_recall_curve
        except ImportError:
            logger.warning("sklearn 未安装，无法搜索最佳阈值，使用默认0.5。")
            return 0.5, 0.0

        self.model.eval()
        all_probs = []
        all_targets = []

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                probs = F.softmax(output, dim=1)
                all_probs.extend(probs[:, 1].cpu().numpy())
                all_targets.extend(target.cpu().numpy())

        if not all_probs:
            return 0.5, 0.0

        all_probs = np.asarray(all_probs)
        all_targets = np.asarray(all_targets)

        precision, recall, thresholds = precision_recall_curve(all_targets, all_probs)
        if thresholds.size == 0:
            return 0.5, 0.0

        f1_scores = 2 * precision[1:] * recall[1:] / (precision[1:] + recall[1:] + 1e-8)
        best_idx = int(np.nanargmax(f1_scores))
        best_threshold = float(thresholds[best_idx])
        best_f1 = float(f1_scores[best_idx])

        return best_threshold, best_f1
    
    def _calculate_precision(self, y_true, y_pred):
        """计算精确率"""
        try:
            from sklearn.metrics import precision_score
            return precision_score(y_true, y_pred, average='binary')
        except ImportError:
            return 0.0
    
    def _calculate_recall(self, y_true, y_pred):
        """计算召回率"""
        try:
            from sklearn.metrics import recall_score
            return recall_score(y_true, y_pred, average='binary')
        except ImportError:
            return 0.0
