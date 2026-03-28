"""
Transformer 分类模型与训练器
用于两阶段训练的第一阶段：在设置 encoder.type='transformer' 时，完全替换 LSTM 分类器
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import numpy as np
from typing import Tuple, Dict, Any

from models.losses import FocalLoss

logger = logging.getLogger(__name__)


class TransformerClassifier(nn.Module):
    """
    使用 TransformerEncoder 进行时序表征，再接分类头输出二分类 logits。
    """

    def __init__(self,
                 input_size: int,
                 d_model: int = 128,
                 nhead: int = 4,
                 num_layers: int = 4,
                 dim_feedforward: int = 256,
                 dropout: float = 0.2,
                 activation: str = 'gelu',
                 norm: str = 'layernorm',
                 num_classes: int = 2,
                 use_cls_token: bool = True,
                 pooling: str = 'cls'):
        super().__init__()

        self.input_size = input_size
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.activation = activation
        self.norm = norm
        self.num_classes = num_classes
        self.use_cls_token = bool(use_cls_token)
        self.pooling = str(pooling).lower()

        self.input_proj = nn.Linear(input_size, d_model, bias=True)

        act = 'gelu' if str(activation).lower() != 'relu' else 'relu'
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=act,
            batch_first=True,
            norm_first=True
        )

        if str(norm).lower() == 'layernorm':
            final_norm = nn.LayerNorm(d_model)
        else:
            final_norm = None

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, norm=final_norm)
        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.normal_(self.cls_token, std=0.02)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU() if act == 'gelu' else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )

    @staticmethod
    def _sinusoidal_pe(seq_len: int, d_model: int, device: torch.device) -> torch.Tensor:
        position = torch.arange(seq_len, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, device=device).float() * (-torch.log(torch.tensor(10000.0, device=device)) / d_model))
        pe = torch.zeros(seq_len, d_model, device=device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # (1, seq_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_size)
        try:
            if not torch.isfinite(x).all():
                x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        except Exception:
            pass

        # 输入投影
        h = self.input_proj(x)
        
        # 添加位置编码
        try:
            pe = self._sinusoidal_pe(h.shape[1], h.shape[2], h.device)
            h = h + pe
        except Exception:
            pass
        
        # 添加CLS token
        if self.use_cls_token:
            cls_tok = self.cls_token.expand(h.size(0), 1, -1)
            h = torch.cat([cls_tok, h], dim=1)
        
        # Transformer编码
        h = self.encoder(h)
        
        # 池化
        if self.pooling == 'cls' and self.use_cls_token:
            context = h[:, 0, :]
        else:
            start = 1 if self.use_cls_token else 0
            context = torch.mean(h[:, start:, :], dim=1)
        
        # 分类头
        logits = self.classifier(context)
        
        # 最终检查
        if not torch.isfinite(logits).all():
            logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)
        
        return logits

    def get_transformer_weights(self) -> Dict[str, torch.Tensor]:
        """
        获取Transformer权重用于后续生存分析模型
        
        Returns:
            Transformer权重字典
        """
        transformer_weights = {}
        # 提取input_proj权重
        for name, param in self.input_proj.named_parameters():
            transformer_weights[f'input_proj.{name}'] = param.data.clone()
        
        # 提取encoder权重
        for name, param in self.encoder.named_parameters():
            transformer_weights[f'encoder.{name}'] = param.data.clone()

        if getattr(self, 'use_cls_token', False):
            transformer_weights['cls_token'] = self.cls_token.data.clone()
        
        return transformer_weights


class TransformerClassifierTrainer:
    """
    Transformer 分类训练器
    """

    def __init__(self, model: TransformerClassifier, config: Any, device: torch.device, class_weights: torch.Tensor = None):
        self.model = model
        self.config = config
        self.device = device
        self.model.to(device)

        cls_cfg = config.model.two_stage.classification_stage
        self.num_epochs = cls_cfg.num_epochs
        self.learning_rate = cls_cfg.learning_rate
        self.batch_size = cls_cfg.batch_size
        self.patience = cls_cfg.early_stopping_patience

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        
        loss_cfg = getattr(cls_cfg, 'loss', {}) or {}
        loss_type = str(getattr(loss_cfg, 'type', 'cross_entropy')).lower()
        self.loss_type = loss_type

        try:
            ls = float(getattr(cls_cfg, 'label_smoothing', 0.1))
        except Exception:
            ls = 0.1

        if loss_type == 'focal':
            alpha = float(getattr(loss_cfg, 'alpha', 0.25))
            gamma = float(getattr(loss_cfg, 'gamma', 2.0))
            self.criterion = FocalLoss(alpha=alpha, gamma=gamma)
            if class_weights is not None:
                logger.warning("FocalLoss 当前不支持类别权重，传入的 class_weights 将被忽略。")
            logger.info(f"使用 FocalLoss(alpha={alpha}, gamma={gamma}) 作为分类损失")
        else:
            if class_weights is not None:
                class_weights = class_weights.to(device)
                self.criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=ls)
                logger.info(f"使用类别权重的交叉熵损失: {class_weights.cpu().numpy()} (label_smoothing={ls})")
            else:
                self.criterion = nn.CrossEntropyLoss(label_smoothing=ls)
                logger.info(f"使用未加权交叉熵损失 (label_smoothing={ls})")

        # 动态学习率调度：warmup + cosine + plateau
        try:
            warmup_epochs = int(getattr(cls_cfg, 'warmup_epochs', 15))
            scheduler_type = getattr(cls_cfg, 'scheduler_type', 'cosine_warmup')
        except Exception:
            warmup_epochs = 15
            scheduler_type = 'cosine_warmup'
        
        total_epochs = max(1, int(self.num_epochs))
        
        if scheduler_type == 'cosine_warmup':
            # Warmup + Cosine Annealing
            cosine_epochs = max(1, total_epochs - warmup_epochs)
            def lr_lambda(epoch):
                if epoch < warmup_epochs:
                    return (epoch + 1) / max(1, warmup_epochs)
                t = epoch - warmup_epochs
                return 0.5 * (1 + torch.cos(torch.tensor(t / max(1, cosine_epochs) * 3.1415926535)))
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_lambda)
            
        elif scheduler_type == 'step_warmup':
            # Warmup + Step Decay
            def lr_lambda(epoch):
                if epoch < warmup_epochs:
                    return (epoch + 1) / max(1, warmup_epochs)
                return 0.5 ** ((epoch - warmup_epochs) // 50)
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_lambda)
            
        elif scheduler_type == 'plateau':
            # ReduceLROnPlateau with warmup
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='max', factor=0.5, patience=15, min_lr=1e-6, verbose=True
            )
            self.use_plateau_scheduler = True
        elif scheduler_type == 'none':
            # 禁用学习率调度
            self.scheduler = None
            self.use_plateau_scheduler = False
        else:
            # Default: cosine_warmup
            cosine_epochs = max(1, total_epochs - warmup_epochs)
            def lr_lambda(epoch):
                if epoch < warmup_epochs:
                    return (epoch + 1) / max(1, warmup_epochs)
                t = epoch - warmup_epochs
                return 0.5 * (1 + torch.cos(torch.tensor(t / max(1, cosine_epochs) * 3.1415926535)))
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_lambda)
        
        self.use_plateau_scheduler = getattr(self, 'use_plateau_scheduler', False)

        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.val_f1_scores = []
        self.val_auc_scores = []
        self.learning_rates = []  # 记录学习率变化
        self.best_val_auc = 0.0
        self.epochs_no_improve = 0
        self.best_model_state = None
        self.best_threshold = 0.5
        self.best_threshold_f1 = 0.0

    def train_epoch(self, train_loader) -> Tuple[float, float]:
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for data, target in train_loader:
            data, target = data.to(self.device), target.to(self.device)
            
            # 检查输入数据
            if not torch.isfinite(data).all():
                data = torch.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            
            # 检查输出
            if not torch.isfinite(output).all():
                output = torch.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0)
            
            loss = self.criterion(output, target)
            
            # 检查损失
            if not torch.isfinite(loss):
                logger.warning("NaN loss detected, skipping batch")
                continue
            
            loss.backward()
            
            # 更温和的梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
            
            self.optimizer.step()
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
        avg_loss = total_loss / len(train_loader)
        accuracy = 100.0 * correct / total
        return avg_loss, accuracy

    def _calculate_f1(self, y_true, y_pred):
        try:
            from sklearn.metrics import f1_score
            return f1_score(y_true, y_pred, average='binary')
        except Exception:
            return 0.0

    def _calculate_auc(self, y_true, y_probs):
        try:
            from sklearn.metrics import roc_auc_score
            return roc_auc_score(y_true, y_probs)
        except Exception:
            return 0.0

    def validate_epoch(self, val_loader) -> Tuple[float, float, float, float]:
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
                probs = F.softmax(output, dim=1)
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())
        avg_loss = total_loss / len(val_loader)
        accuracy = 100.0 * correct / total
        f1 = self._calculate_f1(all_targets, all_preds)
        auc = self._calculate_auc(all_targets, all_probs)
        return avg_loss, accuracy, f1, auc

    def train(self, train_loader, val_loader, output_dir: str = None):
        logger.info(f"开始训练Transformer分类模型，共{self.num_epochs}个epoch")
        for epoch in range(self.num_epochs):
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc, val_f1, val_auc = self.validate_epoch(val_loader)
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            self.val_f1_scores.append(val_f1)
            self.val_auc_scores.append(val_auc)
            
            # 动态学习率调度
            if self.scheduler is not None:
                if self.use_plateau_scheduler:
                    self.scheduler.step(val_auc)  # ReduceLROnPlateau 需要监控指标
                else:
                    self.scheduler.step()  # LambdaLR 按 epoch 调度
            
            # 记录当前学习率
            current_lr = self.optimizer.param_groups[0]['lr']
            self.learning_rates.append(current_lr)
            
            if val_auc > self.best_val_auc:
                self.best_val_auc = val_auc
                self.epochs_no_improve = 0
                self.best_model_state = self.model.state_dict().copy()
                if output_dir and self.config.model.two_stage.classification_stage.save_best_model:
                    model_path = f"{output_dir}/classification_best_model.pth"
                    torch.save(self.best_model_state, model_path)
                    logger.info(f"保存最佳模型到: {model_path}")
            else:
                self.epochs_no_improve += 1
            if epoch % 10 == 0 or epoch == self.num_epochs - 1:
                logger.info(
                    f"Epoch {epoch+1}/{self.num_epochs}: "
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, "
                    f"Val F1: {val_f1:.4f}, Val AUC: {val_auc:.4f}, "
                    f"LR: {current_lr:.2e}"
                )
            if self.epochs_no_improve >= self.patience:
                logger.info(f"早停：验证AUC在{self.patience}个epoch内未提升")
                break
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            logger.info("已加载最佳模型权重")
        logger.info(f"训练完成，最佳验证AUC: {self.best_val_auc:.4f}")

        # 基于验证集确定最佳阈值
        self.best_threshold, self.best_threshold_f1 = self._determine_best_threshold(val_loader)
        logger.info(
            "Transformer 分类器基于验证集的最佳阈值: %.4f (F1=%.4f)",
            self.best_threshold,
            self.best_threshold_f1
        )

    def evaluate(self, test_loader) -> Dict[str, float]:
        self.model.eval()
        all_preds, all_targets, all_probs = [], [], []
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
        accuracy = 100.0 * sum(p == t for p, t in zip(all_preds, all_targets)) / len(all_targets)
        try:
            from sklearn.metrics import precision_score, recall_score
            precision = precision_score(all_targets, all_preds, average='binary')
            recall = recall_score(all_targets, all_preds, average='binary')
        except Exception:
            precision = 0.0
            recall = 0.0
        return {
            'accuracy': accuracy,
            'f1_score': self._calculate_f1(all_targets, all_preds),
            'auc_score': self._calculate_auc(all_targets, all_probs),
            'precision': precision,
            'recall': recall,
            'threshold': getattr(self, 'best_threshold', 0.5)
        }

    def _determine_best_threshold(self, val_loader) -> Tuple[float, float]:
        """在验证集上搜索最佳概率阈值。"""
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


