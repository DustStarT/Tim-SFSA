import logging
from typing import Dict, Optional

import torch
import torch.nn as nn

from models.deepsurv.model import DeepSurvModel


class TransformerEncoderDeepSurv(nn.Module):
    """Transformer 编码器输出上下文，后接 DeepSurv 建模风险。"""

    def __init__(
        self,
        n_features: int,
        config,
        device: torch.device,
        pretrained_transformer_weights: Optional[Dict[str, torch.Tensor]] = None,
    ):
        super().__init__()

        self.config = config
        enc_cfg = getattr(getattr(config.model, 'encoder', {}), 'transformer', {})

        self.d_model = int(getattr(enc_cfg, 'd_model', 128))
        self.nhead = int(getattr(enc_cfg, 'nhead', 4))
        self.num_layers = int(getattr(enc_cfg, 'num_layers', 4))
        self.dim_feedforward = int(getattr(enc_cfg, 'dim_feedforward', 256))
        self.dropout = float(getattr(enc_cfg, 'dropout', 0.1))
        self.pooling = str(getattr(enc_cfg, 'pooling', 'cls')).lower()
        self.use_cls_token = bool(getattr(enc_cfg, 'use_cls_token', True))

        head_hidden_dim = int(getattr(enc_cfg, 'head_hidden_dim', 0))
        head_dropout = float(getattr(enc_cfg, 'head_dropout', self.dropout))
        head_activation = str(getattr(enc_cfg, 'head_activation', getattr(enc_cfg, 'activation', 'gelu'))).lower()

        activation = 'relu' if str(getattr(enc_cfg, 'activation', 'gelu')).lower() == 'relu' else 'gelu'
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            activation=activation,
            batch_first=True,
            norm_first=True,
        )

        norm_name = str(getattr(enc_cfg, 'norm', 'layernorm')).lower()
        if norm_name == 'layernorm':
            final_norm = nn.LayerNorm(self.d_model)
        else:
            final_norm = None

        self.input_proj = nn.Linear(n_features, self.d_model)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers, norm=final_norm)

        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.d_model))
            nn.init.normal_(self.cls_token, std=0.02)

        self.use_head = head_hidden_dim > 0
        if self.use_head:
            head_layers = [nn.Linear(self.d_model, head_hidden_dim)]
            head_layers.append(self._make_activation(head_activation))
            if head_dropout > 0:
                head_layers.append(nn.Dropout(head_dropout))
            self.context_head = nn.Sequential(*head_layers)
            context_dim = head_hidden_dim
        else:
            self.context_head = None
            context_dim = self.d_model

        self.deepsurv = DeepSurvModel(config, context_dim, device)

        if pretrained_transformer_weights is not None:
            self.load_pretrained_transformer_weights(pretrained_transformer_weights)

    @staticmethod
    def _make_activation(name: str) -> nn.Module:
        name = name.lower()
        if name == 'relu':
            return nn.ReLU()
        if name == 'tanh':
            return nn.Tanh()
        if name == 'sigmoid':
            return nn.Sigmoid()
        if name == 'leaky_relu':
            return nn.LeakyReLU()
        return nn.GELU()

    @staticmethod
    def _sinusoidal_pe(seq_len: int, d_model: int, device: torch.device) -> torch.Tensor:
        position = torch.arange(seq_len, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, device=device).float()
            * (-torch.log(torch.tensor(10000.0, device=device)) / d_model)
        )
        pe = torch.zeros(seq_len, d_model, device=device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        context = self._encode_context(x)
        if self.context_head is not None:
            context = self.context_head(context)
        log_risk = self.deepsurv(context)
        return log_risk

    def _encode_context(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)

        try:
            if not torch.isfinite(x).all():
                x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        except Exception:
            pass

        h = self.input_proj(x)

        try:
            pe = self._sinusoidal_pe(h.shape[1], h.shape[2], h.device)
            h = h + pe
        except Exception:
            pass

        if self.use_cls_token:
            cls_tok = self.cls_token.expand(h.size(0), 1, -1)
            h = torch.cat([cls_tok, h], dim=1)

        h = self.encoder(h)
        return self._pool_context(h)

    def _pool_context(self, h: torch.Tensor) -> torch.Tensor:
        if self.pooling == 'cls' and self.use_cls_token:
            return h[:, 0, :]

        start_idx = 1 if self.use_cls_token else 0
        sequence_part = h[:, start_idx:, :]

        if self.pooling == 'max':
            context, _ = torch.max(sequence_part, dim=1)
            return context

        return torch.mean(sequence_part, dim=1)

    def load_pretrained_transformer_weights(self, pretrained_weights: Dict[str, torch.Tensor]):
        logger = logging.getLogger(__name__)
        try:
            current_state = self.state_dict()
            filtered = {}
            missing_keys = []
            unexpected_keys = []

            for key, value in pretrained_weights.items():
                if key in current_state and current_state[key].shape == value.shape:
                    filtered[key] = value
                else:
                    unexpected_keys.append(key)

            for key in current_state.keys():
                if key.startswith('deepsurv'):
                    continue
                if key not in filtered and not key.startswith('context_head'):
                    missing_keys.append(key)

            self.load_state_dict(filtered, strict=False)

            logger.info("成功加载预训练Transformer编码器权重")
            if missing_keys:
                logger.warning(
                    f"缺失的权重键: {missing_keys[:5]}..." if len(missing_keys) > 5 else f"缺失的权重键: {missing_keys}"
                )
            if unexpected_keys:
                logger.warning(
                    f"忽略的权重键: {unexpected_keys[:5]}..." if len(unexpected_keys) > 5 else f"忽略的权重键: {unexpected_keys}"
                )
        except Exception as exc:
            logger.error(f"加载预训练Transformer权重失败: {exc}")
            logger.info("将使用随机初始化的Transformer编码器")

    def freeze_transformer(self):
        for param in self.input_proj.parameters():
            param.requires_grad = False
        for param in self.encoder.parameters():
            param.requires_grad = False
        if self.use_cls_token:
            self.cls_token.requires_grad = False
        logging.getLogger(__name__).info("Transformer编码器已冻结")

    def unfreeze_transformer(self):
        for param in self.input_proj.parameters():
            param.requires_grad = True
        for param in self.encoder.parameters():
            param.requires_grad = True
        if self.use_cls_token:
            self.cls_token.requires_grad = True
        logging.getLogger(__name__).info("Transformer编码器已解冻")

    def get_transformer_weights(self) -> Dict[str, torch.Tensor]:
        weights = {}
        for name, param in self.input_proj.named_parameters():
            weights[f'input_proj.{name}'] = param.data.clone()
        for name, param in self.encoder.named_parameters():
            weights[f'encoder.{name}'] = param.data.clone()
        if self.use_cls_token:
            weights['cls_token'] = self.cls_token.data.clone()
        return weights

    def predict_risk(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            context = self._encode_context(x)
            if self.context_head is not None:
                context = self.context_head(context)
            return self.deepsurv.predict_risk(context)

    def get_risk_scores(self, x: torch.Tensor) -> torch.Tensor:
        return self.predict_risk(x)

