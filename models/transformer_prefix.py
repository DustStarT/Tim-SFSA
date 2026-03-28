import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerPrefix(nn.Module):
    """
    使用TransformerEncoder作为时间序列前缀，提取上下文向量，
    可选连接层与起始原始特征拼接，然后送入下游生存模型。
    支持预训练权重加载和冻结/解冻功能。
    """
    def __init__(self, n_features, config, downstream_model, pretrained_transformer_weights=None):
        super().__init__()

        self.config = config
        enc_cfg = getattr(getattr(config.model, 'encoder', {}), 'transformer', {})
        d_model = int(getattr(enc_cfg, 'd_model', 128))
        nhead = int(getattr(enc_cfg, 'nhead', 4))
        num_layers = int(getattr(enc_cfg, 'num_layers', 4))
        dim_ff = int(getattr(enc_cfg, 'dim_feedforward', 256))
        dropout = float(getattr(enc_cfg, 'dropout', 0.1))
        act_name = str(getattr(enc_cfg, 'activation', 'gelu')).lower()
        norm_name = str(getattr(enc_cfg, 'norm', 'layernorm')).lower()

        self.input_proj = nn.Linear(n_features, d_model, bias=True)

        if act_name == 'relu':
            activation = 'relu'
        else:
            activation = 'gelu'

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=True
        )

        if norm_name == 'layernorm':
            norm = nn.LayerNorm(d_model)
        elif norm_name == 'batchnorm':
            # 仅在序列维合并前可用，这里还是采用 LayerNorm 更稳妥
            norm = nn.LayerNorm(d_model)
        else:
            norm = None

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, norm=norm)

        # 可选连接层
        self.use_connector = False
        connector_cfg = getattr(getattr(config.model, 'two_stage', {}), 'connector', None)
        if connector_cfg is not None and getattr(connector_cfg, 'enabled', False):
            self.use_connector = True
            connector_out = int(getattr(connector_cfg, 'output_dim', d_model))
            norm_type = str(getattr(connector_cfg, 'norm', 'none')).lower()
            act_type = str(getattr(connector_cfg, 'activation', 'relu')).lower()
            dropout_p = float(getattr(connector_cfg, 'dropout', 0.0))

            layers = [nn.Linear(d_model, connector_out)]
            if norm_type == 'batchnorm':
                layers.append(nn.BatchNorm1d(connector_out))
            elif norm_type == 'layernorm':
                layers.append(nn.LayerNorm(connector_out))
            if act_type == 'relu':
                layers.append(nn.ReLU())
            elif act_type == 'gelu':
                layers.append(nn.GELU())
            elif act_type == 'tanh':
                layers.append(nn.Tanh())
            if dropout_p > 0:
                layers.append(nn.Dropout(dropout_p))
            self.connector = nn.Sequential(*layers)
            self.connector_out_dim = connector_out
        else:
            self.connector_out_dim = d_model

        self.downstream_model = downstream_model
        
        # 加载预训练Transformer权重
        if pretrained_transformer_weights is not None:
            self.load_pretrained_transformer_weights(pretrained_transformer_weights)

    def forward(self, x):
        # x: (batch, seq_len, n_features)
        try:
            if not torch.isfinite(x).all():
                x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        except Exception:
            pass

        h = self.input_proj(x)
        # 简单的正弦位置编码（可选）
        try:
            seq_len = h.shape[1]
            d_model = h.shape[2]
            pe = self._sinusoidal_pe(seq_len, d_model, device=h.device)
            h = h + pe
        except Exception:
            pass

        h = self.encoder(h)
        # 池化：mean pool
        context = torch.mean(h, dim=1)

        if self.use_connector:
            context = self.connector(context)

        # 可选融合起始原始特征
        try:
            connector_cfg = getattr(getattr(self.downstream_model.config.model, 'two_stage', {}), 'connector', None)
        except Exception:
            connector_cfg = None
        fuse_with_raw_start = False
        if connector_cfg is not None:
            try:
                fuse_with_raw_start = bool(getattr(connector_cfg, 'fuse_with_raw_start', False))
            except Exception:
                fuse_with_raw_start = False

        if fuse_with_raw_start:
            raw_start_all = x[:, 0, :]
            raw_start_features = raw_start_all
            try:
                n_total = raw_start_all.shape[-1]
                conf = getattr(getattr(self, 'downstream_model', None), 'config', None)
                include_agg = False
                if conf is not None:
                    include_agg = bool(getattr(getattr(conf.data, 'sequence_generation', {}), 'include_aggregated_features', False))
                if include_agg and n_total > 0:
                    if n_total % 7 == 0:
                        base_dim = n_total // 7
                        raw_start_features = raw_start_all[:, :base_dim]
                    elif n_total % 5 == 0:
                        base_dim = n_total // 5
                        raw_start_features = raw_start_all[:, :base_dim]
            except Exception:
                pass
            if raw_start_features.dim() == 1:
                raw_start_features = raw_start_features.unsqueeze(0)
            combined = torch.cat([context, raw_start_features], dim=1)
            return self.downstream_model(combined)
        else:
            return self.downstream_model(context)

    @staticmethod
    def _sinusoidal_pe(seq_len, d_model, device):
        # (seq_len, d_model)
        position = torch.arange(seq_len, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, device=device).float() * (-torch.log(torch.tensor(10000.0, device=device)) / d_model))
        pe = torch.zeros(seq_len, d_model, device=device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def load_pretrained_transformer_weights(self, pretrained_weights):
        """
        加载预训练的Transformer权重，处理键值不匹配问题
        
        Args:
            pretrained_weights (dict): 预训练权重字典
        """
        import logging
        logger = logging.getLogger(__name__)
        
        try:
            # 创建当前模型的权重字典
            current_state_dict = self.state_dict()
            
            # 过滤预训练权重，只保留匹配的键
            filtered_weights = {}
            missing_keys = []
            unexpected_keys = []
            
            for key, value in pretrained_weights.items():
                if key in current_state_dict:
                    # 检查形状是否匹配
                    if current_state_dict[key].shape == value.shape:
                        filtered_weights[key] = value
                    else:
                        logger.warning(f"Shape mismatch for key {key}: current {current_state_dict[key].shape} vs pretrained {value.shape}")
                        missing_keys.append(key)
                else:
                    unexpected_keys.append(key)
            
            # 检查缺失的键
            for key in current_state_dict.keys():
                if key not in filtered_weights and not key.startswith('downstream_model'):
                    missing_keys.append(key)
            
            # 加载过滤后的权重
            self.load_state_dict(filtered_weights, strict=False)
            
            logger.info(f"成功加载预训练Transformer权重")
            if missing_keys:
                logger.warning(f"缺失的权重键: {missing_keys[:5]}..." if len(missing_keys) > 5 else f"缺失的权重键: {missing_keys}")
            if unexpected_keys:
                logger.warning(f"意外的权重键: {unexpected_keys[:5]}..." if len(unexpected_keys) > 5 else f"意外的权重键: {unexpected_keys}")
                
        except Exception as e:
            logger.error(f"加载预训练权重失败: {e}")
            logger.info("将使用随机初始化的权重")
    
    def freeze_transformer(self):
        """冻结Transformer层参数"""
        for param in self.input_proj.parameters():
            param.requires_grad = False
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        import logging
        logger = logging.getLogger(__name__)
        logger.info("Transformer层已冻结")
    
    def unfreeze_transformer(self):
        """解冻Transformer层参数"""
        for param in self.input_proj.parameters():
            param.requires_grad = True
        for param in self.encoder.parameters():
            param.requires_grad = True
        
        import logging
        logger = logging.getLogger(__name__)
        logger.info("Transformer层已解冻")


