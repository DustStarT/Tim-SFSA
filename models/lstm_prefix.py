import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    """一个简单的自注意力模块。"""
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, lstm_output):
        # lstm_output shape: (batch_size, seq_len, hidden_dim)
        # attention_scores shape: (batch_size, seq_len, 1)
        attention_scores = self.attention_net(lstm_output)
        # attention_weights shape: (batch_size, seq_len, 1)
        attention_weights = F.softmax(attention_scores, dim=1)
        # context_vector shape: (batch_size, hidden_dim)
        context_vector = torch.sum(attention_weights * lstm_output, dim=1)
        return context_vector

class LSTMPrefix(nn.Module):
    """
    一个包装器模型，使用可选的双向LSTM和自注意力机制作为特征提取前缀，
    然后将提取的上下文向量与最后一个时间点的原始特征结合，
    送入一个下游的深度生存分析模型。
    支持从预训练分类模型中加载LSTM权重。
    """
    def __init__(self, n_features, config, downstream_model, pretrained_lstm_weights=None):
        """
        初始化模型。
        Args:
            n_features (int): 输入特征的数量。
            config (dict): 包含模型超参数的配置对象。
            downstream_model (nn.Module): 预先实例化的下游模型。
            pretrained_lstm_weights (dict): 预训练的LSTM权重字典。
        """
        super().__init__()
        
        lstm_config = config.model.lstm
        self.use_attention = lstm_config.use_attention
        is_bidirectional = lstm_config.bidirectional
        
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=lstm_config.hidden_size,
            num_layers=lstm_config.get('num_lstm_layers', 1),
            batch_first=True,
            dropout=lstm_config.dropout_rate if lstm_config.get('num_lstm_layers', 1) > 1 else 0,
            bidirectional=is_bidirectional
        )
        
        lstm_output_dim = lstm_config.hidden_size * 2 if is_bidirectional else lstm_config.hidden_size
        
        if self.use_attention:
            self.attention = Attention(lstm_output_dim)
        
        # 可选连接层：规范化/线性变换/激活，用于匹配/增强下游输入
        self.use_connector = False
        connector_cfg = getattr(getattr(config.model, 'two_stage', {}), 'connector', None)
        if connector_cfg is not None and getattr(connector_cfg, 'enabled', False):
            self.use_connector = True
            connector_out = int(getattr(connector_cfg, 'output_dim', lstm_output_dim))
            norm_type = str(getattr(connector_cfg, 'norm', 'none')).lower()
            act_type = str(getattr(connector_cfg, 'activation', 'relu')).lower()
            dropout_p = float(getattr(connector_cfg, 'dropout', 0.0))

            layers = [nn.Linear(lstm_output_dim, connector_out)]
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
            self.connector_out_dim = lstm_output_dim

        self.downstream_model = downstream_model
        
        # 加载预训练LSTM权重
        if pretrained_lstm_weights is not None:
            self.load_pretrained_lstm_weights(pretrained_lstm_weights)
    
    def load_pretrained_lstm_weights(self, pretrained_weights):
        """
        加载预训练的LSTM权重
        
        Args:
            pretrained_weights (dict): 预训练权重字典
        """
        import logging
        logger = logging.getLogger(__name__)
        
        try:
            # 兼容不同保存前缀：可能是 'lstm.' / 'module.lstm.' / 无前缀
            lstm_state_dict = {}
            for name, weight in pretrained_weights.items():
                key = name
                if key.startswith('module.'):
                    key = key[len('module.'):]
                if key.startswith('lstm.'):
                    key = key[len('lstm.'):]
                # 仅收集LSTM相关权重
                if key.startswith(('weight_ih_', 'weight_hh_', 'bias_ih_', 'bias_hh_')):
                    lstm_state_dict[key] = weight

            if len(lstm_state_dict) == 0:
                logger.info("预训练字典中未找到可用的LSTM键，跳过加载。")
                return

            # 仅加载与当前结构 shape 完全匹配的键，避免尺寸/层数不一致报错
            current_state = self.lstm.state_dict()
            filtered = {}
            matched, skipped = 0, 0
            for k, v in lstm_state_dict.items():
                if k in current_state and hasattr(v, 'shape') and hasattr(current_state[k], 'shape') and tuple(v.shape) == tuple(current_state[k].shape):
                    filtered[k] = v
                    matched += 1
                else:
                    skipped += 1

            if matched == 0:
                logger.info("预训练LSTM权重与当前结构无形状匹配项，跳过加载（将使用随机初始化）。")
                return

            missing, unexpected = self.lstm.load_state_dict(filtered, strict=False)
            logger.info(f"部分加载预训练LSTM权重：匹配 {matched} 项，跳过 {skipped} 项。")
            if missing:
                logger.debug(f"LSTM 缺失参数（未加载）: {missing}")
            if unexpected:
                logger.debug(f"LSTM 非预期参数: {unexpected}")

        except Exception as e:
            logger.warning(f"加载预训练LSTM权重失败: {e}")
            logger.info("将使用随机初始化的LSTM权重")
    
    def freeze_lstm(self):
        """冻结LSTM层参数"""
        for param in self.lstm.parameters():
            param.requires_grad = False
        import logging
        logger = logging.getLogger(__name__)
        logger.info("LSTM层已冻结")
    
    def unfreeze_lstm(self):
        """解冻LSTM层参数"""
        for param in self.lstm.parameters():
            param.requires_grad = True
        import logging
        logger = logging.getLogger(__name__)
        logger.info("LSTM层已解冻")

    def forward(self, x):
        # 数值稳定性：对输入进行 NaN/Inf 清洗与裁剪
        try:
            import torch
            if not torch.isfinite(x).all():
                x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            # 读取阈值（若 config 存在）
            sentinel = None
            try:
                conf = getattr(getattr(self, 'downstream_model', None), 'config', None)
                if conf is None:
                    conf = getattr(self, 'config', None)
                if conf is not None:
                    sentinel = float(getattr(conf.data, 'sentinel_threshold', 1e6))
            except Exception:
                sentinel = 1e6
            if sentinel is not None:
                x = torch.clamp(x, -sentinel, sentinel)
        except Exception:
            pass

        # LSTM前缀主forward（在 AMP 下强制用 FP32 进行）
        try:
            from torch.cuda.amp import autocast
            with autocast(enabled=False):
                lstm_out, (h_n, c_n) = self.lstm(x)
        except Exception:
            lstm_out, (h_n, c_n) = self.lstm(x)
        
        # 检查LSTM输出是否有NaN
        if torch.isnan(lstm_out).any():
            print(f'[LSTM-DEBUG] lstm_out contains NaN! mean={lstm_out.mean().item()}, std={lstm_out.std().item()}, min={lstm_out.min().item()}, max={lstm_out.max().item()}, NaN数={(~torch.isfinite(lstm_out)).sum().item()}')
            for name, param in self.lstm.named_parameters():
                print(f'[LSTM-DEBUG] param {name}: mean={param.data.mean().item()}, std={param.data.std().item()}, min={param.data.min().item()}, max={param.data.max().item()}')
        if torch.isnan(h_n).any():
            print(f'[LSTM-DEBUG] h_n contains NaN! mean={h_n.mean().item()}, std={h_n.std().item()}, min={h_n.min().item()}, max={h_n.max().item()}, NaN数={(~torch.isfinite(h_n)).sum().item()}')
        if torch.isnan(c_n).any():
            print(f'[LSTM-DEBUG] c_n contains NaN! mean={c_n.mean().item()}, std={c_n.std().item()}, min={c_n.min().item()}, max={c_n.max().item()}, NaN数={(~torch.isfinite(c_n)).sum().item()}')
        
        # 对 LSTM 输出进行 NaN/Inf 清洗与裁剪
        try:
            if not torch.isfinite(lstm_out).all():
                lstm_out = torch.nan_to_num(lstm_out, nan=0.0, posinf=0.0, neginf=0.0)
            if sentinel is not None:
                lstm_out = torch.clamp(lstm_out, -sentinel, sentinel)
        except Exception:
            pass

        # 获取上下文向量
        if self.use_attention:
            context_vector = self.attention(lstm_out)
            if torch.isnan(context_vector).any():
                print(f'[Attention-DEBUG] context_vector contains NaN! mean={context_vector.mean().item()}, std={context_vector.std().item()}, min={context_vector.min().item()}, max={context_vector.max().item()}, NaN数={(~torch.isfinite(context_vector)).sum().item()}')
        else:
            # 不用attention时直接池化
            context_vector = torch.mean(lstm_out, dim=1)
            if torch.isnan(context_vector).any():
                print(f'[NoAttention-DEBUG] context_vector contains NaN! mean={context_vector.mean().item()}, std={context_vector.std().item()}, min={context_vector.min().item()}, max={context_vector.max().item()}, NaN数={(~torch.isfinite(context_vector)).sum().item()}')
        
        # 确保context_vector的维度正确 [batch_size, hidden_dim]
        if context_vector.dim() == 3:
            context_vector = context_vector.squeeze(0)
        
        # 可选连接层
        if self.use_connector:
            context_vector = self.connector(context_vector)
        
        # 可选：将起始时刻的原始特征与LSTM上下文拼接，作为下游协变量的一部分
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
            # 取起始时刻 t0 的原始观测特征（仅基础特征维度，若存在聚合扩展则剔除）
            # x shape: (batch, seq_len, n_features)
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
            # 拼接: [context | raw_start]
            combined = torch.cat([context_vector, raw_start_features], dim=1)
            output = self.downstream_model(combined)
        else:
            # 仅使用上下文向量
            output = self.downstream_model(context_vector)
        
        # --- 调试输出：下游模型输出分布（受配置开关控制） ---
        try:
            conf = getattr(getattr(self, 'downstream_model', None), 'config', None)
            debug_enabled = False
            if conf is not None:
                # 全局训练期调试开关
                debug_enabled = bool(getattr(getattr(getattr(conf, 'training', object()), 'logging', object()), 'debug_output_stats', False))
                # 第三阶段调试开关（任一为真则输出）
                th_dbg = getattr(getattr(getattr(getattr(conf, 'evaluation', object()), 'time_head', object()), 'debug', None), 'enabled', False)
                debug_enabled = bool(debug_enabled or th_dbg)
        except Exception:
            debug_enabled = False
        if debug_enabled and hasattr(output, 'mean'):
            print(f"[DEBUG][Downstream] output: mean={output.mean().item():.4f}, std={output.std().item():.4f}, min={output.min().item():.4f}, max={output.max().item():.4f}, NaN数={(~torch.isfinite(output)).sum().item()}")
        
        return output 

    def extract_downstream_input(self, x: torch.Tensor) -> torch.Tensor:
        """
        提取送入下游模型的输入表示（在 connector 之后；如配置允许，可与起始原始特征拼接）。
        不经过下游模型，返回形状为 (batch, feature_dim) 的张量。
        训练态/评估态由外部控制：
          - 冻结阶段可在 no_grad + eval 下调用
          - 联合微调阶段需在 train 下调用以保留梯度
        """
        # 数值稳定处理（与 forward 保持一致）
        try:
            if not torch.isfinite(x).all():
                x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            sentinel = None
            try:
                conf = getattr(getattr(self, 'downstream_model', None), 'config', None)
                if conf is None:
                    conf = getattr(self, 'config', None)
                if conf is not None:
                    sentinel = float(getattr(conf.data, 'sentinel_threshold', 1e6))
            except Exception:
                sentinel = 1e6
            if sentinel is not None:
                x = torch.clamp(x, -sentinel, sentinel)
        except Exception:
            pass

        # LSTM 主体
        try:
            from torch.cuda.amp import autocast
            with autocast(enabled=False):
                lstm_out, _ = self.lstm(x)
        except Exception:
            lstm_out, _ = self.lstm(x)

        # 上下文获取（attention 或 mean pool）
        if self.use_attention:
            context_vector = self.attention(lstm_out)
        else:
            context_vector = torch.mean(lstm_out, dim=1)

        # 可选 connector
        if self.use_connector:
            context_vector = self.connector(context_vector)

        # 是否与起始原始特征融合（保持与 forward 同步的判定）
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
            downstream_input = torch.cat([context_vector, raw_start_features], dim=1)
        else:
            downstream_input = context_vector

        return downstream_input