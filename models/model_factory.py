import logging
import torch
from models.coxkan.model import CoxKANModel
from models.deephit.deephit_torch import DeepHitTorch
from models.deephit.deephit_improved import DeepHitImproved
from models.coxph.model import CoxPHModel as CoxPH
from models.deepsurv.model import DeepSurvModel
from models.lstm_prefix import LSTMPrefix
from models.transformer_prefix import TransformerPrefix
from models.transformer_survival import TransformerEncoderDeepSurv
# from models.coxph.model import CoxPH # 为未来扩展预留

# 下游模型的工厂查找表
# 格式: { '模型名称': (模型创建函数, '其配置在文件中的键名') }
_downstream_model_factories = {
    'coxkan': (lambda config, n_features, device: CoxKANModel(config, n_features, device), 'kan'),
    # Ensure DeepHit receives a minimal dict-style config with required keys, including the input feature dim
    'deephit': (
        lambda config, n_features, device: DeepHitImproved({
            'n_features': int(n_features),
            'num_time_bins': int(getattr(config.model.deephit, 'num_time_bins', 24)),
            'num_events': int(getattr(config.model.deephit, 'num_events', 1)),
            'dropout': float(getattr(config.model.deephit, 'dropout', 0.3)),
            'shared_layers': list(getattr(config.model.deephit, 'shared_layers', [128, 64])),
            'risk_specific_layers': list(getattr(config.model.deephit, 'risk_specific_layers', [64]))
        }),
        'deephit'
    ),
    'deepsurv': (lambda config, n_features, device: DeepSurvModel(config, n_features, device), 'deepsurv'),
    'coxph': (lambda config, n_features, device: CoxPH(config), 'coxph')
}

# 主工厂函数
def get_model(config, n_features, device, pretrained_lstm_weights=None):
    """
    根据配置创建模型。
    如果启用LSTM，则创建一个LSTM+下游模型的组合。
    否则，直接创建下游模型。
    
    Args:
        config: 配置对象
        n_features: 特征维度
        device: 计算设备
        pretrained_lstm_weights: 预训练的LSTM权重字典
        
    Returns:
        创建的模型
    """
    model_name = config.model.name.lower()
    use_lstm = getattr(config.model, 'use_lstm', False)
    # 获取编码器类型
    try:
        enc_type = str(getattr(getattr(config.model, 'encoder', {}), 'type', 'lstm')).lower()
    except Exception:
        enc_type = 'lstm'
    
    # 决策逻辑：
    # 1. 如果 use_lstm=False，不使用时序序列，直接创建下游模型
    # 2. 如果 use_lstm=True，使用时序序列，具体类型由 encoder.type 决定
    if not use_lstm:
        use_sequence_model = False
        use_transformer = False
    else:
        use_sequence_model = True
        use_transformer = (enc_type == 'transformer')
    
    logging.info(f"模型工厂决策: use_lstm={use_lstm}, enc_type={enc_type}, use_sequence_model={use_sequence_model}, use_transformer={use_transformer}")

    if model_name not in _downstream_model_factories:
        raise ValueError(f"不支持的模型名称: {model_name}")
    
    # 获取工厂函数和配置键
    downstream_factory, config_key = _downstream_model_factories[model_name]
    
    if use_transformer:
        logging.info(f"创建Transformer+{model_name}模型")
        enc_cfg = getattr(getattr(config.model, 'encoder', {}), 'transformer', {})
        survival_mode = str(getattr(enc_cfg, 'survival_mode', 'prefix')).lower()

        if survival_mode == 'encoder_only':
            model = TransformerEncoderDeepSurv(
                n_features=n_features,
                config=config,
                device=device,
                pretrained_transformer_weights=pretrained_lstm_weights,
            )
        else:
            # 计算下游输入维度（同 LSTMPrefix 逻辑：connector 输出维度 + 可选 raw_start 基础特征维度）
            d_model = int(getattr(enc_cfg, 'd_model', 128))
            connector_cfg = getattr(getattr(config.model, 'two_stage', {}), 'connector', None)
            if connector_cfg is not None and getattr(connector_cfg, 'enabled', False):
                connector_out_dim = int(getattr(connector_cfg, 'output_dim', d_model))
            else:
                connector_out_dim = d_model
            fuse_with_raw_start = False
            if connector_cfg is not None:
                try:
                    fuse_with_raw_start = bool(getattr(connector_cfg, 'fuse_with_raw_start', False))
                except Exception:
                    fuse_with_raw_start = False
            base_feature_dim = n_features
            try:
                include_agg = bool(getattr(getattr(config.data, 'sequence_generation', {}), 'include_aggregated_features', False))
            except Exception:
                include_agg = False
            if include_agg:
                if n_features % 7 == 0:
                    base_feature_dim = n_features // 7
                elif n_features % 5 == 0:
                    base_feature_dim = n_features // 5
            downstream_input_dim = connector_out_dim + (base_feature_dim if fuse_with_raw_start else 0)
            downstream_model = downstream_factory(config, downstream_input_dim, device)
            try:
                setattr(downstream_model, 'config', config)
            except Exception:
                pass
            model = TransformerPrefix(
                n_features=n_features,
                config=config,
                downstream_model=downstream_model,
                pretrained_transformer_weights=pretrained_lstm_weights  # 复用参数名，实际为transformer权重
            )

        # 如未显式提供预训练权重，尝试从配置路径加载
        if pretrained_lstm_weights is None:
            try:
                survival_stage_cfg = getattr(getattr(config.model, 'two_stage', {}), 'survival_stage', None)
                if survival_stage_cfg is not None:
                    pretrained_path = getattr(survival_stage_cfg, 'pretrained_lstm_path', '')
                    if isinstance(pretrained_path, str) and len(pretrained_path) > 0:
                        import torch as _torch
                        ckpt = _torch.load(pretrained_path, map_location='cpu')
                        state_dict = ckpt.get('model_state_dict', ckpt)
                        if survival_mode == 'encoder_only':
                            model.load_pretrained_transformer_weights(state_dict)
                        else:
                            model.load_pretrained_transformer_weights(state_dict)
                        logging.info(f"已从配置路径加载预训练Transformer权重: {pretrained_path}")
            except Exception as e:
                logging.warning(f"从配置路径加载预训练Transformer权重失败: {e}")

        # 如果配置了冻结Transformer，则冻结Transformer层
        if hasattr(config.model.two_stage, 'survival_stage'):
            if getattr(config.model.two_stage.survival_stage, 'freeze_lstm', False):
                if hasattr(model, 'freeze_transformer'):
                    model.freeze_transformer()
    elif use_sequence_model and not use_transformer:
        logging.info(f"创建LSTM+{model_name}模型")
        # 为LSTM创建子配置
        lstm_config = getattr(config.model, 'lstm', {})
        hidden_size = lstm_config.hidden_size
        bidirectional = lstm_config.bidirectional
        # 计算下游输入维度：若启用连接层则以其输出维度为准
        connector_cfg = getattr(getattr(config.model, 'two_stage', {}), 'connector', None)
        if connector_cfg is not None and getattr(connector_cfg, 'enabled', False):
            connector_out_dim = int(getattr(connector_cfg, 'output_dim', hidden_size * (2 if bidirectional else 1)))
        else:
            connector_out_dim = hidden_size * (2 if bidirectional else 1)

        # 如启用“与起始时刻原始特征融合”，需要将下游输入维度加上原始基础特征维度（排除聚合扩展）
        fuse_with_raw_start = True
        if connector_cfg is not None:
            try:
                fuse_with_raw_start = bool(getattr(connector_cfg, 'fuse_with_raw_start', False))
            except Exception:
                fuse_with_raw_start = False
        # 推断基础特征维度：优先按 7x（orig + 6 聚合），其次兼容旧 5x；否则回退为 n_features
        base_feature_dim = n_features
        try:
            include_agg = bool(getattr(getattr(config.data, 'sequence_generation', {}), 'include_aggregated_features', False))
        except Exception:
            include_agg = False
        if include_agg:
            if n_features % 7 == 0:
                base_feature_dim = n_features // 7
            elif n_features % 5 == 0:
                base_feature_dim = n_features // 5
        downstream_input_dim = connector_out_dim + (base_feature_dim if fuse_with_raw_start else 0)
        downstream_model = downstream_factory(config, downstream_input_dim, device)
        # 确保下游模型可访问完整 config（供 LSTMPrefix 在 forward 中读取 connector 配置等）
        try:
            setattr(downstream_model, 'config', config)
        except Exception:
            pass
        model = LSTMPrefix(
            n_features=n_features,
            config=config,
            downstream_model=downstream_model,
            pretrained_lstm_weights=pretrained_lstm_weights
        )
        
        # 如未显式提供预训练权重，尝试从配置路径加载
        if pretrained_lstm_weights is None:
            try:
                survival_stage_cfg = getattr(getattr(config.model, 'two_stage', {}), 'survival_stage', None)
                if survival_stage_cfg is not None:
                    pretrained_path = getattr(survival_stage_cfg, 'pretrained_lstm_path', '')
                    if isinstance(pretrained_path, str) and len(pretrained_path) > 0:
                        import torch as _torch
                        ckpt = _torch.load(pretrained_path, map_location='cpu')
                        state_dict = ckpt.get('model_state_dict', ckpt)
                        model.load_pretrained_lstm_weights(state_dict)
                        logging.info(f"已从配置路径加载预训练LSTM权重: {pretrained_path}")
            except Exception as e:
                logging.warning(f"从配置路径加载预训练LSTM权重失败: {e}")

        # 如果配置了冻结LSTM，则冻结LSTM层
        if hasattr(config.model.two_stage, 'survival_stage'):
            if getattr(config.model.two_stage.survival_stage, 'freeze_lstm', False):
                model.freeze_lstm()
    else:
        logging.info(f"创建{model_name}模型")
        model = downstream_factory(config, n_features, device)
    
    return model 