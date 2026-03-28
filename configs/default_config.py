"""
默认配置文件
包含所有模型和数据处理的默认参数设置
"""
import copy

import torch
from easydict import EasyDict

_C = EasyDict()
cfg = _C
# ---------------------------------------------------------------------------
# 运行模式（优先级与行为说明）
# ---------------------------------------------------------------------------
# 高层控制：将首选设置 `_C.run_mode`，这是主入口 `main()` 读取并用于设置运行流程的字段。
#  - 当通过 `python main.py` 或调用 `main()` 时，`main()` 会根据 `_C.run_mode` 自动设置
#    `_C.data.cross_validation.enabled`：
#      'train_only' / 'test_only' -> cross_validation.enabled = False
#      'cross_validation'         -> cross_validation.enabled = True
#      'two_stage'                -> 先训练分类模型，再微调生存分析模型
#      'classification_only'      -> 仅训练分类模型
#      'time_head_only'           -> 仅训练时间头time_head_only
#
# 直接调用 `run_experiment(config)` 时，函数会直接读取 `_C.data.cross_validation.enabled`
# 来判断是否执行交叉验证流程。因此行为取决于调用路径：
#  - 使用 `main()` / 脚本入口：以 `_C.run_mode` 为准（`main()` 会覆盖 `data.cross_validation.enabled`）。
#  - 直接调用 `run_experiment`：以 `data.cross_validation.enabled` 的当前值为准。
#
# 建议：在常规使用下仅修改 `_C.run_mode`（高层配置），避免手动同时设置两个字段产生歧义；
# 如需强制覆盖，可在调用 `run_experiment` 之前显式设置 `config.data.cross_validation.enabled`。
# run_mode: 'train_only' | 'test_only' | 'cross_validation' | 'two_stage' | 'classification_only' | 'time_head_only'
_C.run_mode = 'two_stage'  # 默认公开发布入口：两阶段训练

# ---------------------------------------------------------------------------
# 数据和特征（输入接口）
# 这些配置用于数据查找、划分、子序列生成与特征级处理
# ---------------------------------------------------------------------------
_C.data = EasyDict()
_C.data.cross_validation = EasyDict(enabled=False, n_splits=3, shuffle=True)
_C.data.data_dir = 'data/SWAN'
_C.data.partitions_to_process = [1, 2, 3, 4, 5]
_C.data.include_nf_data = False
_C.data.min_sequence_length = 5  # 序列的最小时间点数

# --- 特征工程与选择 ---
_C.data.specified_features = [
    'TOTUSJH', 'TOTBSQ', 'TOTPOT', 'TOTUSJZ', 'ABSNJZH', 'SAVNCPP',
    'USFLUX', 'TOTFZ', 'MEANPOT', 'EPSZ', 'MEANSHR', 'SHRGT45', 'MEANGAM',
    'MEANGBT', 'MEANGBZ', 'MEANGBH', 'MEANJZH', 'TOTFY', 'MEANJZD', 'MEANALP',
    'TOTFX', 'EPSY', 'EPSX', 'R_VALUE'
]

# 特征选择（可选）：开启后 pipeline 将尝试降维/选择 n_features
_C.data.feature_selection = EasyDict(
    enabled=True,
    n_features=20,
    vif_threshold=180.0, 
)

# 特征分布漂移过滤：在训练开始前可选地剔除在训练/测试间分布差异过大的特征
_C.data.feature_shift = EasyDict(
    enabled=True,              # 是否启用基于分布差异的特征剔除
    metric='median_mad',        # 目前仅支持 'median_mad'（|med_train-med_test| / (MAD_train+eps)）
    threshold=3.0,             # 当该比值 > threshold 时剔除该特征（默认 3 MADs）
    min_test_samples=50,       # 计算统计量时，被比较的（验证/测试）集中至少要有多少条时序点才启用该检测
    compare_to='val'           # 比较目标：'val' (在训练集内切出验证用于比较) or 'test' (直接使用 test set 进行比较)
)

# feature_shift 扩展：重复比较与处理策略
_C.data.feature_shift.repeats = 5          # 在训练内多次随机切分验证进行比较的次数（越高越稳健）
_C.data.feature_shift.drop_fraction = 0.6  # 当某特征在 repeats 中超过该比例被判定漂移，则对其执行剔除/处理
_C.data.feature_shift.action = 'drop'      # 对漂移特征的处理策略: 'drop' | 'mask' | 'winsorize'
                                           # - 'drop' 将在训练时完全移除该特征
                                           # - 'mask' 将在训练中把该特征置为 NaN（由 imputer 填充），保留列结构
                                           # - 'winsorize' 将在训练时基于训练分位进行截断（推荐，软化极端值）
_C.data.feature_shift.winsorize_lower_pct = 0.01  # 下界分位（例如 0.01 -> 1%）
_C.data.feature_shift.winsorize_upper_pct = 0.99  # 上界分位（例如 0.99 -> 99%）

# 简单变换开关（例如对数、标准化等可在预处理模块实现）
_C.data.feature_transformation = EasyDict(enabled=False)


# 数据清洗/离群点检测策略（仅配置；实现由 data pipeline 使用）
_C.data.outlier_config = EasyDict(
    method='robust',       # 'robust' | 'zscore'
    strategy='none',       # 'none' | 'clip' | 'remove' （pipeline 解读此字段）
    detection_params=dict(iqr_multiplier=1.5, zscore_threshold=3.5),
)


# 数据划分与采样
_C.data.train_test_split_ratio = 0.8
_C.data.validation_ratio = 0.2
_C.data.random_seed = 2345  # 参考历史最佳
# 全局随机种子  
_C.seed = 2345  # 参考历史最佳
# 是否启用验证集。设置为 False 时训练流程将不再从训练集中切出验证子集，
# 早停和基于验证集的监控将被禁用；这在数据量极小或想要最大化训练数据时有用。
_C.data.use_validation = True

# group-based 划分可以按父样本(record_id/AR)做分组，避免数据泄露
# split_mode 可选： 'subsequence_independent' | 'group' | 'group_balanced'
_C.data.use_group_split = True
# 数据划分模式: 可选值:
#  - 'subsequence_independent' : 先将所有父样本展开为子序列，然后按子序列独立划分训练/测试（子序列视为独立个体）
#  - 'group_balanced' : 按归一化 record_id(AR) 做 group 划分，同时尝试在 train/test 之间平衡事件比例
#  - 'group' : 按归一化 record_id(AR) 做 group 划分（原始行为）
# 默认保留原始 group 行为
_C.data.split_mode = 'group_balanced'

# 当 split_mode='group_balanced' 时，尝试在最多 N 次随机划分中以此容忍度平衡事件率
_C.data.group_balance_tolerance = 0.02  # 复刻实验设置
_C.data.group_balance_max_tries = 50

# 当使用按父样本划分的 temporal 子序列策略时，控制每个父样本的训练子序列比例与容忍度
_C.data.temporal_parent_train_fraction = _C.data.train_test_split_ratio
_C.data.temporal_parent_balance_tolerance = 0.05
_C.data.temporal_parent_balance_max_tries = 10


# 子序列生成（影响模型输入的时间窗口与时间分辨率）
_C.data.sequence_generation = EasyDict(
    num_covariate_timesteps=20,    # 历史窗口长度（时间步数）
    prediction_window_hours=48,    # 预测窗口（小时）
    sub_sequence_step=5,           # 采样步长（子序列滑动步长）
)

# 是否在每个子序列附加聚合特征，增加稳定性或信息量
# 新版：每个基础特征将扩展为 6 个统计量：self(last)、mean、var、std、max、min
# 下游维度 = 原始维度 + 6×原始维度（按时间步广播后拼接）
_C.data.sequence_generation.include_aggregated_features = False  # 参考历史最佳：不使用聚合特征

# --- 模型选择与配置 ---
_C.model = EasyDict()
_C.model.name = 'DeepSurv'            # 用于选择下游模型
_C.model.use_lstm = False              # 在输入端使用时序序列（LSTM 或 Transformer）
_C.model.downstream_model = 'deepsurv' # 与 name 对应的更具体实现

# 两阶段训练配置
_C.model.two_stage = EasyDict(
    enabled=True,                    # 是否启用两阶段训练（为复刻/改进实验默认启用）
    classification_stage=EasyDict(
        enabled=False,                 # 启用分类阶段训练（复刻原始分类模型）
        num_epochs=250,
        learning_rate=0.0001,
        batch_size=128,
        early_stopping_patience=120,
        save_best_model=True,
        model_save_path='',
        sampling_mode='oversample',         # 'none' | 'undersample' | 'oversample' | 'smote'
        sampling_params=EasyDict(
            undersample_ratio=1.0,     # 降低欠采样比例，让模型看到更多无事件样本
            random_state=_C.seed,
        ),
        label_smoothing=0.0,          # 减少正类过度平滑
        use_class_weights=False,
        loss=EasyDict(
            type='focal',
            alpha=0.2,
            gamma=3.0
        ),
    ),
    survival_stage=EasyDict(
        enabled=True,                 # 是否训练生存分析阶段（启用微调以复刻最佳实验）
        load_pretrained_lstm=True,   # 是否加载预训练的LSTM/Transformer权重（根据encoder.type自动适配）
        freeze_lstm=False,            # 不冻结LSTM
        fine_tune_epochs=50,         # 调整为更长的微调轮次以进行完整长训练（来自短网格的最佳候选）
        fine_tune_lr=1e-5,           # 微调学习率（允许更快收敛，同时对LSTM使用更小的lstm_lr）
        # 指定从分类阶段导出的checkpoint路径（.pth）；若为空则使用传入权重或默认行为
        # 指向已知表现最好的分类模型（便于复刻）
        pretrained_lstm_path='',
        # 冻结-解冻策略：前 N 个 epoch 冻结 LSTM/Transformer，之后自动解冻
        freeze_first_n_epochs=0,  # 先不冻结，让下游能较快适配，必要时可调整为较大值后再微调
        # 对无事件样本进行欠采样以提高事件信息密度（有助于C-index提升）
        undersample_non_events=True,
        undersample_ratio=0.8,          # 调整为网格最佳候选：非事件欠采样比（0.8）
        undersample_random_state=_C.seed,
        # 新增：时间感知的样本平衡策略（暂时禁用，训练/验证分布不匹配导致性能下降）
        balance_samples=EasyDict(
            enabled=False,  # 禁用样本平衡
            target_event_ratio=0.22,  # 如启用，建议轻度平衡（16%→22%）
            time_bins=5,  # 时间分层数
            preserve_high_risk_censored=True,  # 优先保留高风险删失样本
            random_state=_C.seed,
        ),
    ),
    # LSTM/Transformer 与下游生存模型之间的可选连接层（参考历史最佳：启用）
    connector=EasyDict(
        enabled=True,  # 启用connector层
        output_dim=512,  # 与分类模型一致：Connector 输出维度应匹配分类器的输入（512）以便加载预训练权重
        norm='layernorm',
        activation='gelu',
        dropout=0.4,     # 增加dropout以对抗更大网络容量的过拟合
        fuse_with_raw_start=True,
    )
)

# LSTM 前缀配置
_C.model.lstm = EasyDict(
    # Restored to match the pretrained classification LSTM checkpoint
    hidden_size=256,
    num_lstm_layers=7,
    dropout_rate=0.3,
    bidirectional=True,
    use_attention=True,
)

# Transformer配置优化
_C.model.encoder = EasyDict(
    type='lstm',                 # 'lstm' | 'transformer'
    transformer=EasyDict(
        d_model=96,                    # 适度提高表示能力
        nhead=4,                       # 注意力头数需整除 d_model
        num_layers=5,                  # 减少层数
        dim_feedforward=192,           # 调整FFN维度
        dropout=0.3,                   # 参考历史最佳connector：适度dropout
        activation='gelu',
        norm='layernorm',
        use_cls_token=True,
        pooling='cls',                 # 'cls' | 'mean' | 'max'
        survival_mode='encoder_only',        # 'prefix' 使用 Prefix + 下游模型, 'encoder_only' 使用 Transformer encoder + DeepSurv
        head_hidden_dim=32,            # encoder_only 模式下风险头隐藏层维度（<=0 表示仅线性层）
        head_dropout=0.3,              # encoder_only 模式下风险头dropout
        head_activation='gelu'
    )
)

# KAN (可选) 模型参数
_C.model.kan = EasyDict(
    layers=[32, 16],
    grid_size=5,
    spline_order=3,
    dropout=0.3,
    scale_noise=0.02,
    scale_base=0.5,
    grid_range=[-2, 2],
    base_activation='torch.nn.SiLU',
    use_base_update=True,
    loss_alpha=1.0,
    loss_sigma=0.5,
)

# DeepHit 参数（若使用 DeepHit 分支）
_C.model.deephit = EasyDict(
    shared_layers=[128, 64],
    risk_specific_layers=[64],
    num_time_bins=24,
    num_events=1,
    dropout=0.3,
    loss_alpha=0.5,
    loss_sigma=0.3,
    use_batch_norm=True,
    use_residual=True,
)

# DeepSurv 下游模型参数
_C.model.deepsurv = EasyDict(
    hidden_layers=[256, 128],  # 提升网络容量以拟合更复杂风险函数
    dropout_rate=0.4,  # 适度增加dropout以抑制过拟合
    batch_norm=True,
    activation='relu',  # 复刻实验设置
    l2_reg=0.0001,  # 减小L2强度以配合更深网络（同时启用weight_decay）
    focal_alpha=0.3,
    focal_gamma=1.5,
)

# 可选：在 test-only 模式下从已有 checkpoint 加载模型（支持文件路径或包含 .pth 的 run 目录）
# 如果为 None，默认行为仍然是按照 run_mode 执行（test_only 会按历史逻辑训练后再测试）
_C.model.checkpoint_path = ''


# ---------------------------------------------------------------------------
# 训练与优化
# ---------------------------------------------------------------------------
_C.training = EasyDict()
_C.training.device = 'cuda' if torch.cuda.is_available() else 'cpu'
_C.training.num_epochs = 1000  # 复刻实验设置
_C.training.batch_size = 64
_C.training.learning_rate = 2e-4  # 复刻实验设置
_C.training.early_stopping_patience = 40  # 防止过拟合，及时停止
_C.training.gradient_clipping = 1.0  # 复刻实验设置
_C.training.mixed_precision = True
_C.training.validation_frequency = 1
_C.training.seed = _C.seed

# Balanced batching：控制 batch 中事件样本比例
_C.training.enable_balanced_batching = True
_C.training.target_event_ratio_per_batch = 0.5  # 复刻实验设置
_C.training.min_events_per_batch = 1

# 是否将 sample_weights 应用于训练损失（可能影响训练/验证损失幅度）
_C.training.apply_sample_weights_to_loss = False

# 损失函数配置
_C.training.loss = EasyDict(
    deepsurv_loss_type='combined',
    cox_weight=2.0,  # 提高 Cox 部分权重，强调生存时间回归
    focal_weight=0.1,  # 减少 focal 对训练的影响
    ranking_weight=0.5,  # 提高 ranking 权重以增强排序能力
    # ranking loss variant for DeepSurv combined loss: 'hinge' | 'ipcw_pairwise' | 'logistic'
    ranking_variant='ipcw_pairwise',
    ranking_margin=0.0,  # 复刻实验设置
    focal_alpha=0.25,
    focal_gamma=2.0,
)

_C.training.optimizer = EasyDict(name='AdamW', weight_decay=1e-4, betas=(0.9, 0.999), eps=1e-8)  # 参考历史最佳

# 显式为LSTM参数组配置独立学习率（为空则使用主学习率的0.1倍）
_C.training.optimizer.lstm_lr = 1e-5  # 从网格搜索结果设置：恢复为最佳 lstm_lr=1e-5
_C.training.optimizer.transformer_lr = 2e-5  # Transformer也给予更大的微调步长

# 学习率调度（两套：简单的 ReduceLROnPlateau 与 多模式 scheduler 参考）
_C.training.lr_scheduler = EasyDict(
    enabled=True,
    type='ReduceLROnPlateau',
    mode='max',
    factor=0.5,  # 复刻实验设置
    patience=20,  # 复刻实验设置
    min_lr=1e-9,  # 复刻实验设置
    verbose=True,
    warmup_epochs=10,
)

_C.training.scheduler = EasyDict(name=None, T_0=50, T_mult=2, eta_min=1e-9)

# 训练稳定性相关设置
_C.training.stability = EasyDict(
    grad_clip_norm=1.0,             # 复刻实验设置
    # 是否启用自适应裁剪（根据梯度范数动态调整clip值）
    adaptive_clip_enabled=True,
    # NaN/Inf 处理与损失修复
    replace_input_nan=True,         # 将输入中的NaN替换为0
    replace_logits_nan=True,        # 将logits中的NaN替换为0
    loss_nan_repair=True,           # 当loss为NaN时使用回退损失
    loss_scale_max=50.0,            # 超过该值按比例缩放损失
    min_loss_epsilon=1e-6,          # 过低时添加的最小常数
    # 梯度异常缓解（自动LR降低 + 清零梯度）
    grad_mitigate_enabled=True,
    grad_mitigate_threshold_norm=50.0,  # 复刻实验设置
    grad_mitigate_lr_factor=0.1,         # 复刻实验设置
    grad_mitigate_min_lr=1e-9,          # 复刻实验设置
    grad_mitigate_cooldown_steps=200      # 复刻实验设置
)

# 训练期结构化约束（例如关键特征单调性）
_C.training.constraints = EasyDict(
    monotonic=EasyDict(
        enabled=False,  # 复刻实验设置（禁用单调约束）
        # 直接通过特征下标指定需要单调的特征（针对输入到模型的张量维度）。
        # 若启用 LSTM，则按最后一个时间步的特征做单调约束。
        # 推荐用特征名以避免特征移除/重排后的错位
        feature_names=[
            'TOTUSJH','TOTPOT','ABSNJZH','SAVNCPP','MEANPOT','MEANSHR','SHRGT45','MEANJZH','MEANJZD','R_VALUE'
        ],
        feature_indices=[],
        # 对应每个 feature_indices 的方向，+1 表示单调递增，-1 表示单调递减；
        # 留空(None)表示均为 +1。
        directions=None,
        lambda_penalty=0.03,
    )
)

# 训练日志相关控制
_C.training.logging = EasyDict(
    batch_log_interval=20,          # 每多少个batch记录一次梯度/输出统计
    log_output_stats=True,
    log_grad_stats=False,
    log_event_ratio=True,
    debug_data_shapes=False,
)

# 显式的正则化项（与优化器weight_decay解耦）
_C.training.regularization = EasyDict(
    enable_l1=False,
    enable_l2=True,
    l1_lambda=1e-4,
    l2_lambda=1e-4,
)

_C.training.model_averaging = EasyDict(
    enabled=True,            # 复刻实验设置（启用EMA）
    method='ema',             # 'ema' | 'swa'
    ema_decay=0.995,          # EMA衰减
    swa_start_epoch=100,      # SWA开始轮次
    swa_freq=5                # SWA更新频率
)


# ---------------------------------------------------------------------------
# 输出、日志与可视化
# ---------------------------------------------------------------------------
_C.results_dir = 'results'         # 根结果目录
_C.verbose_logging = True

_C.visualization = EasyDict(
    save_plots=True,
    plot_formats=['png', 'pdf'],
    plot_dpi=200,
    plot_show=False,              # headless 环境默认不弹窗
    save_feature_distribution=True,
    save_curve_data_csv=True,     # 在无法绘图时备份 CSV
)


# dataloader 默认项（会被数据加载器读取，可在运行时覆盖）
_C.dataloader = EasyDict(batch_size=_C.training.batch_size, num_workers=4, shuffle=True, pin_memory=True, prefetch_factor=2)


# ---------------------------------------------------------------------------
# 评估与额外可视化控制
# ---------------------------------------------------------------------------
_C.evaluation = EasyDict()
_C.evaluation.brier_score_num_bins = 24
_C.evaluation.bootstrap_ci = 1000
_C.evaluation.cindex_iqr_multiplier = 3.0  # 预处理C-index时的IQR倍数
_C.evaluation.predictions_are_risk = True  # 明确控制是否将模型输出视为风险（越大越早事件）
_C.evaluation.cindex_preprocess_enabled = False  # 统一主开关（关闭IQR剔除）
_C.evaluation.use_cindex_preproc = False       # 兼容旧字段，保持与上行一致（默认关闭）

# predictions_are_risk: True 表示模型输出为 risk score（值越大风险越高/越早事件）
_C.evaluation.predictions_are_risk = True

# 风险分数校准配置
# 注意：pycox库本身不提供风险分数校准方法，我们使用sklearn的标准校准方法
# （IsotonicRegression, LinearRegression, LogisticRegression），这些是业界标准方法
# 
# ⚠️ 警告：风险分数校准可能降低性能，特别是当原始风险分数与事件时间相关性很弱时。
# 建议：对于生存分析，优先使用时间预测校准（Calibrated方法），而不是风险分数校准。
# 只有在确认风险分数校准能改善性能时才启用。
_C.evaluation.risk_score_calibration = EasyDict(
    enabled=False,    # 是否启用风险分数校准（当前模型风险分数相关性很弱-0.03，校准反而降低性能）
    method='isotonic',  # 校准方法：'isotonic'（推荐，保序回归，保持排序关系）,
                        #           'linear'（线性回归校准）,
                        #           'platt'（Platt scaling，逻辑回归）
    # 通用超参数
    hyperparams=EasyDict(
        # 通用参数
        general=EasyDict(
            # 如果原始风险分数与事件时间的相关性绝对值<weak_correlation_threshold，将自动跳过校准
            # 注意：当前模型相关性约-0.03，远低于此阈值，建议禁用校准或提高阈值
            weak_correlation_threshold=0.15,  # 相关性阈值，低于此值将跳过校准（提高至0.15以更严格）
            min_samples=20,                   # 最小有效样本数，低于此值将跳过校准（提高要求）
            min_events=10,                    # 最小事件样本数，低于此值将跳过校准（提高要求）
        ),
        # Isotonic Regression 超参数
        isotonic=EasyDict(
            out_of_bounds='clip',  # 超出范围的处理方式：'clip'（裁剪到范围），'nan'（设为NaN），'raise'（抛出异常）
            y_min=None,            # 输出最小值限制（None表示不限制）
            y_max=None,            # 输出最大值限制（None表示不限制）
        ),
        # Linear Regression 超参数
        linear=EasyDict(
            fit_intercept=True,    # 是否拟合截距项
            copy_X=True,           # 是否复制X数据
            n_jobs=None,           # 并行作业数（None表示使用所有CPU）
            positive=False,        # 是否强制系数为正（仅适用于某些solver）
        ),
        # Platt Scaling (Logistic Regression) 超参数
        platt=EasyDict(
            penalty='l2',          # 正则化类型：'l1', 'l2', 'elasticnet', 'none'
            C=1.0,                 # 正则化强度的倒数（越小正则化越强）
            solver='newton-cholesky',        # 优化算法：'lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga'
            max_iter=1000,         # 最大迭代次数
            tol=1e-5,              # 收敛容差
            n_jobs=None,           # 并行作业数（None表示使用所有CPU）
            warm_start=False,      # 是否使用上一次拟合的结果作为初始化
        ),
    ),
)

# 是否生成 evaluator 中的额外图像（由 plot_modes 控制具体类别）
_C.evaluation.plot_extra_visuals = True
_C.evaluation.plot_modes = EasyDict(
    enable_ar_subsample_colormap=True,
    enable_cumulative_risk_examples=True,
    enable_risk_timecourse_examples=True,
    enable_hazard_examples=True,
    # 每分钟非累计风险图（基于生存曲线差分的每分钟事件概率）
    enable_per_minute_risk=True,
    per_minute_risk_log_scale=True,
    per_minute_risk_smooth_minutes=None,  # 可选：整数分钟窗口；None/0 关闭
    per_minute_risk_aggregate_minutes=5,  # 可选：按K分钟求和（幅值增大），如 5/15/60
    # 新增：事件 vs 删失 的逐时间点分离可视化
    enable_timewise_separation=True,
    timewise_separation_smooth=False,
    # hazard curves display controls
    hazard_scale=1000.0,          # 把每小时风险率按1000倍显示（每千人小时）
    hazard_clip_q=99.5,           # 上分位裁剪，抑制极端尖峰
    hazard_smooth_window=None,    # 可选：平滑窗口大小(奇数)，None=自适应
    hazard_show_cumhaz=True,      # 同时输出累计风险 H(t)
)

# 预测方法对比功能配置
_C.evaluation.prediction_methods_comparison = EasyDict(
    enabled=True,                    # 是否启用预测方法对比
    output_subdir='prediction_methods_comparison',  # 输出子目录名称
    methods_to_compare=[             # 要对比的预测方法
        'least_squares',
    ],
    plot_settings=EasyDict(
        x_axis_limit=None,           # 横轴最大时间（None=自动使用prediction_window_hours）
        figure_size=(12, 10),        # 图像尺寸
        dpi=300,                     # 图像分辨率
        save_formats=['png', 'pdf'], # 保存格式
    ),
    performance_metrics=[            # 要计算的性能指标
        'mae', 'rmse', 'mape', 'bias', 'correlation', 'r2'
    ],
    generate_separate_plots=True,    # 是否生成分开的对比图
    generate_performance_table=True, # 是否生成性能对比表格
    generate_explanation_doc=True,   # 是否生成方法说明文档
    generate_time_head_style_outputs=True,  # 是否为每种方法生成与time_head相同风格的可视化
    default_split_name='test',        # time-head风格输出的默认数据集标记
    
    # 预处理配置：提升预测稳定性
    preprocessing=EasyDict(
        enforce_monotonic=True,      # 强制生存曲线单调递减
        smoothing_window=2,          # 平滑窗口大小
        normalize_to_unit_interval=True,  # 将每条生存曲线按行缩放到[0,1]
    ),
    
    # 最小二乘法配置 (无需额外参数，自动拟合)
    least_squares=EasyDict(),
    
    # 按活动区的可视化输出
    generate_ar_analysis=True,       # 是否生成按活动区的分析
    ar_analysis=EasyDict(
        min_samples_per_ar=3,       # 每个活动区最少样本数（少于此数不单独分析）
        plot_all_samples=False,       # 是否绘制所有子样本（不仅是事件样本）
        save_individual_plots=True,  # 是否为每个活动区保存单独的图表
    ),
)

_C.evaluation.fixed_auc_time_quantiles = [0.2, 0.4, 0.6, 0.8]
_C.evaluation.log_val_tauc = True  # 在训练验证阶段计算并记录时间依赖ROC的均值

# 第三阶段：时间预测（使用传统方法，基于生存函数）
_C.evaluation.time_head = EasyDict(
    enabled=True,               # 是否启用第三阶段时间预测
    
    # # === 传统预测方法配置 ===
    # traditional_method=EasyDict(
    #     threshold=0.5,          # 生存概率阈值（用于确定预测时间）
    #     # 预测方法：找到生存函数首次低于阈值的时间点
    #     # 不考虑删失数据，仅评估事件样本
    # ),
    
    # # === 评估配置 ===
    # eval_events_only=True,      # 仅评估事件样本（不考虑删失）
    # save_predictions=True,       # 保存预测与指标
    # clip_pred_to_window=True,   # 预测值裁剪到 [0, prediction_window]
    
    # # === 已废弃的配置（保留以防向后兼容） ===
    # # 注意：以下配置仅用于向后兼容，实际已不再使用
    # use_risk_score=True,         # 已废弃
    # use_last_step_features=True, # 已废弃
    # use_classification_prob=False, # 已废弃
    # layers=[16, 8],              # 已废弃
    # dropout=0.3,                 # 已废弃
    # loss='smooth_l1',            # 已废弃
    # l2_reg=5e-4,                # 已废弃
    # epochs=0,                    # 已废弃
    # batch_size=128,              # 已废弃
    # lr=1e-3,                     # 已废弃
    
    # # === 调试配置 ===
    # debug=EasyDict(
    #     enabled=True,            # 是否输出详细日志
    # ),
    
    # # === standalone配置（用于time_head_only模式） ===
    # standalone=EasyDict(
    #     enabled=False,          # 总开关（禁用standalone模式，改用传统方法）
    #     base_model_checkpoint_path='',  # 不再需要checkpoint路径
    #     dataset_split='test',           # 评估分割：'test' | 'train'
    #     freeze_epochs=0,        # 已废弃
    #     joint_epochs=0,         # 已废弃
    #     base_lr=0,              # 已废弃
    #     base_weight_decay=0     # 已废弃
    # )
)

# ---------------------------------------------------------------------------
# 超参搜索保留字段（接口）
# ---------------------------------------------------------------------------
_C.SEARCH = EasyDict()
_C.SEARCH.ENABLED = False
_C.SEARCH.METRIC_TO_OPTIMIZE = 'c_index'
_C.SEARCH.SEARCH_SPACE = {}


def get_config():
    """返回默认配置对象的深拷贝。

    注意：代码内读取配置时假定字段存在并使用默认值；如果需要动态修改，建议
    在调用 run_experiment 前修改返回的对象。
    """
    return copy.deepcopy(_C)
