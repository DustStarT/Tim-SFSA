"""
配置验证器
确保所有必要的配置参数都已提供。
"""
import sys

def validate_config(config):
    """
    验证配置字典中是否存在所有必需的键。
    如果缺少任何键，则打印错误并退出。
    """
    required_keys = [
        # 数据处理
        'data.data_dir',
        'data.partitions_to_process',
        'data.specified_features',
        # 特征选择
        'data.feature_selection.n_features',
        # 数据集划分
        'data.train_test_split_ratio',
        'data.random_seed',
        # 子序列生成
        'data.sequence_generation.num_covariate_timesteps',
        'data.sequence_generation.prediction_window_hours',
        'data.sequence_generation.sub_sequence_step',
        # 模型与训练
        'model.name',
        'training.num_epochs',
        'training.batch_size',
        'training.learning_rate',
        'training.device'
    ]

    missing_keys = []
    for key_path in required_keys:
        # 支持嵌套键，例如 'feature_selection.n_features'
        keys = key_path.split('.')
        temp_dict = config
        key_found = True
        for key in keys:
            if isinstance(temp_dict, dict) and key in temp_dict:
                temp_dict = temp_dict[key]
            else:
                key_found = False
                break
        if not key_found:
            missing_keys.append(key_path)

    if missing_keys:
        print("错误：配置文件 'configs/default_config.py' 中缺少以下必需的参数：", file=sys.stderr)
        for key in missing_keys:
            print(f"  - {key}", file=sys.stderr)
        sys.exit(1)

    print("配置验证通过。") 