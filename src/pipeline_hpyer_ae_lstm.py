import os
import hydra
import optuna
import numpy as np
import shutil
from omegaconf import DictConfig, OmegaConf
import time
from pipeline_ae_lstm import (
    load_processed_data, 
    run_ae_on_salinity, 
    run_wind_processing, 
    run_flow_processing, 
    run_lstm_prediction,
    run_salinity_reconstruction,
    run_evaluation
)
# 建议实现更通用的set_nested_value函数
def set_nested_value(config_dict, path_str, value):
    """安全地在嵌套的OmegaConf配置中设置值，支持数组索引"""
    # 将数组表示法转换: hidden_layers.0 → hidden_layers[0]
    import re
    path_str = re.sub(r'(\w+)\.(\d+)', r'\1[\2]', path_str)
    
    path_parts = path_str.split('.')
    current = config_dict
    
    # 导航到最终属性的父级
    for i, part in enumerate(path_parts[:-1]):
        # 处理数组索引
        if '[' in part and ']' in part:
            base_name = part.split('[')[0]
            idx = int(part.split('[')[1].split(']')[0])
            current = getattr(current, base_name)[idx]
        else:
            current = getattr(current, part)
    
    # 设置最终值
    last_part = path_parts[-1]
    if '[' in last_part and ']' in last_part:
        base_name = last_part.split('[')[0]
        idx = int(last_part.split('[')[1].split(']')[0])
        getattr(current, base_name)[idx] = value
    else:
        setattr(current, last_part, value)
    return True


def objective(trial, cfg, all_processed_data, base_output_dir):
    """优化目标函数，用于超参数搜索"""
    # 1. 为当前试验创建子目录
    trial_dir = os.path.join(base_output_dir, f"trial_{trial.number}")
    os.makedirs(trial_dir, exist_ok=True)

    # 清理旧试验目录以节省磁盘空间
    if trial.number > 10:  # 只保留最近的试验
        old_trial_dir = os.path.join(base_output_dir, f"trial_{trial.number-10}")
        if os.path.exists(old_trial_dir):
            try:
                shutil.rmtree(old_trial_dir)
            except Exception as e:
                print(f"警告: 无法清理 {old_trial_dir}: {e}")
    # 2. 从配置文件中获取搜索空间
    search_spaces = cfg.hyperparameter_search.search_spaces
    
    # === AE超参数 ===
    ae_config = cfg.model.dimensionality_reduction.autoencoder
    ae_space = search_spaces.ae
    
    # 对于离散值列表使用suggest_categorical
    if isinstance(ae_space.encoding_dim, list):
        ae_config.encoding_dim = trial.suggest_categorical("ae_encoding_dim", ae_space.encoding_dim)
    else:
        # 如果未指定为列表，则使用默认范围
        ae_config.encoding_dim = trial.suggest_int("ae_encoding_dim", 8, 24)
    
    # 处理隐藏层配置(嵌套列表)
    ae_config.hidden_layers = []
    for i, layer_choices in enumerate(ae_space.hidden_layers):
        layer_size = trial.suggest_categorical(f"ae_hidden_layer{i+1}", layer_choices)
        ae_config.hidden_layers.append(layer_size)
    
    # 其他AE参数
    ae_config.dropout_rate = trial.suggest_categorical("ae_dropout", ae_space.dropout_rate)
    ae_config.learning_rate = trial.suggest_categorical("ae_learning_rate", ae_space.learning_rate)
    ae_config.batch_size = trial.suggest_categorical("ae_batch_size", ae_space.batch_size)
    ae_config.epochs = trial.suggest_categorical("ae_epochs", ae_space.epochs)
    ae_config.activation = trial.suggest_categorical("ae_activation", ae_space.activation)
    # # 在objective函数中修改激活函数处理
    # if hasattr(ae_space, "activation"):
    #     # 检查激活函数的类型
    #     if isinstance(ae_space.activation, list):
    #         activations = [str(act) for act in ae_space.activation]
    #         ae_config.activation = trial.suggest_categorical("ae_activation", activations)
    #     else:
    #         # 如果不是列表，则直接赋值
    #         ae_config.activation = ae_space.activation    

    # === LSTM超参数 ===
    lstm_config = cfg.model.prediction.lstm_ae
    lstm_space = search_spaces.lstm
    
    lstm_config.hidden_size = trial.suggest_categorical("lstm_hidden_size", lstm_space.hidden_size)
    lstm_config.num_layers = trial.suggest_categorical("lstm_num_layers", lstm_space.num_layers)
    lstm_config.dropout = trial.suggest_categorical("lstm_dropout", lstm_space.dropout)
    lstm_config.sequence_length = trial.suggest_categorical("lstm_sequence_length", lstm_space.sequence_length)
    lstm_config.learning_rate = trial.suggest_categorical("lstm_learning_rate", lstm_space.learning_rate)
    lstm_config.batch_size = trial.suggest_categorical("lstm_batch_size", lstm_space.batch_size)
    lstm_config.patience = trial.suggest_categorical("lstm_patience", lstm_space.patience)
    
    # === 输入源配置 ===
    if hasattr(search_spaces, "input_sources"):
        input_sources = search_spaces.input_sources
        # 盐度是必须的
        cfg.model.prediction.input_sources.use_salinity = True
        
        if hasattr(input_sources, "use_wind") and len(input_sources.use_wind) > 1:
            cfg.model.prediction.input_sources.use_wind = trial.suggest_categorical(
                "use_wind", input_sources.use_wind
            )
            
        if hasattr(input_sources, "use_flow") and len(input_sources.use_flow) > 1:
            cfg.model.prediction.input_sources.use_flow = trial.suggest_categorical(
                "use_flow", input_sources.use_flow
            )
    
    # 3. 修改当前trial的输出路径
    cfg.paths.results_subdir = os.path.join(trial_dir, "results")
    cfg.paths.plots_subdir = os.path.join(trial_dir, "plots")
    cfg.paths.models_subdir = os.path.join(trial_dir, "models")
    cfg.paths.ae_model_path = os.path.join(trial_dir, "models/ae_model.pth")
    cfg.paths.lstm_model_path = os.path.join(trial_dir, "models/lstm_model.pth")
    
    # 4. 运行pipeline
    try:
        ae_results = run_ae_on_salinity(cfg, all_processed_data)
        if not ae_results.get('success'):
            return float('inf')  # 失败时返回无穷大（如果是最小化目标）
            
        wind_results = run_wind_processing(cfg)
        
        # 可选：处理径流数据
        flow_results = None
        if cfg.model.prediction.input_sources.get("use_flow", False):
            flow_results = run_flow_processing(cfg)
        
        pred_results = run_lstm_prediction(cfg, ae_results, wind_results, flow_results)
        if not pred_results.get('success'):
            return float('inf')
            
        rec_results = run_salinity_reconstruction(cfg, ae_results, pred_results)
        eval_results = run_evaluation(cfg, rec_results, all_processed_data, ae_results)
        
        # 5. 返回评估指标
        eval_details = eval_results.get('evaluation_details', {}) 
        mean_metric = 0
        valid_splits = 0
        validation_splits = ['val', 'test'] # 只计算验证和测试集的RMSE
        for split in validation_splits:
            split_metrics = eval_details[split].get('metrics', {})
            if 'mean_rmse' in split_metrics and split_metrics['mean_rmse'] is not None:
                mean_metric += split_metrics['mean_rmse']
                valid_splits += 1

        # 确保至少有一个有效的分割
        if valid_splits == 0:
            print(f"Trial {trial.number}: 没有找到有效的RMSE指标")
            return float('inf')  # 表示失败
            
        # 计算平均RMSE
        return mean_metric / valid_splits
        
    except Exception as e:
        print(f"Trial {trial.number} failed: {e}")
        return float('inf')  # 出错时返回无穷大（如果是最小化目标）

def validate_search_spaces(search_spaces):
    """验证搜索空间配置的合法性"""
    # 检查AE配置
    ae_space = search_spaces.get("ae", {})
    if not ae_space:
        print("警告: 未找到AE搜索空间定义")
    else:
        required_fields = ["encoding_dim", "hidden_layers", "dropout_rate", "learning_rate", "batch_size", "epochs"]
        for field in required_fields:
            if field not in ae_space:
                print(f"警告: AE搜索空间缺少 {field} 定义")
            elif not isinstance(ae_space[field], list) and field != "hidden_layers":
                print(f"警告: {field} 应当是一个列表")
    
    # 检查LSTM配置
    lstm_space = search_spaces.get("lstm", {})
    if not lstm_space:
        print("警告: 未找到LSTM搜索空间定义")
    else:
        required_fields = ["hidden_size", "num_layers", "dropout", "sequence_length", 
                          "learning_rate", "batch_size", "patience"]
        for field in required_fields:
            if field not in lstm_space:
                print(f"警告: LSTM搜索空间缺少 {field} 定义")
            elif not isinstance(lstm_space[field], list):
                print(f"警告: {field} 应当是一个列表")

@hydra.main(config_path="../conf", config_name="hyperparameter_search", version_base=None)
def run_hyperparameter_search(cfg: DictConfig) -> None:
    print("=== 开始超参数搜索 ===")
    
    # 检查配置文件是否包含搜索空间定义
    if not hasattr(cfg, "hyperparameter_search") or not hasattr(cfg.hyperparameter_search, "search_spaces"):
        raise ValueError("配置文件中缺少hyperparameter_search.search_spaces定义")
    
    # 验证必要的搜索空间是否存在
    search_spaces = cfg.hyperparameter_search.search_spaces
    if not hasattr(search_spaces, "ae"):
        raise ValueError("缺少AE搜索空间配置")
    if not hasattr(search_spaces, "lstm"):
        raise ValueError("缺少LSTM搜索空间配置")
    
    # 输出搜索空间信息
    print("搜索空间配置:")
    print(f"- AE参数: {OmegaConf.to_yaml(search_spaces.ae)}")
    print(f"- LSTM参数: {OmegaConf.to_yaml(search_spaces.lstm)}")
    
    # 1. 加载数据（只加载一次，所有trial共用）
    all_processed_data = load_processed_data(cfg)
    
    # 2. 设置Optuna研究
    base_output_dir = os.getcwd()  # Hydra自动创建的输出目录
    # 在创建study的地方
    study_name = f"{cfg.hyperparameter_search.study_name}_{int(time.time())}"
    print(f"创建新的研究: {study_name}")
    study = optuna.create_study(
        study_name=study_name,
        direction=cfg.hyperparameter_search.direction,
        storage=cfg.hyperparameter_search.db_storage,
        load_if_exists=False  # 强制创建新的study
    )
    
    # 3. 运行优化
    n_trials = cfg.hyperparameter_search.get("n_trials", 50)
    study.optimize(lambda trial: objective(trial, cfg, all_processed_data, base_output_dir), n_trials=n_trials)
    
    # 4. 打印最佳结果
    print("=== 超参数搜索完成 ===")
    print(f"最佳性能: {study.best_value}")
    print("最佳超参数:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # 5. 保存最佳配置
    best_config = OmegaConf.create(cfg)

    # 使用param_mapping来映射参数
    param_mapping = cfg.hyperparameter_search.pipeline.param_mapping
    for param_name, param_value in study.best_params.items():
        if param_name in param_mapping:
            path_str = param_mapping[param_name]
            if 'hidden_layers.0' in path_str:
                # 将 hidden_layers.0 转换为 hidden_layers[0]
                path_str = path_str.replace('hidden_layers.0', 'hidden_layers[0]')
            if 'hidden_layers.1' in path_str:
                # 将 hidden_layers.1 转换为 hidden_layers[1]
                path_str = path_str.replace('hidden_layers.1', 'hidden_layers[1]')
                
            success = set_nested_value(best_config, path_str, param_value)
            if not success:
                print(f"警告: 无法设置参数 {param_name} 的值")
        else:
            print(f"警告: 参数 {param_name} 没有对应的映射路径")
    
    best_config_path = os.path.join(base_output_dir, "best_config.yaml")
    with open(best_config_path, "w") as f:
        f.write(OmegaConf.to_yaml(best_config))
    print(f"最佳配置已保存至: {best_config_path}")

if __name__ == "__main__":
    run_hyperparameter_search()