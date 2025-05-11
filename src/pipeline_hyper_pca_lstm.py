import os
import hydra
import optuna
import numpy as np
import shutil
import time
from omegaconf import DictConfig, OmegaConf
import logging

# 导入PCA管道函数
from pipline_pca import (
    run_dimensionality_reduction,
    run_prediction,
    run_reconstruction,
    run_evaluation,
    set_global_seeds
)

logger = logging.getLogger(__name__)

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


def objective(trial, cfg, base_output_dir):
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
                logger.warning(f"无法清理 {old_trial_dir}: {e}")
    
    # 2. 从配置文件中获取搜索空间
    search_spaces = cfg.hyperparameter_search.search_spaces
    
    # === PCA参数 ===
    pca_config = cfg.model.dimensionality_reduction.pca
    if hasattr(search_spaces, "pca"):
        pca_space = search_spaces.pca
        
        # 如果有n_components搜索空间
        if hasattr(pca_space, "n_components"):
            target_features = cfg.model.dimensionality_reduction.pca.target_features
            for feature in target_features:
                if hasattr(pca_space.n_components, feature):
                    n_comp = trial.suggest_categorical(
                        f"pca_n_components_{feature}", 
                        pca_space.n_components[feature]
                    )
                    # 直接使用set_nested_value函数而不是尝试字典式赋值
                    set_nested_value(
                        cfg, 
                        f"model.dimensionality_reduction.pca.n_components_{feature}", 
                        n_comp
                    )
    
    # === LSTM超参数 ===
    lstm_config = cfg.model.prediction.lstm_pca
    lstm_space = search_spaces.lstm
    
    lstm_config.hidden_size = trial.suggest_categorical("lstm_hidden_size", lstm_space.hidden_size)
    lstm_config.num_layers = trial.suggest_categorical("lstm_num_layers", lstm_space.num_layers)
    lstm_config.dropout = trial.suggest_categorical("lstm_dropout", lstm_space.dropout)
    lstm_config.sequence_length = trial.suggest_categorical("lstm_sequence_length", lstm_space.sequence_length)
    lstm_config.learning_rate = trial.suggest_categorical("lstm_learning_rate", lstm_space.learning_rate)
    lstm_config.batch_size = trial.suggest_categorical("lstm_batch_size", lstm_space.batch_size)
    lstm_config.patience = trial.suggest_categorical("lstm_patience", lstm_space.patience)
    # 在 objective 函数内部
    external_features_choice = trial.suggest_categorical("lstm_external_features", ["none", "flow"]) # 使用字符串选项

    if external_features_choice == "none":
        lstm_config.external_features = []
    elif external_features_choice == "flow":
        lstm_config.external_features = ["flow"]
    else:
        # 可以添加一个错误处理或默认值
        lstm_config.external_features = []
        logger.warning(f"未知的 external_features_choice: {external_features_choice}, 使用空列表。")

    # 同样的方法处理 input_features
    input_features_choice_str = trial.suggest_categorical("input_features_str", ["salinity_only", "salinity_wind"]) # 使用字符串

    if input_features_choice_str == "salinity_only":
        lstm_config.input_features = ["salinity"]
    elif input_features_choice_str == "salinity_wind":
        lstm_config.input_features = ["salinity", "wind"]
    else:
        lstm_config.input_features = ["salinity"] # 默认值
        logger.warning(f"未知的 input_features_choice_str: {input_features_choice_str}, 使用 ['salinity']。")

    # === 输入源配置 ===
    if hasattr(search_spaces, "input_sources"):
        input_sources = search_spaces.input_sources
        # 检查输入特征配置
        if hasattr(input_sources, "input_features") and len(input_sources.input_features) > 0:
            # 从选项中随机选择输入特征组合
            selected_input_features_idx = trial.suggest_categorical(
                "input_features_idx", 
                list(range(len(input_sources.input_features)))
            )
            lstm_config.input_features = input_sources.input_features[selected_input_features_idx]
            
    # 3. 修改当前trial的输出路径
    cfg.paths.results_subdir = os.path.join(trial_dir, "results")
    cfg.paths.plots_subdir = os.path.join(trial_dir, "plots")
    
    # 4. 运行pipeline
    try:
        # 设置随机种子确保可复现性
        set_global_seeds(cfg.get('random_seed', 42))
        
        # 运行降维
        dr_results = run_dimensionality_reduction(cfg)
        if not dr_results.get('success'):
            logger.error(f"Trial {trial.number}: 降维步骤失败")
            return float('inf')
            
        # 运行预测
        pred_results = run_prediction(cfg, dr_results)
        if not pred_results.get('success'):
            logger.error(f"Trial {trial.number}: 预测步骤失败")
            return float('inf')
            
        # 运行重建
        recon_results = run_reconstruction(cfg, dr_results, pred_results)
        if not recon_results.get('success'):
            logger.error(f"Trial {trial.number}: 重建步骤失败")
            return float('inf')
            
        # 运行评估
        eval_results = run_evaluation(cfg, recon_results, dr_results['method'], pred_results['method'])
        
        # 5. 返回评估指标
        all_eval_results = eval_results.get('evaluation_details', {})
        mean_metric = 0
        valid_splits = 0
        validation_splits = ['val', 'test']  # 只计算验证和测试集的RMSE
        
        for split in validation_splits:
            if split in all_eval_results and 'metrics' in all_eval_results[split]:
                split_metrics = all_eval_results[split]['metrics']
                if 'mean_rmse' in split_metrics and split_metrics['mean_rmse'] is not None:
                    mean_metric += split_metrics['mean_rmse']
                    valid_splits += 1

        # 确保至少有一个有效的分割
        if valid_splits == 0:
            logger.warning(f"Trial {trial.number}: 没有找到有效的RMSE指标")
            return float('inf')  # 表示失败
            
        # 计算平均RMSE
        final_metric = mean_metric / valid_splits
        logger.info(f"Trial {trial.number}: 平均RMSE={final_metric:.6f}")
        return final_metric
        
    except Exception as e:
        logger.error(f"Trial {trial.number} 失败: {e}", exc_info=True)
        return float('inf')  # 出错时返回无穷大

def validate_search_spaces(search_spaces):
    """验证搜索空间配置的合法性"""
    # 检查PCA配置
    pca_space = search_spaces.get("pca", {})
    if not pca_space:
        logger.warning("未找到PCA搜索空间定义")
    else:
        if not hasattr(pca_space, "n_components"):
            logger.warning("PCA搜索空间缺少n_components定义")
    
    # 检查LSTM配置
    lstm_space = search_spaces.get("lstm", {})
    if not lstm_space:
        logger.warning("未找到LSTM搜索空间定义")
    else:
        required_fields = ["hidden_size", "num_layers", "dropout", "sequence_length", 
                          "learning_rate", "batch_size", "patience"]
        for field in required_fields:
            if field not in lstm_space:
                logger.warning(f"LSTM搜索空间缺少 {field} 定义")
            elif not isinstance(lstm_space[field], list):
                logger.warning(f"{field} 应当是一个列表")
    
    # 检查输入源配置
    input_sources = search_spaces.get("input_sources", {})
    if not input_sources:
        logger.warning("未找到输入源搜索空间定义")
    else:
        if not hasattr(input_sources, "input_features"):
            logger.warning("输入源搜索空间缺少input_features定义")

@hydra.main(config_path="../conf", config_name="hyperparameter_search_pca", version_base=None)
def run_hyperparameter_search(cfg: DictConfig) -> None:
    """运行PCA-LSTM超参数搜索"""
    logger.info("=== 开始PCA-LSTM超参数搜索 ===")
    
    # 检查配置文件是否包含搜索空间定义
    if not hasattr(cfg, "hyperparameter_search") or not hasattr(cfg.hyperparameter_search, "search_spaces"):
        raise ValueError("配置文件中缺少hyperparameter_search.search_spaces定义")
    
    # 验证必要的搜索空间是否存在
    search_spaces = cfg.hyperparameter_search.search_spaces
    validate_search_spaces(search_spaces)
    
    # 输出搜索空间信息
    logger.info("搜索空间配置:")
    if hasattr(search_spaces, "pca"):
        logger.info(f"- PCA参数: {OmegaConf.to_yaml(search_spaces.pca)}")
    logger.info(f"- LSTM参数: {OmegaConf.to_yaml(search_spaces.lstm)}")
    if hasattr(search_spaces, "input_sources"):
        logger.info(f"- 输入源参数: {OmegaConf.to_yaml(search_spaces.input_sources)}")
    
    # 设置Optuna研究
    base_output_dir = os.getcwd()  # Hydra自动创建的输出目录
    study_name = f"{cfg.hyperparameter_search.study_name}_{int(time.time())}"
    logger.info(f"创建新的研究: {study_name}")
    
    study = optuna.create_study(
        study_name=study_name,
        direction=cfg.hyperparameter_search.direction,
        storage=cfg.hyperparameter_search.db_storage,
        load_if_exists=False  # 强制创建新的study
    )
    
    # 运行优化
    n_trials = cfg.hyperparameter_search.get("n_trials", 50)
    study.optimize(lambda trial: objective(trial, cfg, base_output_dir), n_trials=n_trials)
    
    # 打印最佳结果
    logger.info("=== 超参数搜索完成 ===")
    logger.info(f"最佳性能: {study.best_value}")
    logger.info("最佳超参数:")
    for key, value in study.best_params.items():
        logger.info(f"  {key}: {value}")
    
    # 保存最佳配置
    best_config = OmegaConf.create(cfg)

    # 使用param_mapping来映射参数
    param_mapping = cfg.hyperparameter_search.pipeline.param_mapping
    for param_name, param_value in study.best_params.items():
        if param_name in param_mapping:
            path_str = param_mapping[param_name]
            success = set_nested_value(best_config, path_str, param_value)
            if not success:
                logger.warning(f"无法设置参数 {param_name} 的值")
        else:
            # 特殊处理input_features_idx
            if param_name == "input_features_idx":
                selected_features = search_spaces.input_sources.input_features[param_value]
                set_nested_value(best_config, "model.prediction.lstm_pca.input_features", selected_features)
            # 特殊处理PCA组件
            elif param_name.startswith("pca_n_components_"):
                feature = param_name.replace("pca_n_components_", "")
                set_nested_value(best_config, f"model.dimensionality_reduction.pca.components.{feature}", param_value)
            else:
                logger.warning(f"参数 {param_name} 没有对应的映射路径")
    
    best_config_path = os.path.join(base_output_dir, "best_config.yaml")
    with open(best_config_path, "w") as f:
        f.write(OmegaConf.to_yaml(best_config))
    logger.info(f"最佳配置已保存至: {best_config_path}")

if __name__ == "__main__":
    run_hyperparameter_search()