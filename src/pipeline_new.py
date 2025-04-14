# src/run_pipeline.py (概念结构)
import os
import sys
import logging
import time
import hydra
import random  # Added import for random module
from omegaconf import DictConfig, OmegaConf, listconfig
from typing import Optional, Dict, Any, List
import numpy as np
import joblib
import torch
import shutil # 确保导入 shutil 用于文件复制

# --- 项目设置 和 import ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils.hydra_config import DrprConfig
from src.evaluation.visualization import (
    plot_time_series_comparison,
    plot_spatial_rmse,
    plot_spatial_comparison_at_timestep, # 新增：绘制特定时间点对比图
    plot_spatial_difference,            # 新增：绘制特定时间点差异图
    plot_spatial_statistic,             # 新增：绘制通用空间统计图
    calculate_correlation_map,          # 新增：计算相关性图的函数
    generate_evaluation_report,         # 新增：生成评估报告的函数
    plot_spatial_distribution            # 如果需要 heatmap 也导入
)
from src.utils.model_utils import get_device_from_config  
from src.dimensionality_reduction.som_pytorch import SOMTorch
from src.utils.data_loader import load_raw_data, load_mask, load_scaler, load_split_indices

# --- 导入训练函数 ---
from src.training.train_som import train_single_feature_som, train_combined_feature_som
from src.training.train_hmm import main as train_hmm_main
from src.training.train_lstm import train_and_predict_lstm
from src.training.train_pca import train_and_transform_pca
from src.training.train_autoencoder import train_and_transform_ae
from src.reconstruction.reconstruct_bmu import reconstruct_from_bmu # 用于基于 SOM 的重建

logger = logging.getLogger(__name__)

def set_global_seeds(seed):
    if seed is None:
        logger.warning("未提供随机种子，结果可能不可重现")
        return 
    logger.info(f"设置全局随机种子: {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # 使用确定性算法
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        logger.info("已设置CUDA随机种子")

    logger.info("全局随机种子设置完成")

# === Pipeline 各阶段的辅助函数 ===

def run_dimensionality_reduction(cfg: DictConfig) -> Dict[str, Any]:
    """运行所选的降维方法。"""
    config = DrprConfig.from_hydra_config(cfg)
    method = cfg.model.dimensionality_reduction.method
    logger.info(f"--- 阶段：维度降低 (方法: {method}) ---")
    results = {'method': method, 'success': False} # 初始化 success 为 False
    # 这个字典将存储最终传递给预测阶段的低维数据路径
    # 结构: {feature_name: {split: {'positions': path}}}
    low_dim_data_paths_for_pred = {}

    # --- 定义降维所需的路径 (通用) ---
    target_field_dr = cfg.model.dimensionality_reduction.get("target_feature", "salinity") # Target for state DR
    dr_model_dir = os.path.join(config.paths.models_base_dir, method)
    os.makedirs(dr_model_dir, exist_ok=True)
    transformed_data_dir = os.path.join(config.paths.processed_data_dir, f"{method}_{target_field_dr}_low_dim")
    os.makedirs(transformed_data_dir, exist_ok=True)

    # --- High-dim paths (needed for PCA/AE) ---
    high_dim_paths = {}
    for split in ["train", "val", "test"]:
        try:
            path = os.path.join(config.paths.processed_data_dir, f"{split}_{target_field_dr}_processed.npy")
            if os.path.exists(path):
                high_dim_paths[split] = path
        except Exception as e:
            logger.warning(f"无法构造 {split} 的高维数据路径 ({target_field_dr}): {e}")

    # --- Specific DR Method Logic ---
    if method == 'som':
        som_state_results = None
        som_obs_results = None
        state_bmu_positions = {} # 临时存储 {split: path}
        obs_bmu_positions = {}   # 临时存储 {split: path}

        # --- 步骤 1: 训练状态 SOM (总是需要) ---
        logger.info(f"训练状态 SOM ({target_field_dr})...")
        som_state_results = train_single_feature_som(cfg, feature_name=target_field_dr)
        logger.info(f"状态 SOM 返回结果: {som_state_results}") # 打印详细结果

        if som_state_results and isinstance(som_state_results, dict):
            results['som_state'] = som_state_results # 存储完整的状态 SOM 结果
            results['model_path'] = som_state_results.get('model_path') # 状态 SOM 模型用于重建
            results['target_feature'] = target_field_dr # 记录降维的目标特征名
            bmu_paths_state = som_state_results.get('bmu_indices_paths', {})

            if bmu_paths_state:
                try:
                    # 提取 {split: path} 格式的状态 BMU 位置路径
                    state_bmu_positions = {split: p['positions'] for split, p in bmu_paths_state.items() if isinstance(p, dict) and 'positions' in p and p['positions']}
                    logger.info(f"提取的状态 BMU 位置路径: {state_bmu_positions}")
                    if not state_bmu_positions: # 如果提取结果为空字典
                         raise ValueError("未能从状态 SOM 结果中提取有效的 BMU 位置路径。")
                    # 将状态 BMU 路径按所需格式添加到 low_dim_data_paths_for_pred
                    low_dim_data_paths_for_pred[target_field_dr] = {
                        split: {'positions': path} for split, path in state_bmu_positions.items()
                    }
                except Exception as e:
                    logger.error(f"从状态 SOM 结果提取或格式化 BMU 位置路径时出错: {e}", exc_info=True)
                    # results['success'] 已经是 False，直接返回
                    return results
            else:
                logger.error("状态 SOM 结果中未找到 'bmu_indices_paths' 或其为空。")
                # results['success'] 已经是 False，直接返回
                return results
        else:
            logger.error("状态 SOM 训练失败或未返回有效字典。")
            # results['success'] 已经是 False，直接返回
            return results

        # --- 步骤 2: 检查是否需要观测 SOM ---
        needs_obs_som = False
        observation_features = []
        pred_method = cfg.model.prediction.method
        obs_feature_name = None # 初始化

        if pred_method == 'hmm':
            needs_obs_som = True
            observation_features = list(cfg.model.prediction.hmm.get('observation_features', []))
            if not observation_features:
                 logger.error("HMM 预测方法已选择，但未在配置中指定 observation_features。")
                 return results # 配置错误
            obs_feature_name = "_".join(sorted(observation_features))

        elif pred_method == 'lstm' and cfg.model.prediction.lstm.get('use_observation_features', False):
             needs_obs_som = True
             observation_features = list(cfg.model.prediction.lstm.get('observation_features', []))
             if not observation_features:
                  logger.error("LSTM 配置了 use_observation_features=True 但未提供 observation_features。")
                  return results # 配置错误
             obs_feature_name = "_".join(sorted(observation_features))
        # 添加 elif 用于其他需要观测 SOM 的预测器

        # --- 步骤 3: 如果需要，训练观测 SOM ---
        if needs_obs_som:
            logger.info(f"训练观测 SOM ({obs_feature_name}) for {pred_method}...")

            # --- (临时设置观测 SOM 的 map_size，如果配置不同) ---
            original_map_size = cfg.training.som.map_size
            map_size_obs = cfg.training.som.get("map_size_obs", original_map_size)
            if map_size_obs != original_map_size:
                logger.info(f"临时为观测 SOM 设置地图大小: {map_size_obs}")
                try:
                    # 使用 OmegaConf.set_struct 防止意外添加新键，如果需要灵活性则设为 False
                    # OmegaConf.set_struct(cfg.training.som, False)
                    cfg.training.som.map_size = map_size_obs # 假设 map_size 是 list 或 ListConfig
                    # 如果是 list，需要转换: cfg.training.som.map_size = list(map_size_obs)
                except Exception as e:
                     logger.error(f"修改观测 SOM 地图大小时出错: {e}")
                     # 可能需要更健壮的配置修改方式，例如创建副本或使用 with 语句

            # --- 调用 SOM 训练 ---
            if len(observation_features) == 1:
                som_obs_results = train_single_feature_som(cfg, feature_name=observation_features[0])
            else: # 组合特征
                som_obs_results = train_combined_feature_som(cfg, output_feature_name=obs_feature_name)

            # --- (恢复原始 map_size) ---
            if map_size_obs != original_map_size:
                 cfg.training.som.map_size = original_map_size
                 logger.info(f"恢复 SOM 地图大小为: {original_map_size}")

            logger.info(f"观测 SOM 返回结果: {som_obs_results}") # 打印详细结果

            # --- 处理观测 SOM 结果 ---
            if som_obs_results and isinstance(som_obs_results, dict):
                results['som_observation'] = som_obs_results # 存储完整结果
                results['som_observation_model_path'] = som_obs_results.get('model_path') # 存储模型路径
                bmu_paths_obs = som_obs_results.get('bmu_indices_paths', {})
                if bmu_paths_obs:
                    try:
                        # 提取 {split: path} 格式
                        obs_bmu_positions = {split: p['positions'] for split, p in bmu_paths_obs.items() if isinstance(p, dict) and 'positions' in p and p['positions']}
                        logger.info(f"提取的观测 BMU 位置路径: {obs_bmu_positions}")
                        if not obs_bmu_positions:
                             raise ValueError("未能从观测 SOM 结果中提取有效的 BMU 位置路径。")
                        # 将观测 BMU 路径按所需格式添加到 low_dim_data_paths_for_pred
                        low_dim_data_paths_for_pred[obs_feature_name] = {
                            split: {'positions': path} for split, path in obs_bmu_positions.items()
                        }
                    except Exception as e:
                        logger.error(f"从观测 SOM 结果提取或格式化 BMU 位置路径时出错: {e}", exc_info=True)
                        # results['success'] 已经是 False，直接返回
                        return results
                else:
                    logger.error(f"观测 SOM ({obs_feature_name}) 结果中未找到 'bmu_indices_paths' 或其为空 ({pred_method} 需要)。")
                    # results['success'] 已经是 False，直接返回
                    return results
            else:
                logger.error(f"观测 SOM ({obs_feature_name}) 训练失败或未返回有效字典 ({pred_method} 需要)。")
                # results['success'] 已经是 False，直接返回
                return results

        # --- 步骤 4: 设置最终的 low_dim_data_paths 和 success 标志 ---
        if low_dim_data_paths_for_pred:
            # 检查是否所有需要的特征路径都已生成
            required_features = [target_field_dr]
            if needs_obs_som:
                 required_features.append(obs_feature_name)

            all_features_present = all(feat in low_dim_data_paths_for_pred for feat in required_features)

            if all_features_present:
                 results['low_dim_data_paths'] = low_dim_data_paths_for_pred
                 logger.info(f"最终设置的 low_dim_data_paths (for prediction): {results['low_dim_data_paths']}")
                 results['success'] = True # <--- 只有在这里才设置成功
            else:
                 missing_features = [f for f in required_features if f not in low_dim_data_paths_for_pred]
                 logger.error(f"未能为预测阶段生成所有必需的 SOM BMU 路径。缺少: {missing_features}")
                 # results['success'] 保持 False
        else:
            # 如果状态 SOM 成功但观测 SOM 失败（如果需要），则 low_dim_data_paths_for_pred 可能不为空，但上面会提前返回
            # 这个 else 理论上只会在状态 SOM 提取路径就失败时到达（也已提前返回）
            # 作为最后的保险：
            logger.error("未能为预测阶段生成任何 SOM BMU 输出路径。")
            # results['success'] 保持 False


    elif method == 'pca':
        # --- PCA 逻辑 ---
        # ... (调用 train_and_transform_pca) ...
        dr_results_pca = train_and_transform_pca(
             cfg, high_dim_paths, model_path, dr_scaler_path, transformed_data_dir
        )
        if dr_results_pca and dr_results_pca.get('success') and 'low_dim_data_paths' in dr_results_pca:
             results.update(dr_results_pca) # 合并 PCA 返回的结果
             # 确保 PCA 返回的 low_dim_data_paths 格式正确
             # PCA 通常只处理一个目标特征，格式应为 {target_field_dr: {split: path}}
             pca_paths = dr_results_pca['low_dim_data_paths']
             if isinstance(pca_paths, dict) and target_field_dr in pca_paths and isinstance(pca_paths[target_field_dr], dict):
                 results['low_dim_data_paths'] = pca_paths # 格式似乎兼容
                 results['success'] = True
             else:
                 logger.error(f"PCA 返回的 low_dim_data_paths 格式不正确: {pca_paths}")
                 results['success'] = False
        else:
             logger.error("PCA 降维失败或未返回预期结果。")
             results['success'] = False

    elif method == 'autoencoder':
         # --- Autoencoder 逻辑 ---
         # 类似 PCA，需要 train_and_transform_ae 返回 {'success': True, 'low_dim_data_paths': {target_field_dr: {split: path}}, ...}
         # ...
         pass # Placeholder
         # ...
         # results['success'] = True # 如果成功
         # results['low_dim_data_paths'] = ae_results['low_dim_data_paths']

    else:
        logger.error(f"未知的降维方法: {method}")
        # results['success'] is already False

    # --- Final Check and Logging ---
    # 这个最终检查现在更可靠，因为它只在前面明确设置 success=True 后才检查路径
    if not results.get('success'): # 检查是否为 False
         logger.error("降维步骤未能成功完成。") # 统一错误消息
    elif 'low_dim_data_paths' not in results or not results['low_dim_data_paths'] or not isinstance(results['low_dim_data_paths'], dict):
         # 这个检查只在 success=True 时运行
         logger.error("降维步骤标记为成功，但 low_dim_data_paths 丢失、为空或格式不正确！")
         results['success'] = False # 修正状态为 False
    # else: # 如果成功且路径有效，不需要额外日志，前面已经打过了
    #      logger.info(f"降维步骤成功，生成的低维数据路径: {results['low_dim_data_paths']}")


    logger.info(f"--- 降维完成 (方法: {method}, 成功: {results.get('success')}) ---") # Log final status
    return results

def run_prediction(cfg: DictConfig, dr_results: Dict[str, Any]) -> Dict[str, Any]:
    """运行所选的预测方法。"""
    config = DrprConfig.from_hydra_config(cfg)
    method = cfg.model.prediction.method
        # --- >>> 添加这部分代码 <<< ---
    # 在函数开始处，从 dr_results 获取降维方法
    dr_method = dr_results.get('method')
    if not dr_method:
        # 如果 dr_results 字典中没有 'method' 键，这是个问题
        logger.error("无法从 dr_results 中获取降维方法 ('method' key missing)。预测阶段无法继续。")
        # 返回一个表示失败的字典
        return {'method': method, 'success': False}
    # --- >>> 添加结束 <<< ---
    logger.info(f"--- 阶段：预测 (方法: {method}) ---")
    results = {'method': method}
    results['success'] = False # Initialize success to False
    pred_results = None

    low_dim_data_paths = dr_results.get('low_dim_data_paths')
    if not low_dim_data_paths:
        logger.error("缺少来自降维步骤的低维数据路径，无法进行预测。")
        results['success'] = False
        return results

    # --- 预测模型输出的通用路径 ---
    pred_model_dir = os.path.join(config.paths.models_base_dir, f"{dr_results['method']}_{method}") # 例如 models/som_hmm
    os.makedirs(pred_model_dir, exist_ok=True)
    pred_output_dir = os.path.join(config.paths.predictions_base_dir, f"{dr_results['method']}_{method}") # 例如 predictions/som_hmm
    os.makedirs(pred_output_dir, exist_ok=True)

    if method == 'hmm':
        dr_method = dr_results['method']
        hmm_input_type = cfg.model.prediction.hmm.get('input_type', 'bmu_rank')

        if dr_method == 'som':
            state_som_model_path = dr_results.get('som_state', {}).get('model_path')
            obs_som_model_path = dr_results.get('som_observation', {}).get('model_path')
            if not state_som_model_path or not obs_som_model_path:
                logger.error("SOM-HMM 模式缺少状态或观测 SOM 模型路径。")
                results['success'] = False; return results

            # 注意: train_hmm_main 期望基于特征名称找到 BMU 文件，
            # 但 dr_results['low_dim_data_paths'] 包含状态 BMU 的路径。
            # 这需要仔细对齐。现在假设 train_hmm_main 能根据配置找到必需的 BMU。
            # 此外，HMM 输出预测的状态秩/索引。
            hmm_param_filename = f"hmm_params_obs_{'_'.join(sorted(list(cfg.model.prediction.hmm.observation_features)))}.pkl"
            param_save_path = os.path.join(pred_model_dir, hmm_param_filename)
            pred_state_pattern = "predicted_state_ranks_{split}.npy" # HMM 预测状态序列

            pred_results = train_hmm_main( # 使用导入的函数
                cfg,
                state_som_model_path=state_som_model_path,
                obs_som_model_path=obs_som_model_path,
                param_save_path=param_save_path,
                predicted_state_save_pattern=pred_state_pattern # HMM 预测状态
            )
            # 调整结果键：HMM 预测状态，重建时需要映射
            if pred_results and 'predicted_states_paths' in pred_results:
                 results['predicted_low_dim_paths'] = pred_results['predicted_states_paths'] # 存储状态路径用于重建
                 results['model_path'] = pred_results.get('standard_model_path') or pred_results.get('model_path')
                 results['success'] = True # 标记成功
                 

        else:
             logger.error(f"不支持的降维方法 ({dr_method}) 与 HMM 组合。")
             results['success'] = False; return results

    elif method == 'lstm':
        logger.info("使用 LSTM 进行预测...")
        # --- 获取 LSTM 特定配置 ---
        lstm_cfg = cfg.model.prediction.lstm
        lstm_input_type = lstm_cfg.get('input_type', 'bmu_rank') # 默认或从配置读取
        target_field_name = dr_results.get('target_feature', 'salinity') # 获取目标特征名，可能需要从 dr_results 更可靠地获取

        # --- 检查输入类型兼容性 ---
        if dr_method == 'som' and lstm_input_type != 'bmu_rank':
            logger.warning(f"降维方法是 SOM，但 LSTM 输入类型配置为 '{lstm_input_type}' 而不是 'bmu_rank'。请检查配置。假设使用 BMU 索引。")
            lstm_input_type = 'bmu_rank' # 强制或报错
        elif dr_method in ['pca', 'autoencoder'] and lstm_input_type == 'bmu_rank':
            logger.error(f"降维方法是 {dr_method} (连续)，但 LSTM 输入类型配置为 'bmu_rank'。不兼容。")
            return results # 返回失败
        # 可以添加对 'continuous' 或 'dv' 类型的处理逻辑，如果 train_lstm 支持的话
        elif lstm_input_type not in ['bmu_rank']: # 目前假设只支持 bmu_rank
             logger.error(f"当前实现的 LSTM 预测仅支持 'bmu_rank' 输入类型，但配置为 '{lstm_input_type}'。")
             return results
        
        # --- 设置 LSTM 保存路径 ---
        model_save_dir_lstm = pred_model_dir # 模型保存在方法组合目录下
        # LSTM 通常不需要单独的 scaler 文件，内部处理 embedding
        # scaler_save_path_lstm = os.path.join(config.paths.processed_data_dir, f"{dr_method}_lstm_input_scaler.pkl") # 可能不需要
        pred_output_dir_lstm = pred_output_dir # 预测结果保存在方法组合目录下
        prediction_filename_pattern = f"predicted_lstm_target_{target_field_name}_{{split}}.npy" # 预测文件名模式

        # --- 调用 LSTM 训练和预测函数 ---
        # 注意：train_and_predict_lstm 需要 low_dim_data_paths 的结构是 {feature: {split: {'positions': path}}}
        # dr_results['low_dim_data_paths'] 的结构需要与此匹配。
        # pipeline_new.py 中的 run_dimensionality_reduction (SOM部分) 应该已经生成了这种结构。
        logger.info(f"传递给 LSTM 的低维数据路径结构: {low_dim_data_paths}") # 打印检查


        try:
            pred_results = train_and_predict_lstm(
                cfg=cfg,
                low_dim_data_paths=low_dim_data_paths, # 包含目标和（可选）观测特征的 BMU 路径
                # target_field_name 和 input_feature_info 现在由 train_lstm 内部推断或从cfg读取
                model_save_dir=model_save_dir_lstm,
                prediction_save_dir=pred_output_dir_lstm,
                prediction_filename_pattern=prediction_filename_pattern
            )
        except Exception as e:
             logger.error(f"调用 train_and_predict_lstm 时发生错误: {e}", exc_info=True)

        # --- 处理 LSTM 返回结果 ---
        if pred_results:
            # 关键：将预测的 *目标* 低维路径存储到通用键中
            if 'predicted_target_low_dim_paths' in pred_results:
                results['predicted_low_dim_paths'] = pred_results['predicted_target_low_dim_paths']
                results['model_path'] = pred_results.get('model_path') # 保存 LSTM 模型路径
                results['success'] = True # 标记成功
                logger.info(f"LSTM 预测成功，预测的目标路径: {results['predicted_low_dim_paths']}")
            else:
                logger.error("train_and_predict_lstm 未返回 'predicted_target_low_dim_paths'。")
        else:
            logger.error("train_and_predict_lstm 调用失败或未返回结果。")

    else:
        logger.error(f"未知的预测方法: {method}")

    # --- 最终检查和日志 ---
    if not results.get('success'): # <--- 因为 'success' 键不存在或不是 True，这里会执行
        logger.error(f"预测阶段 ({method}) 失败。") # <--- 这就是你看到的错误日志
    elif 'predicted_low_dim_paths' not in results or not results['predicted_low_dim_paths']:
        logger.error(f"预测阶段 ({method}) 标记为成功，但缺少 'predicted_low_dim_paths'！")
        results['success'] = False # 修正状态

    logger.info(f"--- 预测完成 (方法: {method}, 成功: {results.get('success')}) ---") # <--- 这里打印 "成功: None"
    return results

def run_reconstruction(cfg: DictConfig, dr_results: Dict[str, Any], pred_results: Dict[str, Any]) -> Dict[str, Any]:
    """根据降维方法和预测的低维数据运行重建。"""
    config = DrprConfig.from_hydra_config(cfg)
    dr_method = dr_results['method']
    pred_method = pred_results['method'] # 获取预测方法
    logger.info(f"--- 阶段：重建 (DR: {dr_method}, Pred: {pred_method}) ---")
    results = {'success': False} # 初始化

    predicted_low_dim_paths = pred_results.get('predicted_low_dim_paths')
    dr_model_path = dr_results.get('model_path')
    high_dim_scaler_path = dr_results.get('scaler_path') # PCA/AE 的输入 Scaler

    if not predicted_low_dim_paths:
        logger.error("缺少预测的低维数据路径，无法重建。")
        return results

    recon_output_dir = os.path.join(config.paths.reconstructions_base_dir, f"{dr_method}_{pred_method}")
    os.makedirs(recon_output_dir, exist_ok=True)
    results['reconstructed_high_dim_paths'] = {}

    if dr_method == 'som':
        # 重建需要预测的目标状态 BMU 和状态 SOM 模型
        # 注意：即使预测是 LSTM，重建仍然依赖于 *状态* SOM
        state_som_model_path = dr_results.get('som_state', {}).get('model_path')
        if not state_som_model_path:
            logger.error("缺少状态 SOM 模型路径，无法进行 BMU 重建。")
            return results

        # --- 条件化传递 hmm_params_path ---
        hmm_params_path_for_recon = None
        if pred_method == 'hmm':
            # 只有 HMM 预测时才需要 HMM 参数进行可能的等级转换
            hmm_params_path_for_recon = pred_results.get('model_path') # 获取 HMM 参数路径
            if not hmm_params_path_for_recon:
                 logger.warning("HMM 预测方法，但缺少 HMM 参数路径。假设预测是线性索引或重建函数能处理。")
        # 如果 pred_method 是 'lstm'，则 hmm_params_path_for_recon 保持为 None

        logger.info(f"开始 SOM 重建 (Pred: {pred_method})...")
        logger.info(f"  状态 SOM 模型路径: {state_som_model_path}")
        logger.info(f"  HMM 参数路径 (仅 HMM rank 输入时使用): {hmm_params_path_for_recon}")

        # 获取目标特征的原始高维数据 scaler (通常是 'salinity_scaler.npy')
        # 这个 scaler 是必须的，用于将 SOM 重建的结果反标准化回原始范围
        target_feature_name = cfg.reconstruction.target_field # 例如 "salinity"
        target_scaler_path = os.path.join(config.paths.processed_data_dir, f"{target_feature_name}_scaler.npy")
        target_scaler_params = None
        if os.path.exists(target_scaler_path):
             try:
                 # load_scaler 可能需要改进以加载 .npy 文件
                 scaler_content = np.load(target_scaler_path, allow_pickle=True).item()
                 if isinstance(scaler_content, dict) and 'mean' in scaler_content and 'std' in scaler_content:
                      target_scaler_params = scaler_content
                      logger.info(f"成功加载目标特征 '{target_feature_name}' 的 Scaler。")
                 else:
                      logger.warning(f"加载的目标特征 Scaler 文件格式不正确: {target_scaler_path}")
             except Exception as e:
                 logger.warning(f"加载目标特征 Scaler ({target_scaler_path}) 失败: {e}。将跳过反标准化。")
        else:
             logger.warning(f"未找到目标特征 Scaler 文件: {target_scaler_path}。将跳过反标准化。")


        for split, pred_path in predicted_low_dim_paths.items():
            logger.info(f"  重建 {split} 分割 (BMU)...")
            output_path = os.path.join(recon_output_dir, f"reconstructed_high_dim_{split}.npy")
            try:
                # 调用重建函数，根据 pred_method 条件传递 hmm_params_path
                reconstructed_flat = reconstruct_from_bmu(
                    cfg=cfg, # 传递 cfg 以便内部访问配置
                    som_model_path=state_som_model_path,
                    predicted_bmu_path=pred_path, # 这是预测的目标 BMU 索引
                    output_path=output_path,
                    hmm_params_path=hmm_params_path_for_recon # <-- 条件传递
                )

                if reconstructed_flat is not None:
                    # --- 反标准化 ---
                    if target_scaler_params:
                         logger.info(f"  对 {split} 重建结果执行反标准化...")
                         mean_vec = target_scaler_params['mean']
                         std_vec = target_scaler_params['std']
                         epsilon = 1e-8

                         # 检查维度匹配 (scaler vs 重建数据)
                         if isinstance(mean_vec, np.ndarray) and mean_vec.ndim == 1 and len(mean_vec) == reconstructed_flat.shape[1]:
                             reconstructed_inv_flat = reconstructed_flat * (std_vec + epsilon) + mean_vec
                         elif np.isscalar(mean_vec): # 处理全局 scaler
                             reconstructed_inv_flat = reconstructed_flat * (std_vec + epsilon) + mean_vec
                         else:
                             logger.error(f"目标特征 Scaler 维度 ({mean_vec.shape if isinstance(mean_vec, np.ndarray) else type(mean_vec)}) "
                                          f"与重建特征 ({reconstructed_flat.shape[1]}) 不匹配。跳过反标准化。")
                             reconstructed_inv_flat = reconstructed_flat # 回退
                    else:
                         reconstructed_inv_flat = reconstructed_flat # 没有 scaler，使用原始重建结果

                    # 保存最终的（可能反标准化的）扁平数据
                    np.save(output_path, reconstructed_inv_flat)
                    results['reconstructed_high_dim_paths'][split] = output_path
                    logger.info(f"  {split} 重建的高维数据 (flat) 已保存: {output_path}")
                else:
                    logger.error(f"  {split} BMU 重建失败 (reconstruct_from_bmu 返回 None)。")

            except Exception as e:
                logger.error(f"  重建 {split} (BMU) 出错: {e}", exc_info=True)

    elif dr_method in ['pca', 'autoencoder']:
         # --- 使用 DR 模型进行逆变换 ---
         if not dr_model_path:
             logger.error(f"缺少 {dr_method} 模型路径，无法重建。")
             results['success'] = False; return results
         # --- 反标准化需要高维输入的 Scaler ---
         if not high_dim_scaler_path or not os.path.exists(high_dim_scaler_path):
             logger.error(f"缺少 {dr_method} 的高维输入 Scaler 路径 ({high_dim_scaler_path})，无法反标准化。")
             results['success'] = False; return results

         try:
             # 加载 DR 模型和高维 Scaler
             if dr_method == 'pca':
                 dr_model = joblib.load(dr_model_path)
             else: # autoencoder
                 # 假设 AE 类有 load 方法
                 ae_model_instance = AutoencoderDimensionalityReduction(cfg=cfg) # 可能需要传递参数
                 dr_model = ae_model_instance.load(dr_model_path)

             high_dim_scaler = joblib.load(high_dim_scaler_path)
         except Exception as e:
             logger.error(f"加载 DR 模型或高维 Scaler 失败: {e}")
             results['success'] = False; return results

         for split, pred_path in predicted_low_dim_paths.items():
             logger.info(f"  重建 {split} 分割 ({dr_method})...")
             output_path = os.path.join(recon_output_dir, f"reconstructed_high_dim_{split}.npy")
             try:
                 predicted_low_dim = np.load(pred_path)

                 # 使用 DR 模型进行逆变换
                 reconstructed_scaled = dr_model.inverse_transform(predicted_low_dim)

                 # 使用高维 scaler 进行反向缩放
                 reconstructed_high_dim = high_dim_scaler.inverse_transform(reconstructed_scaled)

                 np.save(output_path, reconstructed_high_dim)
                 results['reconstructed_high_dim_paths'][split] = output_path
                 logger.info(f"  {split} 重建的高维数据 (flat) 已保存: {output_path}")

             except Exception as e:
                 logger.error(f"  重建 {split} ({dr_method}) 失败: {e}", exc_info=True)

    else:
         logger.error(f"未知的降维方法 '{dr_method}'，无法重建。")
         results['success'] = False; return results

    results['success'] = bool(results.get('reconstructed_high_dim_paths')) # 如果任何分割被重建则为成功
    logger.info(f"--- 重建完成 (DR: {dr_method}, Pred: {pred_method}) ---")
    return results

def run_evaluation(cfg: DictConfig, recon_results: Dict[str, Any], dr_method: str, pred_method: str):
    """运行评估，比较重建数据与原始数据。"""
    config = DrprConfig.from_hydra_config(cfg)
    logger.info(f"--- 阶段：评估 (DR: {dr_method}, Pred: {pred_method}) ---")
    results = {}

    reconstructed_paths = recon_results.get('reconstructed_high_dim_paths', {})
    if not reconstructed_paths:
        logger.error("没有重建的高维数据路径可供评估。")
        return results

    # --- 加载 Mask (用于重塑) ---
    mask = load_mask(cfg) # 假设盐度 mask 具有代表性 (0=有效, 1=无效)
    if mask is None:
        logger.warning("无法加载 Mask，空间评估可能不准确。")
        boolean_mask = None
        boolean_flat_mask = None
        target_spatial_shape = None
    else:
        boolean_mask = (mask == 0) # 布尔掩码: True 表示有效点
        boolean_flat_mask = boolean_mask.flatten()
        target_spatial_shape = mask.shape
        logger.info(f"加载 Mask 并创建布尔掩码 (True=有效)，形状: {mask.shape}, 有效点数: {np.sum(boolean_mask)}")


    # --- 加载原始高维数据以供比较 ---
    target_field_eval = "salinity" # 假设总是评估盐度
    original_high_dim = {}
    split_indices = load_split_indices(cfg) # 需要原始索引
    try:
        # 加载原始数据进行评估是关键点，需要确保与预测对齐
        # 最好的方法是加载原始的、未缩放的高维数据，然后使用 split_indices 切片
        raw_salt_full = load_raw_data(cfg, "salinity") # 加载原始盐度数据
        if raw_salt_full is None: raise RuntimeError("无法加载原始盐度数据进行评估。")

        for split, indices in split_indices.items():
             if indices is not None and len(indices) > 0:
                 # --- 对齐时间步长是关键 ---
                 # 预测和重建后的样本数可能少于原始分割的样本数
                 # 需要一种方法来知道重建数据对应原始数据的哪些时间步
                 # TODO: Pipeline 需要传递原始样本数或对齐信息给评估阶段
                 # 暂时假设重建数据与原始数据分割的前 N 步对齐
                 original_high_dim[split] = raw_salt_full[indices]
                 logger.info(f"加载原始高维数据 ({split}) 形状: {original_high_dim[split].shape}")

    except Exception as e:
         logger.error(f"加载原始高维数据进行评估时出错: {e}", exc_info=True)
         return results

    # --- 对每个分割执行评估 ---
    all_eval_results = {}
    for split, recon_path in reconstructed_paths.items():
        if split not in original_high_dim:
             logger.warning(f"跳过评估 {split}，因为缺少对应的原始数据。")
             continue

        logger.info(f"\n--- 评估分割: {split} ---")
        try:
            recon_flat = np.load(recon_path) # 加载重建的扁平数据
            orig_split_raw = original_high_dim[split] # 获取对应的原始数据切片 (未缩放)


            # --- 对齐时间步长 ---
            n_recon_samples = recon_flat.shape[0]
            n_orig_samples = orig_split_raw.shape[0]
            if n_recon_samples == 0: continue # 没有可评估的数据

            if n_recon_samples > n_orig_samples:
                 logger.warning(f"{split}: 重建样本数 ({n_recon_samples}) 大于原始样本数 ({n_orig_samples})。将截断重建数据。")
                 recon_flat = recon_flat[:n_orig_samples]
            elif n_recon_samples < n_orig_samples:
                 logger.warning(f"{split}: 重建样本数 ({n_recon_samples}) 小于原始样本数 ({n_orig_samples})。将使用后 {n_recon_samples} 个原始样本进行比较。")
                 orig_split_raw = orig_split_raw[-n_recon_samples:]


            # --- 比较重建数据和原始数据 ---
            # 原始数据可能是 (T, H, W)，需要根据 mask 展平以匹配 recon_flat
            if orig_split_raw.ndim > 2 and boolean_flat_mask is not None:
                 T = orig_split_raw.shape[0]
                 # 从原始数据中提取有效点 (使用 boolean_flat_mask)
                 orig_flat_full = orig_split_raw.reshape(T, -1)
                 if orig_flat_full.shape[1] != len(boolean_flat_mask):
                      logger.error(f"{split}: 原始数据展平后的特征数 ({orig_flat_full.shape[1]}) 与 Mask 大小 ({len(boolean_flat_mask)}) 不匹配！")
                      continue
                 orig_valid_flat = orig_flat_full[:, boolean_flat_mask] # 使用布尔掩码选择列

                 # 检查特征数是否匹配
                 num_valid_points_expected = np.sum(boolean_flat_mask)
                 if recon_flat.shape[1] != num_valid_points_expected:
                     logger.error(f"{split}: 原始数据有效点数 ({num_valid_points_expected}) 与重建数据特征数 ({recon_flat.shape[1]}) 不匹配！")
                     continue
                 orig_for_comparison = orig_valid_flat
            elif orig_split_raw.ndim == 2:
                 # 假设原始数据已经是扁平的并且只包含有效点
                 num_valid_points_expected = np.sum(boolean_flat_mask) if boolean_flat_mask is not None else recon_flat.shape[1] # 最佳猜测
                 if recon_flat.shape[1] != num_valid_points_expected:
                      logger.warning(f"{split}: 原始数据是2D，但其特征数 ({orig_split_raw.shape[1]}) 与 Mask 定义的有效点数 ({num_valid_points_expected}) 或重建数据特征数 ({recon_flat.shape[1]}) 不匹配。假设原始数据已对齐。")
                 # 进一步检查是否与重建数据匹配
                 if orig_split_raw.shape[1] != recon_flat.shape[1]:
                     logger.error(f"{split}: 原始数据(2D)特征数 ({orig_split_raw.shape[1]}) 与重建数据特征数 ({recon_flat.shape[1]}) 不匹配！")
                     continue
                 orig_for_comparison = orig_split_raw
            else:
                 logger.error(f"{split}: 原始数据维度 ({orig_split_raw.ndim}) 无法处理。")
                 continue

            # --- 计算指标 ---
            logger.info(f"  计算 RMSE 和 MAE ({split})...")
            metrics = {}
            rmse_field = np.full(target_spatial_shape, np.nan) if target_spatial_shape is not None else None
            
            if boolean_mask is not None and target_spatial_shape is not None:
                 # --- 重塑回空间网格进行比较 ---
                 recon_spatial = np.full((n_recon_samples,) + target_spatial_shape, np.nan)
                 orig_spatial_masked = np.full((n_recon_samples,) + target_spatial_shape, np.nan)
                 # 使用 boolean_flat_mask (True=有效) 来放置数据
                 temp_flat_base = np.full(boolean_mask.size, np.nan) # 创建基础 NaN 数组
                 for t in range(n_recon_samples):
                      temp_flat = temp_flat_base.copy() # 每次重置
                      temp_flat[boolean_flat_mask] = recon_flat[t] # 放入有效位置
                      recon_spatial[t] = temp_flat.reshape(target_spatial_shape)

                      temp_flat = temp_flat_base.copy() # 每次重置
                      temp_flat[boolean_flat_mask] = orig_for_comparison[t] # 放入有效位置
                      orig_spatial_masked[t] = temp_flat.reshape(target_spatial_shape)

                 # diff = recon_spatial - orig_spatial_masked # 现在两者都只在有效点有值
                 # valid_points_mask_3d = boolean_mask # 直接使用布尔掩码

                 # 计算差值只在有效点上进行
                 diff = np.full_like(recon_spatial, np.nan)
                 diff[:, boolean_mask] = recon_spatial[:, boolean_mask] - orig_spatial_masked[:, boolean_mask]

                 if np.any(boolean_mask): # 检查是否有任何有效点
                      # 计算指标只在有效点上进行
                      diff_sq_masked = np.square(diff[:, boolean_mask])
                      mean_rmse_val = np.sqrt(np.mean(diff_sq_masked))
                      mae_val = np.mean(np.abs(diff[:, boolean_mask]))

                      # 计算空间 RMSE 图
                      mean_diff_sq_spatial = np.nanmean(np.square(diff[:, boolean_mask]), axis=0) 
                      # valid_spatial_points = boolean_mask # 有效点由 boolean_mask 定义
                      if np.any(boolean_mask):
                           rmse_field[boolean_mask] = np.sqrt(mean_diff_sq_spatial)

                      metrics = {
                           "mean_rmse": float(mean_rmse_val), "mean_mae": float(mae_val),
                           "rmse_map": rmse_field,
                           "max_rmse": float(np.nanmax(rmse_field)) if np.any(np.isfinite(rmse_field)) else np.nan,
                           "min_rmse": float(np.nanmin(rmse_field)) if np.any(np.isfinite(rmse_field)) else np.nan
                      }
                 else:
                      # ... (处理没有有效点的情况) ...
                      metrics = {k: np.nan for k in ["mean_rmse", "mean_mae", "max_rmse", "min_rmse"]}
                      metrics["rmse_map"] = rmse_field 

            # --- 保存指标和生成图表 ---
            eval_output_dir = os.path.join(config.paths.evaluation_base_dir, f"{dr_method}_{pred_method}", split)
            os.makedirs(eval_output_dir, exist_ok=True)
            metrics_path = os.path.join(eval_output_dir, f"metrics_{split}.npy")
            np.save(metrics_path, metrics)
            logger.info(f"  评估指标 ({split}) 已保存到: {metrics_path}")

            if metrics.get("mean_rmse") is not np.nan:
                 rec_mean_ts = np.nanmean(recon_flat, axis=1)
                 orig_mean_ts = np.nanmean(orig_for_comparison, axis=1)
                 ts_save_path = os.path.join(eval_output_dir, f"time_series_comparison_{split}.png")
                 plot_time_series_comparison(rec_mean_ts, orig_mean_ts, cfg=cfg, save_path=ts_save_path,
                                               title=f"Spatial Mean TS ({split} - {dr_method}_{pred_method})")

                 if rmse_field is not None and np.any(np.isfinite(rmse_field)):
                     spatial_rmse_save_path = os.path.join(eval_output_dir, f"spatial_rmse_{split}.png")
                     plot_spatial_rmse(rmse_field, cfg=cfg, mask=boolean_mask, save_path=spatial_rmse_save_path,
                                       title=f"Spatial RMSE ({split} - {dr_method}_{pred_method})")
                 logger.info(f"  评估图表 ({split}) 已保存到: {eval_output_dir}")


            # --- (可选) 计算并绘制: 空间相关性图 (Cartopy) ---
            if cfg.evaluation.get("calculate_correlation", False): # 添加配置开关
                 logger.info(f"  计算空间相关性图 ({split})...")
                 try:
                     # 注意 calculate_correlation_map 需要 mask=True 表示无效
                     corr_map = calculate_correlation_map(recon_spatial, orig_spatial_masked, mask)
                     metrics['correlation_map'] = corr_map
                     metrics['mean_correlation'] = float(np.nanmean(corr_map)) if np.any(np.isfinite(corr_map)) else np.nan
                     logger.info(f"  {split} - Mean Correlation: {metrics['mean_correlation']:.6f}")

                     if cfg.evaluation.visualization.get("plot_spatial_correlation", True):
                         logger.info(f"  绘制空间相关性图 (Cartopy) ({split})...")
                         corr_save_path = os.path.join(eval_output_dir, f"spatial_correlation_{split}.png")
                         # plot_spatial_statistic 需要 mask=True 表示无效
                         plot_spatial_statistic(corr_map, cfg, mask, corr_save_path,
                                                title=f"Spatial Correlation ({split} - {dr_method}_{pred_method})",
                                                cmap='coolwarm', vmin=-1, vmax=1, cbar_label="Correlation Coeff.")
                        #  figure_paths_split.append(corr_save_path)

                 except Exception as e:
                     logger.error(f"  计算或绘制空间相关性图失败 ({split}): {e}", exc_info=True)


            # --- (可选) 绘制: 特定时间点的对比和差异图 (Cartopy) ---
            if cfg.evaluation.visualization.get("plot_instantaneous", False): # 添加配置开关
                 time_indices_to_plot = cfg.evaluation.visualization.get("time_indices_to_plot", [0, n_recon_samples // 2, n_recon_samples - 1])
                 logger.info(f"  绘制特定时间点的空间图 ({split}, indices={time_indices_to_plot})...")
                 for t_idx in time_indices_to_plot:
                     if 0 <= t_idx < n_recon_samples:
                         try:
                             orig_t = np.full(target_spatial_shape, np.nan) if target_spatial_shape is not None else None
                             recon_t = np.full(target_spatial_shape, np.nan) if target_spatial_shape is not None else None
                             diff_t = np.full(target_spatial_shape, np.nan) if target_spatial_shape is not None else None
                             orig_t[boolean_mask] = orig_spatial_masked[t_idx, boolean_mask]
                             recon_t[boolean_mask] = recon_spatial[t_idx, boolean_mask]
                             diff_t[boolean_mask] = diff[t_idx, boolean_mask]

                             # 并排对比图
                             comp_save_path = os.path.join(eval_output_dir, f"spatial_comparison_{split}_t{t_idx}.png")
                             plot_spatial_comparison_at_timestep(
                                 orig_t, recon_t, diff_t,
                                 cfg, mask, t_idx, comp_save_path, # plot 函数需要 mask=True 表示无效
                                 title_prefix=f"Spatial Comparison ({split} - {dr_method}_{pred_method})"
                             )

                             # (可选) 单独的差异图
                             if cfg.evaluation.visualization.get("plot_instantaneous_difference_only", True):
                                 diff_save_path = os.path.join(eval_output_dir, f"spatial_difference_{split}_t{t_idx}.png")
                                 plot_spatial_difference(
                                     diff_t, cfg, mask, t_idx, diff_save_path, # plot 函数需要 mask=True 表示无效
                                     title=f"Spatial Difference ({split} - {dr_method}_{pred_method})"
                                 )
                                #  figure_paths_split.append(diff_save_path)

                         except Exception as e:
                             logger.error(f"  绘制瞬时图失败 (t={t_idx}, split={split}): {e}", exc_info=True)



            # 存储此分割的结果
            all_eval_results[split] = {'metrics': metrics}
            logger.info(f"--- 评估 {split} 完成 ---")

        except Exception as e:
            logger.error(f"评估分割 {split} 时出错: {e}", exc_info=True)
            all_eval_results[split] = {"error": str(e)}

    results['evaluation_details'] = all_eval_results
    results['success'] = bool(all_eval_results) # 如果任何分割被评估则为成功

    for split, eval_data in all_eval_results.items():
        if 'metrics' in eval_data and 'mean_rmse' in eval_data['metrics']:
            logger.info(f"--- {split} RMSE: {eval_data['metrics']['mean_rmse']:.6f}")

    logger.info(f"--- 评估完成 (DR: {dr_method}, Pred: {pred_method}) ---")
    return results

# === 主 Pipeline ===
@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """主集成流程"""
    start_run_time = time.time()
    config = DrprConfig.from_hydra_config(cfg)

    logger.info("==== 开始集成流程 ====")
    logger.info(f"配置:\n{OmegaConf.to_yaml(cfg, resolve=True)}") # 解析插值

    # --- 创建基础目录 ---
    os.makedirs(config.paths.models_base_dir, exist_ok=True)
    os.makedirs(config.paths.params_base_dir, exist_ok=True)
    os.makedirs(config.paths.bmu_base_dir, exist_ok=True)
    os.makedirs(config.paths.predictions_base_dir, exist_ok=True)
    os.makedirs(config.paths.reconstructions_base_dir, exist_ok=True)
    os.makedirs(config.paths.evaluation_base_dir, exist_ok=True)

    # 设置全局随机种子
    set_global_seeds(cfg.get('random_seed', 22))

    # === 阶段 1: 降维 ===
    dr_results = run_dimensionality_reduction(cfg)
    if not dr_results or not dr_results.get('success'):
        logger.error("维度降低阶段失败，流程终止。")
        return

    # === 阶段 2: 预测 ===
    pred_results = run_prediction(cfg, dr_results)
    if not pred_results or not pred_results.get('success'):
        logger.error("预测阶段失败，流程终止。")
        return

    # === 阶段 3: 重建 ===
    recon_results = run_reconstruction(cfg, dr_results, pred_results)
    if not recon_results or not recon_results.get('success'):
        logger.error("重建阶段失败，流程终止。")
        return

    # === 阶段 4: 评估 ===
    eval_results = run_evaluation(cfg, recon_results, dr_results['method'], pred_results['method'])
    if not eval_results or not eval_results.get('success'):
        logger.warning("评估阶段失败或未生成任何结果。")
        # 继续保存总体结果

    # --- 保存总体 Pipeline 结果 ---
    final_results = {
        'config': OmegaConf.to_container(cfg, resolve=True),
        'dimensionality_reduction': dr_results,
        'prediction': pred_results,
        'reconstruction': recon_results,
        'evaluation': eval_results,
    }
    # 保存到以方法组合命名的子目录中
    results_save_dir = os.path.join(config.paths.evaluation_base_dir, f"{dr_results['method']}_{pred_results['method']}")
    os.makedirs(results_save_dir, exist_ok=True)
    results_path = os.path.join(results_save_dir, "pipeline_summary.pkl")
    try:
        joblib.dump(final_results, results_path)
        logger.info(f"Pipeline 总结结果已保存到: {results_path}")
    except Exception as e:
        logger.error(f"保存 Pipeline 总结失败: {e}")

    total_run_time = time.time() - start_run_time
    logger.info(f"==== 集成流程完成 (DR: {dr_results['method']}, Pred: {pred_results['method']}) ====")
    logger.info(f"总用时: {total_run_time:.2f} 秒")
    logger.info(f"Hydra 输出目录: {os.getcwd()}") # Hydra 会改变目录

if __name__ == "__main__":
    main()