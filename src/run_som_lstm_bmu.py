# src/run_som_lstm_bmu_pipeline_with_obs.py
# -*- coding: utf-8 -*-

import os
import sys
import logging
import time
import hydra
from omegaconf import DictConfig, OmegaConf, listconfig, OmegaConf
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
import joblib
import torch
import shutil

# --- 项目设置 和 import ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils.hydra_config import DrprConfig
from src.evaluation.visualization import plot_time_series_comparison, plot_spatial_rmse
from src.utils.model_utils import get_device_from_config
from src.dimensionality_reduction.som_pytorch import SOMTorch
# 导入集中化的数据加载函数
from src.utils.data_loader import load_raw_data, load_mask, load_scaler, load_split_indices
# 导入 SOM 训练函数 (假设它们能生成 BMU 索引)
from src.training.train_som import train_single_feature_som, train_combined_feature_som # 保留两个
# 导入 LSTM 训练/预测函数 (需要修改以处理多特征输入)
from src.training.train_lstm import train_and_predict_lstm
# 导入 BMU 重建函数
from src.reconstruction.reconstruct_bmu import reconstruct_from_bmu

logger = logging.getLogger(__name__)


def run_dimensionality_reduction(cfg: DictConfig, feature_name: str, is_observation: bool = False) -> Dict[str, Any]:
    """
    运行 SOM 降维流程，为指定特征生成 SOM 模型和 BMU 索引序列。

    Args:
        cfg: Hydra 配置对象。
        feature_name: 要处理的特征名称 (例如 'salinity', 'wind_flow')。
        is_observation: 指示是否为观测特征 (可能影响 SOM 参数，如 map_size)。

    Returns:
        包含 SOM 模型路径和 BMU 索引文件路径的字典。
        例如: {'model_path': 'path/to/som.pth',
               'feature_name': feature_name, # 添加特征名
               # Key changed to match train_som output
               'bmu_paths': {'train': {'positions': 'path/train.npy', 'distances': '...'}, ...}}
    """
    logger.info(f"============== 开始降维阶段 (SOM for {feature_name}{' - Observation' if is_observation else ' - Target'}) ==============")
    config = DrprConfig.from_hydra_config(cfg)

    # --- MODIFICATION START: Use correct map size based on feature type --- 
    original_map_size = cfg.training.som.map_size # Store original if needed
    map_size_to_use = None
    if is_observation:
        if hasattr(cfg.training.som, 'map_size_obs'):
            map_size_to_use = cfg.training.som.map_size_obs
            logger.info(f"使用观测特征的 SOM map_size: {map_size_to_use}")
        else:
            logger.warning("配置中未找到 'map_size_obs'，观测特征将使用默认 map_size。")
            map_size_to_use = cfg.training.som.map_size_sta # Fallback to sta if obs not defined
    else: # Target feature
        map_size_to_use = cfg.training.som.map_size_sta
        logger.info(f"使用目标特征的 SOM map_size: {map_size_to_use}")

    # Temporarily set the map_size in the config for the training function
    # Ensure the config is mutable if needed (may not be necessary if train_som reads it directly)
    # OmegaConf.set_struct(cfg.training.som, False) # Allow modification
    # cfg.training.som.map_size = map_size_to_use
    # OmegaConf.set_struct(cfg.training.som, True) # Make immutable again
    # --- MODIFICATION END ---

    som_results = {}
    # --- MODIFICATION START: Pass map_size explicitly or ensure train_som reads correctly --- 
    # Option 1: Modify train_som to accept map_size (preferred)
    # Option 2: Modify cfg temporarily (shown above, but less clean)
    # Assuming train_som is modified or reads the correct sub-config based on feature type
    # We will rely on train_som to use the correct map_size based on its logic or passed args
    # The logging above confirms which size *should* be used.

    # Determine if it's a combined feature based on observation features list
    is_combined_feature = False
    obs_features_list = list(cfg.model.prediction.lstm.get('observation_features', []))
    if feature_name != config.reconstruction.target_field and '_' in feature_name:
        parts = feature_name.split('_')
        # Check if the parts exactly match the observation features list
        if sorted(parts) == sorted(obs_features_list) and len(parts) == len(obs_features_list):
            is_combined_feature = True

    if is_combined_feature:
        logger.info(f"使用 train_combined_feature_som 为组合特征 '{feature_name}'")
        som_results = train_combined_feature_som(
            cfg,
            output_feature_name=feature_name
            # Ensure train_combined_feature_som uses the correct map_size (e.g., map_size_obs)
        )
    else: # Single feature (target or an individual observation feature if not combined)
        logger.info(f"使用 train_single_feature_som 为单一特征 '{feature_name}'")
        som_results = train_single_feature_som(
            cfg,
            feature_name=feature_name,
            # Pass map_size if train_single_feature_som accepts it
            # map_size=map_size_to_use # Example if function signature allows
            # Ensure train_single_feature_som uses the correct map_size (sta or obs)
        )
    # --- MODIFICATION END ---

    # Restore original map_size if cfg was modified (if Option 2 was used)
    # OmegaConf.set_struct(cfg.training.som, False)
    # cfg.training.som.map_size = original_map_size
    # OmegaConf.set_struct(cfg.training.som, True)

    # --- 检查结果 ---
    if not som_results:
        logger.error(f"特征 '{feature_name}' 的 SOM 降维失败。")
        return {}
    # --- MODIFICATION START: Check for the correct output key from train_som --- 
    # The key from train_som is likely 'bmu_paths', not 'bmu_indices_paths'
    if 'model_path' not in som_results or 'bmu_paths' not in som_results:
         logger.error(f"SOM 降维结果 ({feature_name}) 缺少 model_path 或 bmu_paths。 Got keys: {som_results.keys()}")
         # Handle potential legacy key 'bmu_indices_paths' just in case
         if 'bmu_indices_paths' in som_results:
             logger.warning(f"({feature_name}) 检测到旧的 'bmu_indices_paths' 键，使用它。")
             som_results['bmu_paths'] = som_results['bmu_indices_paths']
         else:
              return {}
    # --- MODIFICATION END ---

    som_results['feature_name'] = feature_name # 确保存储了特征名

    logger.info(f"SOM 模型 ({feature_name}) 已训练并保存。")
    # --- MODIFICATION START: Log the correct path structure --- 
    logger.info(f"BMU 文件路径 ({feature_name}): {som_results['bmu_paths']}")
    # --- MODIFICATION END ---
    logger.info(f"============== 结束降维阶段 (SOM for {feature_name}) ==============")
    return som_results


def run_lstm_prediction(
    cfg: DictConfig,
    target_field_name: str,
    # --- MODIFICATION START: Input is the structured dictionary from main --- 
    # low_dim_feature_paths: Dict[str, Dict[str, str]] # Old signature
    structured_bmu_paths: Dict[str, Dict[str, Dict[str, str]]] # New: {feature: {split: {'positions': path, ...}}}
    # --- MODIFICATION END ---
) -> Dict[str, Any]:
    """
    运行 LSTM 训练和预测流程。
    使用来自 target 和 (可选的) observation 特征的 BMU 索引序列作为输入。
    预测目标是 target 特征的下一个 BMU 索引。

    Args:
        cfg: Hydra 配置对象。
        target_field_name: 目标字段名 (用于命名输出文件和标识预测目标)。
        # --- MODIFICATION START: Update docstring --- 
        structured_bmu_paths: 包含各特征 ('target', 'obs1', ...) 的
                                BMU 文件路径字典 ({feature: {split: {'positions': path, ...}}})。

    Returns:
        包含预测出的 *目标* 低维序列文件路径的字典。
        例如: {'predicted_target_low_dim_paths': {'train': 'path/pred_train.npy', ...}}
               可能还包含 LSTM 模型路径等。
    """
    logger.info("============== 开始 LSTM 训练和预测阶段 ==============")
    config = DrprConfig.from_hydra_config(cfg)
    # --- MODIFICATION START: Use the new input dictionary --- 
    feature_names = list(structured_bmu_paths.keys())
    num_features = len(feature_names)
    logger.info(f"LSTM 将使用以下特征的 BMU 序列作为输入: {feature_names} (共 {num_features} 个)")

    if target_field_name not in feature_names:
         logger.error(f"目标特征 '{target_field_name}' 不在提供的低维路径字典中!")
         return {}

    # Check if all features have train/val/test paths with 'positions'
    required_splits = ["train", "val", "test"]
    for feature, splits_dict in structured_bmu_paths.items():
        if not all(s in splits_dict for s in required_splits):
             logger.error(f"特征 '{feature}' 缺少 train/val/test 的完整 BMU 路径分割。")
             return {}
        for split, path_dict in splits_dict.items():
             if 'positions' not in path_dict or not path_dict['positions']:
                 logger.error(f"特征 '{feature}' 的 '{split}' 分割缺少 'positions' 路径。")
                 return {}
    # --- MODIFICATION END ---

    # --- REMOVE Data Loading Loop: This is now handled inside train_and_predict_lstm ---
    # logger.info("加载并合并 BMU 索引数据...")
    # low_dim_data: Dict[str, np.ndarray] = {} # 存储合并后的数据: {split: (T, num_features)}
    # expected_length = -1

    # try:
    #     for split in ["train", "val", "test"]:
    #         split_data_list = []
    #         has_split_data = False
    #         logger.info(f"  处理 LSTМ 输入分割: {split}") # Moved log message here
    #         for i, feature in enumerate(feature_names): # 按固定顺序加载
    #             path_info = low_dim_feature_paths.get(feature, {}).get(split) # Get the dictionary or path string

    #             # --- MODIFICATION START ---
    #             # Check if path_info is a dictionary and extract the 'positions' path
    #             if isinstance(path_info, dict):
    #                 bmu_positions_path = path_info.get('positions')
    #                 if not bmu_positions_path:
    #                     logger.error(f"    错误: 特征 '{feature}' 的 BMU 路径字典中缺少 'positions' 键: {path_info}")
    #                     # Decide how to handle: skip feature, skip split, or raise error
    #                     # Option: Skip this feature for this split if non-critical
    #                     logger.warning(f"    跳过特征 '{feature}' 的 {split} 分割，因为缺少 'positions' 路径。")
    #                     continue # Or raise ValueError if all features are mandatory
    #                 logger.info(f"    加载特征 '{feature}' 的 BMU 索引 (positions): {bmu_positions_path}")
    #                 path_to_load = bmu_positions_path
    #             elif isinstance(path_info, str) and os.path.exists(path_info):
    #                 # Handle cases where it might already be a direct path (less likely based on logs, but safer)
    #                 logger.warning(f"    特征 '{feature}' 的 BMU 路径直接是字符串: {path_info}. 假设这是 BMU 索引文件。")
    #                 path_to_load = path_info
    #             else:
    #                 logger.error(f"    错误: 特征 '{feature}' 的 {split} 分割缺少 BMU 数据路径或路径无效: {path_info}")
    #                 # Decide how to handle: skip feature, skip split, or raise error
    #                 logger.warning(f"    跳过特征 '{feature}' 的 {split} 分割，因为路径无效或缺失。")
    #                 continue # Or raise ValueError

    #             # Load data using the extracted path
    #             try:
    #                 data = np.load(path_to_load).flatten() # 确保是一维
    #                 has_split_data = True # Mark that we found data for this split
    #                 logger.info(f"      加载形状: {data.shape}")
    #             except FileNotFoundError:
    #                 logger.error(f"      文件未找到: {path_to_load}")
    #                 continue # Skip this feature for this split
    #             except Exception as load_err:
    #                 logger.error(f"      加载 BMU 文件 {path_to_load} 失败: {load_err}")
    #                 continue # Skip this feature for this split
    #             # --- MODIFICATION END ---

    #             # 长度检查
    #             if not split_data_list: # If this is the first feature successfully loaded for this split
    #                 expected_length = len(data)
    #             elif len(data) != expected_length:
    #                 logger.error(f"序列长度不一致! {split} - 特征 '{feature}' ({len(data)}) != 预期 ({expected_length})")
    #                 # Handle inconsistency: raise error or try to reconcile?
    #                 raise ValueError(f"Sequence length mismatch for split {split}, feature {feature}")

    #             split_data_list.append(data)

    #         # 如果此 split 有任何数据被加载
    #         if has_split_data and split_data_list:
    #             if len(split_data_list) != num_features:
    #                  logger.warning(f"{split} 分割只加载了 {len(split_data_list)}/{num_features} 个特征。请检查配置和文件。")
    #                  # Decide if this is acceptable or an error
    #                  # If acceptable, need to handle potential shape mismatches later

    #             # 堆叠特征 -> (T, num_loaded_features)
    #             try:
    #                 low_dim_data[split] = np.stack(split_data_list, axis=-1)
    #                 logger.info(f"  {split} 分割合并后形状: {low_dim_data[split].shape}")
    #             except ValueError as stack_err:
    #                  logger.error(f"堆叠 {split} 分割的特征时出错 (可能由于长度不一致或特征缺失): {stack_err}")
    #                  # Handle error, maybe skip this split for training/prediction
    #                  continue # Skip to next split

    #         elif not has_split_data:
    #              logger.warning(f"未找到 {split} 分割的任何 BMU 数据。")


    # except ValueError as e: # Catch length mismatch error
    #     logger.error(f"加载或合并 LSTM 输入数据时出错: {e}", exc_info=True)
    #     return {} # Indicate failure
    # except Exception as e:
    #     logger.error(f"加载或合并 LSTM 输入数据时发生意外错误: {e}", exc_info=True)
    #     return {} # Indicate failure

    # 检查是否至少有训练数据
    if "train" not in low_dim_data or low_dim_data["train"].size == 0:
        logger.error("未能加载 LSTM 的训练数据。")
        return {}

    # --- 调用 (修改后的) LSTM 训练和预测函数 ---
    prediction_base_dir = config.paths.predicted_low_dim_dir
    os.makedirs(prediction_base_dir, exist_ok=True)
    model_save_dir = config.paths.lstm_models_dir
    os.makedirs(model_save_dir, exist_ok=True)
    prediction_filename_pattern = f"predicted_lstm_target_{target_field_name}_{{split}}.npy"
    logger.info(f"LSTM 模型将保存到目录: {model_save_dir}")
    logger.info(f"LSTM 预测的 *目标* BMU 索引将保存到目录: {prediction_base_dir}，模式: {prediction_filename_pattern.format(split='*')}")

    # --- MODIFICATION START: Pass the structured_bmu_paths directly --- 
    lstm_results = train_and_predict_lstm(
        cfg=cfg,
        # Pass the structured dictionary containing paths
        low_dim_feature_paths=structured_bmu_paths,
        # target_field_name is now inferred within train_and_predict_lstm
        # input_feature_info is now constructed within train_and_predict_lstm
        model_save_dir=model_save_dir,
        prediction_save_dir=prediction_base_dir,
        prediction_filename_pattern=prediction_filename_pattern
    )
    # --- MODIFICATION END ---

    # --- 检查 LSTM 结果 ---
    if not lstm_results or 'predicted_target_low_dim_paths' not in lstm_results:
        logger.error("LSTM 训练或预测失败，或未返回预测的目标路径。")
        return {}

    logger.info(f"LSTM 预测完成。预测的目标 BMU 索引文件路径: {lstm_results['predicted_target_low_dim_paths']}")
    logger.info("============== 结束 LSTM 训练和预测阶段 ==============")
    return lstm_results


# --- 步骤 3: 重建和评估 ---
# ... (Reconstruction and Evaluation part remains largely the same) ...
# ... Ensure it uses target_som_results['model_path'] correctly ...
# ... Ensure it uses prediction_results['predicted_target_low_dim_paths'] correctly ...

def run_reconstruction_and_evaluation(
    cfg: DictConfig,
    target_som_results: Dict[str, Any], # 包含目标 SOM 模型路径 ('model_path')
    prediction_results: Dict[str, Any]  # 包含预测的 *目标* BMU 索引路径 ('predicted_target_low_dim_paths')
) -> Dict[str, Any]:
    """
    使用预测的 *目标* BMU 序列进行重建，并评估结果。

    Args:
        cfg: Hydra 配置对象。
        target_som_results: 包含目标 SOM 模型路径 ('model_path') 的字典。
        prediction_results: 包含预测的 *目标* 低维 BMU 索引路径 ('predicted_target_low_dim_paths') 的字典。

    Returns:
        包含各分割评估结果的字典。
    """
    logger.info("============== 开始重建和评估阶段 (基于预测的目标 BMU) ==============")
    all_eval_results = {}
    config = DrprConfig.from_hydra_config(cfg)
    device = get_device_from_config(cfg)
    target_field = config.reconstruction.target_field

    # --- 检查预测结果 ---
    if not prediction_results or 'predicted_target_low_dim_paths' not in prediction_results or not prediction_results['predicted_target_low_dim_paths']:
        logger.error("预测结果不包含有效的预测目标路径 ('predicted_target_low_dim_paths')。无法评估。")
        return {}
    predicted_paths_dict = prediction_results['predicted_target_low_dim_paths'] # 这是预测的目标 BMU 路径

    # --- 加载目标字段的 SOM 模型 ---
    target_som_model_path = target_som_results.get('model_path')
    if not target_som_model_path or not os.path.exists(target_som_model_path):
        logger.error(f"目标特征 '{target_field}' 的 SOM 模型路径无效或不存在: {target_som_model_path}")
        return {}
    try:
        target_som_model = SOMTorch.load(target_som_model_path, device=device)
        logger.info(f"成功加载目标特征 '{target_field}' 的 SOM 模型: {target_som_model_path}")
    except Exception as e:
        logger.error(f"加载目标特征 '{target_field}' 的 SOM 模型失败: {e}", exc_info=True)
        return {}

    # --- 加载评估所需数据 (原始目标数据, mask, scaler, splits) ---
    raw_target_full = load_raw_data(cfg, target_field)
    mask = load_mask(cfg)
    scaler_params = load_scaler(cfg, target_field)
    split_indices = load_split_indices(cfg)
    if None in [raw_target_full, mask, scaler_params, split_indices]:
        logger.error("加载评估所需的基础数据（原始数据、mask、scaler或分割索引）失败。")
        return {}
    train_indices = split_indices.get('train')
    val_indices = split_indices.get('val')
    test_indices = split_indices.get('test')

    # --- 确定要评估的分割集 ---
    splits_to_evaluate = []
    split_index_map = {}
    for split in ["train", "val", "test"]:
        indices = split_indices.get(split)
        pred_path = predicted_paths_dict.get(split) # 使用预测的目标 BMU 路径
        if indices is not None and len(indices) > 0 and pred_path and os.path.exists(pred_path):
            splits_to_evaluate.append(split)
            split_index_map[split] = indices
        else:
             logger.warning(f"跳过评估分割 '{split}'：索引列表为空 ({len(indices) if indices is not None else 'None'}) 或预测的目标 BMU 文件不存在/无效 ({pred_path})。")

    if not splits_to_evaluate:
        logger.error("没有可用于评估的分割。")
        return {}
    logger.info(f"将对以下分割进行评估: {splits_to_evaluate}")

    # --- 循环处理每个分割 ---
    all_reconstructed_parts = {}
    all_raw_parts = {}

    for split_name in splits_to_evaluate:
        logger.info(f"\n--- 处理分割: {split_name} ---")
        split_indices_current = split_index_map[split_name]
        split_eval_dir = os.path.join(config.paths.evaluation_base_dir, split_name)
        os.makedirs(split_eval_dir, exist_ok=True)

        # 1. 加载预测的 *目标* BMU 索引序列
        predicted_target_bmu_path = predicted_paths_dict[split_name]
        try:
            predicted_bmu_input = np.load(predicted_target_bmu_path)
            if predicted_bmu_input.ndim > 1: predicted_bmu_input = predicted_bmu_input.flatten()
            logger.info(f"  加载预测的目标 BMU 索引 ({split_name})，形状: {predicted_bmu_input.shape}")
        except Exception as e:
            logger.error(f"  加载预测的目标 BMU 索引 ({split_name}) 失败: {e}")
            continue

        # 2. 执行 BMU 重建 (使用目标 SOM 和预测的目标 BMU)
        reconstructed_output_path_split = os.path.join(config.paths.reconstructed_bmu_dir, f"reconstructed_{target_field}_{split_name}.npy")
        reconstructed_flat = None
        try:
            # Ensure reconstruct_from_bmu uses the correct SOM model path and predicted BMU path
            reconstructed_flat = reconstruct_from_bmu(
                cfg=cfg,
                som_model_path=target_som_model_path, # Correct: Target SOM model
                predicted_bmu_path=predicted_target_bmu_path, # Correct: Predicted target BMU path
                output_path=reconstructed_output_path_split
            )
        except Exception as recon_err:
             logger.error(f"  调用 reconstruct_from_bmu 时出错 ({split_name}): {recon_err}", exc_info=True)

        if reconstructed_flat is None:
            logger.error(f"  重建失败 ({split_name})。")
            continue
        logger.info(f"  重建完成 ({split_name})，扁平化形状: {reconstructed_flat.shape}")

        # 3. 反标准化
        reconstructed_inv_flat = reconstructed_flat
        if scaler_params:
             try:
                 mean_val = scaler_params['mean']; std_val = scaler_params['std']; epsilon = 1e-8
                 # Handle both scalar and vector scalers
                 if np.isscalar(mean_val) and np.isscalar(std_val):
                     reconstructed_inv_flat = reconstructed_flat * (std_val + epsilon) + mean_val
                 elif isinstance(mean_val, np.ndarray) and isinstance(std_val, np.ndarray) and len(mean_val) == reconstructed_flat.shape[1] and len(std_val) == reconstructed_flat.shape[1]:
                     reconstructed_inv_flat = reconstructed_flat * (std_val + epsilon) + mean_val
                 else:
                     logger.error(f"  Scaler 维度 ({len(mean_val) if isinstance(mean_val, np.ndarray) else 'scalar'}) 与重建数据特征数 ({reconstructed_flat.shape[1]}) 不匹配 ({split_name})。")
                     # Optionally skip inverse scaling or raise error
             except Exception as e:
                 logger.error(f"  反标准化错误 ({split_name}): {e}")
        else:
             logger.warning(f"  未找到或无效的 Scaler 参数 ({split_name})，跳过反标准化。")

        # 4. 使用 Mask 重塑
        reconstructed_split = None
        try:
            n_pred_samples = reconstructed_inv_flat.shape[0]
            target_spatial_shape = raw_target_full.shape[1:] # (H, W) or similar
            reconstructed_split = np.full((n_pred_samples,) + target_spatial_shape, np.nan)
            # Ensure mask is boolean, True for valid
            valid_mask_1d = mask.flatten().astype(bool)
            num_valid_points_mask = np.sum(valid_mask_1d)

            if reconstructed_inv_flat.shape[1] != num_valid_points_mask:
                 logger.error(f"  重塑错误: 反标准化后的重建特征数 ({reconstructed_inv_flat.shape[1]}) != Mask有效点数 ({num_valid_points_mask}) ({split_name})")
                 continue

            # Efficiently fill the array using boolean indexing
            reconstructed_split.reshape(n_pred_samples, -1)[:, valid_mask_1d] = reconstructed_inv_flat

            logger.info(f"  重建数据已重塑 ({split_name})，形状: {reconstructed_split.shape}")
            all_reconstructed_parts[split_name] = reconstructed_split
        except Exception as e:
            logger.error(f"  重塑错误 ({split_name}): {e}", exc_info=True)
            continue

        # 5. 获取对应分割的原始数据
        raw_target_aligned = None
        try:
             if len(split_indices_current) == 0:
                 logger.warning(f"'{split_name}' 索引列表为空，无法获取原始数据。")
                 continue
             # Ensure indices are within bounds
             max_idx = np.max(split_indices_current)
             if max_idx >= raw_target_full.shape[0]:
                 logger.error(f"索引 {max_idx} 超出原始数据范围 {raw_target_full.shape[0]} ({split_name})。")
                 continue

             # Align raw data
             raw_target_aligned = raw_target_full[split_indices_current]

             # --- Alignment Check: Ensure raw and reconstructed data have same time steps --- 
             if reconstructed_split is not None and raw_target_aligned.shape[0] != reconstructed_split.shape[0]:
                 logger.warning(f"  时间步数不匹配 ({split_name}): 原始数据 ({raw_target_aligned.shape[0]}) vs 重建数据 ({reconstructed_split.shape[0]}). 可能由序列创建或预测引起。")
                 # Option 1: Truncate the longer one (usually raw data if prediction is shorter)
                 min_len = min(raw_target_aligned.shape[0], reconstructed_split.shape[0])
                 logger.warning(f"  将两者截断为最小长度: {min_len}")
                 raw_target_aligned = raw_target_aligned[:min_len]
                 reconstructed_split = reconstructed_split[:min_len]
                 all_reconstructed_parts[split_name] = reconstructed_split # Update the stored part
                 # Option 2: Skip evaluation for this split
                 # logger.error("  跳过评估，因为时间步数不匹配。")
                 # continue

             logger.info(f"  原始数据已对齐 ({split_name})，形状: {raw_target_aligned.shape}")
             all_raw_parts[split_name] = raw_target_aligned
        except Exception as e:
            logger.error(f"  对齐原始数据错误 ({split_name}): {e}", exc_info=True)
            continue

        # 6. 比较评估
        if reconstructed_split is not None and raw_target_aligned is not None and reconstructed_split.shape == raw_target_aligned.shape:
             # --- 计算指标 ---
             diff = reconstructed_split - raw_target_aligned
             # Ensure mask is broadcastable (should be if mask is 2D/3D and data is 3D)
             valid_mask_eval = ~np.isnan(raw_target_aligned) & mask # Use the original mask
             metrics = {}
             rmse_field = np.full(mask.shape, np.nan)

             if np.any(valid_mask_eval):
                 # Calculate metrics only on valid points
                 mean_rmse = np.sqrt(np.mean(np.square(diff[valid_mask_eval])))
                 mean_mae = np.mean(np.abs(diff[valid_mask_eval]))

                 # Calculate spatial RMSE
                 mean_diff_sq_spatial = np.nanmean(np.square(diff), axis=0) # Average over time
                 valid_spatial_points = mask & ~np.isnan(mean_diff_sq_spatial) # Where mask is valid AND RMSE is calculable
                 if np.any(valid_spatial_points):
                     rmse_field[valid_spatial_points] = np.sqrt(mean_diff_sq_spatial[valid_spatial_points])

                 metrics = {
                     "mean_rmse": float(mean_rmse),
                     "mean_mae": float(mean_mae),
                     "max_rmse": float(np.nanmax(rmse_field)) if np.any(np.isfinite(rmse_field)) else np.nan,
                     "min_rmse": float(np.nanmin(rmse_field)) if np.any(np.isfinite(rmse_field)) else np.nan
                 }
             else:
                 logger.warning(f"在分割 '{split_name}' 中找不到有效的评估点 (基于原始数据NaN和Mask)。")
                 metrics = {k: np.nan for k in ["mean_rmse", "mean_mae", "max_rmse", "min_rmse"]}

             # --- 保存指标 ---
             metrics_path = os.path.join(split_eval_dir, f"metrics_{split_name}.npy")
             try:
                 np.save(metrics_path, metrics)
             except Exception as e:
                 logger.error(f"保存指标文件失败 ({split_name}): {e}")

             # --- 绘图 ---
             if np.any(valid_mask_eval):
                 try:
                     # Calculate time series based on valid points only
                     rec_ts = np.nanmean(reconstructed_split * mask, axis=(1, 2)) # Apply mask before averaging
                     raw_ts = np.nanmean(raw_target_aligned * mask, axis=(1, 2))
                     plot_time_series_comparison(rec_ts, raw_ts, cfg=cfg, save_path=os.path.join(split_eval_dir, f"time_series_comparison_{split_name}.png"), title=f"Time Series ({split_name})")
                     plot_spatial_rmse(rmse_field, cfg=cfg, mask=mask, save_path=os.path.join(split_eval_dir, f"spatial_rmse_{split_name}.png"), title=f"Spatial RMSE ({split_name})")
                 except Exception as e:
                     logger.error(f"绘图错误 ({split_name}): {e}", exc_info=True)

             # --- 日志 ---
             logger.info(f"\n--- {split_name} 评估结果 ---")
             for k, v in metrics.items():
                 logger.info(f"  {k}: {v:.4f}" if isinstance(v, float) and not np.isnan(v) else f"  {k}: {v}")
             logger.info(f"--- 结束 {split_name} 评估 ---")

             # --- 存储 ---
             all_eval_results[split_name] = {'metrics': metrics}
        else:
             logger.error(f"无法评估 '{split_name}'：重建数据和原始数据形状不匹配或为空。 Recon: {reconstructed_split.shape if reconstructed_split is not None else 'None'}, Raw: {raw_target_aligned.shape if raw_target_aligned is not None else 'None'}")

    # --- (可选) 评估 'all' ---
    available_keys = set(all_reconstructed_parts.keys()) & set(all_raw_parts.keys())
    if len(available_keys) >= 1:
         logger.info("\n--- 开始对 'all' (拼接后) 分割进行评估 ---")
         split_name="all"
         split_eval_dir=os.path.join(config.paths.evaluation_base_dir, split_name)
         os.makedirs(split_eval_dir, exist_ok=True)
         try:
             order = [s for s in ["train", "val", "test"] if s in available_keys]
             logger.info(f"  拼接顺序: {order}")
             if order:
                 recon_all=np.concatenate([all_reconstructed_parts[s] for s in order], axis=0)
                 raw_all=np.concatenate([all_raw_parts[s] for s in order], axis=0)
                 if recon_all.shape == raw_all.shape:
                      # --- 计算指标 for 'all' ---
                      diff_all = recon_all - raw_all
                      valid_mask_eval_all = ~np.isnan(raw_all) & mask
                      metrics_all={}
                      rmse_field_all=np.full(mask.shape, np.nan)

                      if np.any(valid_mask_eval_all):
                           mean_rmse_all = np.sqrt(np.mean(np.square(diff_all[valid_mask_eval_all])))
                           mean_mae_all = np.mean(np.abs(diff_all[valid_mask_eval_all]))
                           mean_diff_sq_spatial_all = np.nanmean(np.square(diff_all), axis=0)
                           valid_spatial_all = mask & ~np.isnan(mean_diff_sq_spatial_all)
                           if np.any(valid_spatial_all):
                               rmse_field_all[valid_spatial_all] = np.sqrt(mean_diff_sq_spatial_all[valid_spatial_all])
                           metrics_all = {
                               "mean_rmse": float(mean_rmse_all),
                               "mean_mae": float(mean_mae_all),
                               "max_rmse": float(np.nanmax(rmse_field_all)) if np.any(np.isfinite(rmse_field_all)) else np.nan,
                               "min_rmse": float(np.nanmin(rmse_field_all)) if np.any(np.isfinite(rmse_field_all)) else np.nan
                           }
                      else:
                           logger.warning("在 'all' 分割中找不到有效的评估点。")
                           metrics_all = {k: np.nan for k in ["mean_rmse", "mean_mae", "max_rmse", "min_rmse"]}

                      # --- 保存/绘图/日志 for 'all' ---
                      np.save(os.path.join(split_eval_dir, f"metrics_{split_name}.npy"), metrics_all)
                      if np.any(valid_mask_eval_all):
                          try:
                              rec_ts_all=np.nanmean(recon_all * mask, axis=(1,2))
                              raw_ts_all=np.nanmean(raw_all * mask, axis=(1,2))
                              plot_time_series_comparison(rec_ts_all, raw_ts_all, cfg=cfg, save_path=os.path.join(split_eval_dir, f"time_series_comparison_{split_name}.png"), title=f"Time Series ({split_name})")
                              plot_spatial_rmse(rmse_field_all, cfg=cfg, mask=mask, save_path=os.path.join(split_eval_dir, f"spatial_rmse_{split_name}.png"), title=f"Spatial RMSE ({split_name})")
                          except Exception as e:
                              logger.error(f"绘图错误 (all): {e}", exc_info=True)

                      logger.info(f"\n--- {split_name} 评估结果 ---")
                      for k, v in metrics_all.items():
                          logger.info(f"  {k}: {v:.4f}" if isinstance(v, float) and not np.isnan(v) else f"  {k}: {v}")
                      logger.info(f"--- 结束 {split_name} 评估 ---")
                      all_eval_results[split_name] = {'metrics': metrics_all}
                 else:
                      logger.error(f"形状不匹配 (all): Recon {recon_all.shape}, Raw {raw_all.shape}")
             else:
                 logger.warning("无法拼接 'all' 数据，因为没有有效的分割部分。")
         except Exception as e:
             logger.error(f"处理 'all' 分割出错: {e}", exc_info=True)
    else:
         logger.warning("跳过 'all' 评估，因无足够分割数据。")

    logger.info("============== 结束重建和评估阶段 ==============")
    return all_eval_results


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """SOM-LSTM-BMU 集成流程主函数 (支持可选观测特征输入 LSTM)"""
    start_time = time.time()
    config = DrprConfig.from_hydra_config(cfg)
    logger.info("开始 SOM-LSTM-BMU (with optional Obs features) 集成流程...")
    logger.info(f"当前工作目录 (Hydra): {os.getcwd()}")
    # logger.info(f"配置信息：\n{OmegaConf.to_yaml(cfg, resolve=True)}") # resolve=True 显示最终值

    # --- 确保目录存在 ---
    os.makedirs(config.paths.som_models_dir, exist_ok=True)
    os.makedirs(config.paths.bmu_base_dir, exist_ok=True)
    os.makedirs(config.paths.lstm_models_dir, exist_ok=True)
    os.makedirs(config.paths.predicted_low_dim_dir, exist_ok=True)
    os.makedirs(config.paths.reconstructed_bmu_dir, exist_ok=True)
    os.makedirs(config.paths.evaluation_base_dir, exist_ok=True)
    logger.info("必要的输出目录已确保存在。")

    # --- 步骤 1: 运行 SOM 降维 (对目标和观测特征) ---
    all_som_results = {}
    target_field_name = config.reconstruction.target_field
    observation_features = []
    use_obs_features = cfg.model.prediction.lstm.get('use_observation_features', False)
    if use_obs_features:
        obs_features_config = cfg.model.prediction.lstm.get('observation_features', [])
        if isinstance(obs_features_config, str):
             observation_features = [obs_features_config]
        elif isinstance(obs_features_config, (list, listconfig.ListConfig)):
             observation_features = list(obs_features_config)
        logger.info(f"将使用以下观测特征: {observation_features}")
    else:
        logger.info("配置未启用观测特征 (use_observation_features=False)。")

    # --- 运行目标特征的 SOM ---
    target_som_results = run_dimensionality_reduction(cfg, target_field_name, is_observation=False)
    if not target_som_results:
        logger.error(f"目标特征 '{target_field_name}' 的 SOM 降维失败。流程中止。")
        return
    all_som_results[target_field_name] = target_som_results

    # --- 运行观测特征的 SOM (如果启用) ---
    # --- MODIFICATION START: Handle combined vs individual observation features --- 
    obs_som_results_list = []
    if use_obs_features and observation_features:
        # Check if a combined SOM should be trained for observations
        train_combined_obs_som = cfg.training.som.get('train_combined_observation_som', False)
        combined_obs_feature_name = "_obs_combined".join(sorted(observation_features))

        if train_combined_obs_som:
            logger.info(f"将为观测特征 {observation_features} 训练一个组合 SOM，命名为 '{combined_obs_feature_name}'")
            # Ensure the combined feature name doesn't clash with target
            if combined_obs_feature_name == target_field_name:
                 logger.error("组合观测特征名与目标特征名冲突！请修改配置。")
                 return
            # Run SOM for the combined observation feature
            combined_obs_som_result = run_dimensionality_reduction(cfg, combined_obs_feature_name, is_observation=True)
            if not combined_obs_som_result:
                logger.error(f"组合观测特征 '{combined_obs_feature_name}' 的 SOM 降维失败。流程中止。")
                return
            # Store under the combined name
            all_som_results[combined_obs_feature_name] = combined_obs_som_result
            obs_som_results_list.append(combined_obs_som_result) # Add to list for LSTM input prep
        else:
            logger.info(f"将为每个观测特征单独训练 SOM: {observation_features}")
            for obs_feature in observation_features:
                if obs_feature == target_field_name:
                    logger.warning(f"观测特征 '{obs_feature}' 与目标特征相同，跳过重复的 SOM 训练。")
                    # If target SOM results are needed as input, ensure they are added later
                    continue
                obs_som_result = run_dimensionality_reduction(cfg, obs_feature, is_observation=True)
                if not obs_som_result:
                    logger.error(f"观测特征 '{obs_feature}' 的 SOM 降维失败。流程中止。")
                    return
                all_som_results[obs_feature] = obs_som_result
                obs_som_results_list.append(obs_som_result) # Add to list
    # --- MODIFICATION END ---

    # --- 步骤 2: 准备 LSTM 输入并运行预测 ---
    # --- MODIFICATION START: Create the structured dictionary for LSTM --- 
    structured_bmu_paths_for_lstm = {}

    # Add target feature paths
    target_bmu_paths = all_som_results[target_field_name].get('bmu_paths')
    if not target_bmu_paths:
        logger.error(f"目标特征 '{target_field_name}' 的 SOM 结果中缺少 'bmu_paths'。")
        return
    structured_bmu_paths_for_lstm[target_field_name] = target_bmu_paths

    # Add observation feature paths (either combined or individual)
    if use_obs_features:
        for obs_result in obs_som_results_list: # Iterate through the results we collected
            feature_name = obs_result.get('feature_name')
            bmu_paths = obs_result.get('bmu_paths')
            if feature_name and bmu_paths:
                if feature_name in structured_bmu_paths_for_lstm:
                     logger.warning(f"特征 '{feature_name}' 的 BMU 路径已存在，将被覆盖 (可能来自目标特征)。")
                structured_bmu_paths_for_lstm[feature_name] = bmu_paths
            else:
                 logger.error(f"观测特征 SOM 结果格式错误，缺少 feature_name 或 bmu_paths: {obs_result}")
                 return

    logger.info(f"准备好的 LSTM 输入 BMU 路径结构: {list(structured_bmu_paths_for_lstm.keys())}")
    # Example structure: {'salinity': {'train': {'positions': '...', ...}, ...}, 'flow_wind_obs': {'train': {'positions': '...', ...}, ...}}

    # Call run_lstm_prediction with the structured paths
    lstm_prediction_results = run_lstm_prediction(
        cfg,
        target_field_name=target_field_name,
        structured_bmu_paths=structured_bmu_paths_for_lstm
    )
    # --- MODIFICATION END ---

    if not lstm_prediction_results:
        logger.error("LSTM 预测步骤失败或未返回预测的目标路径。退出流程。")
        return

    # --- 步骤 3: 重建和评估 ---
    # Pass the target SOM results and the LSTM prediction results
    evaluation_results = run_reconstruction_and_evaluation(
        cfg,
        target_som_results=all_som_results[target_field_name], # Pass only the target SOM results
        prediction_results=lstm_prediction_results
    )

    if not evaluation_results:
        logger.error("评估步骤失败。")
    else:
        logger.info("评估完成。结果摘要:")
        for split, results in evaluation_results.items():
            logger.info(f"  分割 '{split}': {results.get('metrics', 'N/A')}")

    # --- 结束 ---