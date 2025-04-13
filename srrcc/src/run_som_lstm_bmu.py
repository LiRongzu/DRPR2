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
               'bmu_indices_paths': {'train': 'path/train.npy', ...}}
    """
    logger.info(f"============== 开始降维阶段 (SOM for {feature_name}{' - Observation' if is_observation else ' - Target'}) ==============")
    config = DrprConfig.from_hydra_config(cfg)

    if is_observation and hasattr(cfg.training.som, 'map_size_obs'):
        logger.info(f"使用观测特征的 SOM map_size: {cfg.training.som.map_size_obs}")
        cfg.training.som.map_size = cfg.training.som.map_size_obs
    else:
        cfg.training.som.map_size = cfg.training.som.map_size_sta

    som_results = {}
    if feature_name == config.reconstruction.target_field or "_" not in feature_name: # 假设组合特征用 '_' 连接
        logger.info(f"使用 train_single_feature_som 为 '{feature_name}' (map_size: {cfg.training.som.map_size})")
        som_results = train_single_feature_som(
            cfg,
            feature_name=feature_name
            )
    else: # 假设是组合特征检查是否真的包含 wind 和 flow (或配置中的观测特征)
         parts = feature_name.split('_')
         expected_obs = list(cfg.model.prediction.lstm.get('observation_features', [])) # 从 lstm 配置获取
         if all(p in parts for p in expected_obs) and len(parts) == len(expected_obs):
              logger.info(f"使用 train_combined_feature_som 为 '{feature_name}' (map_size: {cfg.training.som.map_size})")
              som_results = train_combined_feature_som(
                  cfg,
                  output_feature_name=feature_name
                  )
         else:
              logger.error(f"特征名 '{feature_name}' 看起来像组合特征，但与配置的观测特征 {expected_obs} 不匹配。")
              return {}


    # 恢复原始 map_size (如果之前修改了 cfg)
    # if is_observation and hasattr(cfg.training.som, 'map_size_obs'):
    #     OmegaConf.set_struct(cfg.training.som, False)
    #     cfg.training.som.map_size = original_map_size
    #     OmegaConf.set_struct(cfg.training.som, True)

    # --- 检查结果 ---
    if not som_results:
        logger.error(f"特征 '{feature_name}' 的 SOM 降维失败。")
        return {}
    if 'model_path' not in som_results or 'bmu_indices_paths' not in som_results:
         logger.error(f"SOM 降维结果 ({feature_name}) 缺少 model_path 或 bmu_indices_paths。")
         if 'bmu_paths' in som_results:
             logger.warning(f"({feature_name}) 检测到旧的 'bmu_paths' 键，假设其包含 BMU 索引路径。")
             som_results['bmu_indices_paths'] = som_results['bmu_paths']
         else:
              return {}

    som_results['feature_name'] = feature_name # 确保存储了特征名

    logger.info(f"SOM 模型 ({feature_name}) 已训练并保存。")
    logger.info(f"BMU 索引文件路径 ({feature_name}): {som_results['bmu_indices_paths']}")
    logger.info(f"============== 结束降维阶段 (SOM for {feature_name}) ==============")
    return som_results

def run_lstm_prediction(
    cfg: DictConfig,
    target_field_name: str,
    # 输入变为: Dict[feature_name, Dict[split, path]]
    low_dim_feature_paths: Dict[str, Dict[str, str]]
) -> Dict[str, Any]:
    """
    运行 LSTM 训练和预测流程。
    使用来自 target 和 (可选的) observation 特征的 BMU 索引序列作为输入。
    预测目标是 target 特征的下一个 BMU 索引。

    Args:
        cfg: Hydra 配置对象。
        target_field_name: 目标字段名 (用于命名输出文件和标识预测目标)。
        low_dim_feature_paths: 包含各特征 ('target', 'obs1', ...) 的
                                BMU 索引文件路径字典 ({feature: {split: path}})。

    Returns:
        包含预测出的 *目标* 低维序列文件路径的字典。
        例如: {'predicted_target_low_dim_paths': {'train': 'path/pred_train.npy', ...}}
               可能还包含 LSTM 模型路径等。
    """
    logger.info("============== 开始 LSTM 训练和预测阶段 ==============")
    config = DrprConfig.from_hydra_config(cfg)
    feature_names = list(low_dim_feature_paths.keys())
    num_features = len(feature_names)
    logger.info(f"LSTM 将使用以下特征的 BMU 序列作为输入: {feature_names} (共 {num_features} 个)")

    if target_field_name not in feature_names:
         logger.error(f"目标特征 '{target_field_name}' 不在提供的低维路径字典中!")
         return {}

    # 检查所有特征都有 train/val/test 路径
    required_splits = ["train", "val", "test"]
    for feature, paths in low_dim_feature_paths.items():
        if not all(s in paths for s in required_splits):
             logger.error(f"特征 '{feature}' 缺少 train/val/test 的完整 BMU 索引路径。")
             return {}

    # --- 加载并合并多特征的低维数据 ---
    # 结果: combined_low_dim_data[split] = np.array(T, num_features)
    combined_low_dim_data = {}
    expected_length = -1
    temp_save_paths = {} # 临时保存合并后的文件

    try:
        for split in required_splits:
             split_data_list = []
             logger.info(f"  处理 LSTМ 输入分割: {split}")
             for i, feature in enumerate(feature_names): # 按固定顺序加载
                 path = low_dim_feature_paths[feature][split]
                 logger.info(f"    加载特征 '{feature}' 的 BMU 索引: {path}")
                 data = np.load(path).flatten() # 确保是一维
                 logger.info(f"      加载形状: {data.shape}")

                 if expected_length == -1:
                      expected_length = len(data)
                 elif len(data) != expected_length:
                      logger.error(f"    错误: 特征 '{feature}' 的序列长度 ({len(data)}) 与预期 ({expected_length}) 不符!")
                      raise ValueError("序列长度不一致")

                 split_data_list.append(data)

             # 堆叠特征 -> (T, num_features)
             combined_split_data = np.stack(split_data_list, axis=-1)
             logger.info(f"  合并后 {split} 数据形状: {combined_split_data.shape}")
             combined_low_dim_data[split] = combined_split_data
             expected_length = -1 # 重置下一个 split 的检查

             # 将合并后的数据保存到临时文件，因为 train_and_predict_lstm 需要文件路径
             temp_dir = os.path.join(os.getcwd(), "temp_lstm_input") # Hydra 当前运行目录下的临时目录
             os.makedirs(temp_dir, exist_ok=True)
             temp_path = os.path.join(temp_dir, f"combined_lstm_input_{split}.npy")
             np.save(temp_path, combined_split_data)
             temp_save_paths[split] = temp_path
             logger.info(f"  临时保存合并后的 {split} 输入到: {temp_path}")

    except Exception as e:
        logger.error(f"加载或合并 LSTM 输入数据时出错: {e}", exc_info=True)
        # 清理可能已创建的临时文件
        if 'temp_dir' in locals() and os.path.exists(temp_dir):
             shutil.rmtree(temp_dir)
        return {}


    # --- 调用 (修改后的) LSTM 训练和预测函数 ---
    prediction_base_dir = config.paths.predicted_low_dim_dir
    os.makedirs(prediction_base_dir, exist_ok=True)
    # 注意：保存的是预测的 *目标* BMU 索引
    predicted_target_save_pattern = os.path.join(prediction_base_dir, f"predicted_lstm_target_{target_field_name}_{{split}}.npy")
    logger.info(f"LSTM 预测的 *目标* BMU 索引将保存到模式: {predicted_target_save_pattern.format(split='*')}")

    lstm_results = train_and_predict_lstm(
        cfg=cfg,
        low_dim_feature_paths=temp_save_paths,          # 传递包含合并后数据的临时文件路径
        output_save_pattern=predicted_target_save_pattern, # 保存模式
        target_field_name=target_field_name,           # 告知哪个是目标
        input_dim=num_features                         # 告知输入特征数
        # 可能需要传递 feature_names 列表给 train_and_predict_lstm
    )

    # 清理临时文件
    if 'temp_dir' in locals() and os.path.exists(temp_dir):
        logger.info(f"清理临时 LSTM 输入文件目录: {temp_dir}")
        shutil.rmtree(temp_dir)

    # --- 检查 LSTM 结果 ---
    # 返回的 key 可能需要更新为 predicted_target_low_dim_paths
    # if not lstm_results or 'predicted_low_dim_paths' not in lstm_results:
    if not lstm_results or 'predicted_target_low_dim_paths' not in lstm_results: # 假设返回 key 已更新
        logger.error("LSTM 训练或预测失败，或未返回预测的目标路径。")
        return {}

    logger.info(f"LSTM 预测完成。预测的目标 BMU 索引文件路径: {lstm_results['predicted_target_low_dim_paths']}")
    logger.info("============== 结束 LSTM 训练和预测阶段 ==============")
    return lstm_results


# --- 步骤 3: 重建和评估 ---
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
    # 确保使用正确的 key
    if not prediction_results or 'predicted_target_low_dim_paths' not in prediction_results or not prediction_results['predicted_target_low_dim_paths']:
        logger.error("预测结果不包含有效的预测目标路径 ('predicted_target_low_dim_paths')。无法评估。")
        return {}
    predicted_paths_dict = prediction_results['predicted_target_low_dim_paths'] # 这是预测的目标 BMU 路径

    # --- 加载目标字段的 SOM 模型 ---
    target_som_model_path = target_som_results.get('model_path')
    if not target_som_model_path or not os.path.exists(target_som_model_path):
        # ... (错误处理) ...
        return {}
    try:
        target_som_model = SOMTorch.load(target_som_model_path, device=device)
        # ... (日志) ...
    except Exception as e:
        # ... (错误处理) ...
        return {}

    # --- 加载评估所需数据 (原始目标数据, mask, scaler, splits) ---
    # ... (这部分代码与之前的版本完全相同，省略以保持简洁) ...
    raw_target_full = load_raw_data(cfg, target_field)
    mask = load_mask(cfg)
    scaler_params = load_scaler(cfg, target_field)
    split_indices = load_split_indices(cfg)
    if None in [raw_target_full, mask, scaler_params, split_indices]: return {}
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
             logger.warning(f"跳过评估分割 '{split}'：索引列表为空或预测的目标 BMU 文件不存在 ({pred_path})。")
    if not splits_to_evaluate: return {} # 错误信息已记录
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
            predicted_bmu_input = np.load(predicted_target_bmu_path) # 加载目标 BMU 预测
            # ... (确保是一维, 日志) ...
            if predicted_bmu_input.ndim > 1: predicted_bmu_input = predicted_bmu_input.flatten()
            logger.info(f"  加载预测的目标 BMU 索引 ({split_name})，形状: {predicted_bmu_input.shape}")
        except Exception as e:
            logger.error(f"  加载预测的目标 BMU 索引 ({split_name}) 失败: {e}")
            continue

        # 2. 执行 BMU 重建 (使用目标 SOM 和预测的目标 BMU)
        reconstructed_output_path_split = os.path.join(config.paths.reconstructed_bmu_dir, f"reconstructed_{target_field}_{split_name}.npy")
        reconstructed_flat = None
        try:
            reconstructed_flat = reconstruct_from_bmu(
                cfg=cfg,
                som_model_path=target_som_model_path, # 使用目标 SOM
                predicted_bmu_path=predicted_target_bmu_path, # 使用预测的目标 BMU 路径
                output_path=reconstructed_output_path_split
            )
        except Exception as recon_err:
             logger.error(f"  调用 reconstruct_from_bmu 时出错 ({split_name}): {recon_err}", exc_info=True)

        if reconstructed_flat is None:
            logger.error(f"  重建失败 ({split_name})。")
            continue
        logger.info(f"  重建完成 ({split_name})，扁平化形状: {reconstructed_flat.shape}")


        # 3. 反标准化 (与之前相同)
        # ... (省略代码) ...
        reconstructed_inv_flat = reconstructed_flat # 默认
        if scaler_params:
             try:
                 mean_val = scaler_params['mean']; std_val = scaler_params['std']; epsilon = 1e-8
                 if np.isscalar(mean_val): reconstructed_inv_flat = reconstructed_flat * (std_val + epsilon) + mean_val
                 elif len(mean_val) == reconstructed_flat.shape[1]: reconstructed_inv_flat = reconstructed_flat * (std_val + epsilon) + mean_val
                 else: logger.error(f"  Scaler/重建特征数不匹配 ({split_name})。")
             except Exception as e: logger.error(f"  反标准化错误 ({split_name}): {e}")
        else: logger.warning(f"  Scaler 参数无效 ({split_name})。")


        # 4. 使用 Mask 重塑 (与之前相同)
        # ... (省略代码) ...
        reconstructed_split = None
        try:
            n_pred_samples = reconstructed_inv_flat.shape[0]; target_spatial_shape = raw_target_full.shape[1:]
            reconstructed_split = np.full((n_pred_samples,) + target_spatial_shape, np.nan)
            flat_mask = mask.flatten(); valid_mask_indices = np.where(flat_mask)[0]
            if reconstructed_inv_flat.shape[1] != len(valid_mask_indices):
                 logger.error(f"  重塑错误: 重建特征数 ({reconstructed_inv_flat.shape[1]}) != Mask有效点数 ({len(valid_mask_indices)})")
                 continue
            for t in range(n_pred_samples):
                temp_flat = np.full(mask.size, np.nan); temp_flat[valid_mask_indices] = reconstructed_inv_flat[t]
                reconstructed_split[t] = temp_flat.reshape(target_spatial_shape)
            logger.info(f"  重建数据已重塑 ({split_name})，形状: {reconstructed_split.shape}")
            all_reconstructed_parts[split_name] = reconstructed_split
        except Exception as e: logger.error(f"  重塑错误 ({split_name}): {e}", exc_info=True); continue


        # 5. 获取对应分割的原始数据 (与之前相同)
        # ... (省略代码) ...
        raw_target_aligned = None
        try:
             if len(split_indices_current) == 0: logger.warning(f"'{split_name}' 索引列表为空。"); continue
             max_idx = np.max(split_indices_current)
             if max_idx >= raw_target_full.shape[0]: logger.error(f"索引 {max_idx} 超出范围 {raw_target_full.shape[0]} ({split_name})。"); continue
             raw_target_aligned = raw_target_full[split_indices_current]
             logger.info(f"  原始数据已对齐 ({split_name})，形状: {raw_target_aligned.shape}")
             all_raw_parts[split_name] = raw_target_aligned
        except Exception as e: logger.error(f"  对齐原始数据错误 ({split_name}): {e}", exc_info=True); continue


        # 6. 比较评估 (与之前相同)
        # ... (省略代码，包含计算指标、保存指标、绘图、日志输出、存储结果) ...
        if reconstructed_split is not None and raw_target_aligned is not None and reconstructed_split.shape == raw_target_aligned.shape:
             # --- 计算指标 ---
             diff = reconstructed_split - raw_target_aligned; valid_mask_3d = ~np.isnan(raw_target_aligned) & mask
             metrics = {}; rmse_field = np.full(mask.shape, np.nan)
             if np.any(valid_mask_3d):
                 mean_rmse = np.sqrt(np.mean(np.square(diff[valid_mask_3d]))); mean_mae = np.mean(np.abs(diff[valid_mask_3d]))
                 mean_diff_sq_spatial = np.nanmean(np.square(diff), axis=0); valid_spatial = mask & ~np.isnan(mean_diff_sq_spatial)
                 if np.any(valid_spatial): rmse_field[valid_spatial] = np.sqrt(mean_diff_sq_spatial[valid_spatial])
                 metrics = {"mean_rmse": float(mean_rmse), "mean_mae": float(mean_mae),
                            "max_rmse": float(np.nanmax(rmse_field)) if np.any(np.isfinite(rmse_field)) else np.nan,
                            "min_rmse": float(np.nanmin(rmse_field)) if np.any(np.isfinite(rmse_field)) else np.nan}
             else: metrics = {k: np.nan for k in ["mean_rmse", "mean_mae", "max_rmse", "min_rmse"]}
             # --- 保存指标 ---
             metrics_path = os.path.join(split_eval_dir, f"metrics_{split_name}.npy"); np.save(metrics_path, metrics)
             # --- 绘图 ---
             if np.any(valid_mask_3d):
                 try:
                     rec_ts = np.nanmean(reconstructed_split, axis=(1, 2)); raw_ts = np.nanmean(raw_target_aligned, axis=(1, 2))
                     plot_time_series_comparison(rec_ts, raw_ts, cfg=cfg, save_path=os.path.join(split_eval_dir, f"time_series_comparison_{split_name}.png"), title=f"Time Series ({split_name})")
                     plot_spatial_rmse(rmse_field, cfg=cfg, mask=mask, save_path=os.path.join(split_eval_dir, f"spatial_rmse_{split_name}.png"), title=f"Spatial RMSE ({split_name})")
                 except Exception as e: logger.error(f"绘图错误 ({split_name}): {e}", exc_info=True)
             # --- 日志 ---
             logger.info(f"\n--- {split_name} 评估结果 ---"); [logger.info(f"  {k}: {v:.4f}" if isinstance(v, float) and not np.isnan(v) else f"  {k}: {v}") for k, v in metrics.items()]; logger.info(f"--- 结束 {split_name} 评估 ---")
             # --- 存储 ---
             all_eval_results[split_name] = {'metrics': metrics}
        else: logger.error(f"无法评估 '{split_name}'：形状不匹配或数据为空。")


    # --- (可选) 评估 'all' ---
    # ... (与之前相同，省略代码) ...
    available_keys = set(all_reconstructed_parts.keys()) & set(all_raw_parts.keys())
    if len(available_keys) >= 1:
         logger.info("\n--- 开始对 'all' (拼接后) 分割进行评估 ---")
         split_name="all"; split_eval_dir=os.path.join(config.paths.evaluation_base_dir, split_name); os.makedirs(split_eval_dir, exist_ok=True)
         try:
             order = [s for s in ["train", "val", "test"] if s in available_keys]; logger.info(f"  拼接顺序: {order}")
             if order:
                 recon_all=np.concatenate([all_reconstructed_parts[s] for s in order], axis=0); raw_all=np.concatenate([all_raw_parts[s] for s in order], axis=0)
                 if recon_all.shape == raw_all.shape:
                      # --- 计算指标 for 'all' ---
                      diff_all = recon_all - raw_all; valid_mask_3d_all = ~np.isnan(raw_all) & mask; metrics_all={}; rmse_field_all=np.full(mask.shape, np.nan)
                      if np.any(valid_mask_3d_all):
                           mean_rmse_all = np.sqrt(np.mean(np.square(diff_all[valid_mask_3d_all]))); mean_mae_all = np.mean(np.abs(diff_all[valid_mask_3d_all]))
                           mean_diff_sq_spatial_all = np.nanmean(np.square(diff_all), axis=0); valid_spatial_all = mask & ~np.isnan(mean_diff_sq_spatial_all)
                           if np.any(valid_spatial_all): rmse_field_all[valid_spatial_all] = np.sqrt(mean_diff_sq_spatial_all[valid_spatial_all])
                           metrics_all = {"mean_rmse": float(mean_rmse_all), "mean_mae": float(mean_mae_all), "max_rmse": float(np.nanmax(rmse_field_all)) if np.any(np.isfinite(rmse_field_all)) else np.nan, "min_rmse": float(np.nanmin(rmse_field_all)) if np.any(np.isfinite(rmse_field_all)) else np.nan}
                      else: metrics_all = {k: np.nan for k in ["mean_rmse", "mean_mae", "max_rmse", "min_rmse"]}
                      # --- 保存/绘图/日志 for 'all' ---
                      np.save(os.path.join(split_eval_dir, f"metrics_{split_name}.npy"), metrics_all)
                      if np.any(valid_mask_3d_all):
                          try:
                              rec_ts_all=np.nanmean(recon_all,axis=(1,2)); raw_ts_all=np.nanmean(raw_all,axis=(1,2))
                              plot_time_series_comparison(rec_ts_all, raw_ts_all, cfg=cfg, save_path=os.path.join(split_eval_dir, f"time_series_comparison_{split_name}.png"), title=f"Time Series ({split_name})")
                              plot_spatial_rmse(rmse_field_all, cfg=cfg, mask=mask, save_path=os.path.join(split_eval_dir, f"spatial_rmse_{split_name}.png"), title=f"Spatial RMSE ({split_name})")
                          except Exception as e: logger.error(f"绘图错误 (all): {e}", exc_info=True)
                      logger.info(f"\n--- {split_name} 评估结果 ---"); [logger.info(f"  {k}: {v:.4f}" if isinstance(v, float) and not np.isnan(v) else f"  {k}: {v}") for k, v in metrics_all.items()]; logger.info(f"--- 结束 {split_name} 评估 ---")
                      all_eval_results[split_name] = {'metrics': metrics_all}
                 else: logger.error(f"形状不匹配 (all): Recon {recon_all.shape}, Raw {raw_all.shape}")
         except Exception as e: logger.error(f"处理 'all' 分割出错: {e}", exc_info=True)
    else: logger.warning("跳过 'all' 评估，因无足够分割数据。")


    logger.info("============== 结束重建和评估阶段 ==============")
    return all_eval_results


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """SOM-LSTM-BMU 集成流程主函数 (支持可选观测特征输入 LSTM)"""
    start_time = time.time()
    config = DrprConfig.from_hydra_config(cfg)
    logger.info("开始 SOM-LSTM-BMU (with optional Obs features) 集成流程...")
    logger.info(f"当前工作目录 (Hydra): {os.getcwd()}")
    logger.info(f"配置信息：\n{OmegaConf.to_yaml(cfg, resolve=True)}") # resolve=True 显示最终值

    # --- 确保目录存在 ---
    # ... (省略 os.makedirs 调用，与之前版本相同) ...
    os.makedirs(config.paths.som_models_dir, exist_ok=True); os.makedirs(config.paths.bmu_base_dir, exist_ok=True)
    os.makedirs(config.paths.lstm_models_dir, exist_ok=True); os.makedirs(config.paths.predicted_low_dim_dir, exist_ok=True)
    os.makedirs(config.paths.reconstructed_bmu_dir, exist_ok=True); os.makedirs(config.paths.evaluation_base_dir, exist_ok=True)


    # --- 步骤 1a: 训练 Target SOM ---
    target_field = config.reconstruction.target_field
    logger.info(f"目标字段: {target_field}")
    target_som_results = run_dimensionality_reduction(cfg, feature_name=target_field, is_observation=False)

    if not target_som_results or 'model_path' not in target_som_results or 'bmu_indices_paths' not in target_som_results:
        logger.error(f"目标 SOM ({target_field}) 降维失败。退出流程。")
        return

    # --- 步骤 1b: (可选) 训练 Observation SOM(s) ---
    observation_som_results_dict = {} # 存储观测 SOM 结果
    use_obs_features = cfg.model.prediction.lstm.get('use_observation_features', False)
    observation_features_config = list(cfg.model.prediction.lstm.get('observation_features', []))

    if use_obs_features and observation_features_config:
        logger.info(f"配置了使用观测特征: {observation_features_config}")
        obs_feature_name_combined = "_".join(sorted(observation_features_config)) # 用于组合特征

        # 判断是训练单个观测 SOM 还是组合观测 SOM
        if len(observation_features_config) >= 2: # 假设 >= 2 就合并
             logger.info(f"将为组合观测特征 '{obs_feature_name_combined}' 训练 SOM...")
             # 使用 run_dimensionality_reduction 包装调用
             obs_results = run_dimensionality_reduction(cfg, feature_name=obs_feature_name_combined, is_observation=True)
             if obs_results:
                  observation_som_results_dict[obs_feature_name_combined] = obs_results
             else:
                  logger.error(f"组合观测特征 '{obs_feature_name_combined}' SOM 训练失败。")
                  # 决定是否中止流程，这里选择继续，LSTM 将只使用 Target
                  use_obs_features = False # 标记为不使用，因为失败了

        elif len(observation_features_config) == 1: # 单个观测特征
             obs_feature_name_single = observation_features_config[0]
             logger.info(f"将为单个观测特征 '{obs_feature_name_single}' 训练 SOM...")
             # 使用 run_dimensionality_reduction 包装调用
             obs_results = run_dimensionality_reduction(cfg, feature_name=obs_feature_name_single, is_observation=True)
             if obs_results:
                  observation_som_results_dict[obs_feature_name_single] = obs_results
             else:
                  logger.error(f"单个观测特征 '{obs_feature_name_single}' SOM 训练失败。")
                  use_obs_features = False # 标记为不使用

    elif use_obs_features:
        logger.warning("配置了 use_observation_features=true 但 model.prediction.lstm.observation_features 为空。将不使用观测特征。")
        use_obs_features = False
    else:
        logger.info("未配置使用观测特征作为 LSTM 输入。")


    # --- 步骤 2: 准备 LSTM 输入路径并进行预测 ---
    lstm_input_paths = {} # 格式: {feature_name: {split: path}}

    # 始终添加 Target BMU 路径
    lstm_input_paths[target_field] = target_som_results['bmu_indices_paths']
    logger.info(f"添加 Target '{target_field}' BMU 路径到 LSTM 输入。")

    # 如果配置了且成功训练了观测 SOM，则添加观测 BMU 路径
    if use_obs_features and observation_som_results_dict:
        for obs_name, obs_results in observation_som_results_dict.items():
            if 'bmu_indices_paths' in obs_results:
                lstm_input_paths[obs_name] = obs_results['bmu_indices_paths']
                logger.info(f"添加 Observation '{obs_name}' BMU 路径到 LSTM 输入。")
            else:
                 logger.warning(f"观测特征 '{obs_name}' 的 SOM 结果中缺少 bmu_indices_paths，无法用于 LSTM 输入。")
    elif use_obs_features:
         logger.warning("配置了使用观测特征，但未能成功获取观测 SOM BMU 路径。LSTM 将仅使用 Target 输入。")


    # 调用 LSTM 预测 (它内部会处理单特征或多特征输入)
    lstm_results = run_lstm_prediction(cfg, target_field_name=target_field, low_dim_feature_paths=lstm_input_paths)

    if not lstm_results or 'predicted_target_low_dim_paths' not in lstm_results:
        logger.error("LSTM 预测步骤失败或未返回预测的目标路径。退出流程。")
        return

    # --- 步骤 3: 重建和评估 (只使用目标 SOM 和预测的目标 BMU) ---
    eval_results = run_reconstruction_and_evaluation(
        cfg,
        target_som_results, # 只传递目标 SOM 的结果
        lstm_results        # 传递 LSTM 的结果 (包含预测的目标 BMU 路径)
    )
    if not eval_results:
        logger.error("重建或评估失败，退出流程。")
        return

    # --- 步骤 4: 保存总体结果 ---
    final_results = {
        'pipeline_type': f'SOM-LSTM{"-Obs" if use_obs_features and observation_som_results_dict else ""}-BMU',
        'config': OmegaConf.to_container(cfg, resolve=True),
        'target_som': target_som_results,
        'observation_som': observation_som_results_dict, # 可能为空字典
        'lstm': lstm_results,
        'evaluation': eval_results,
    }
    results_path = os.path.join(config.paths.evaluation_base_dir, "pipeline_results.pkl") # 可以用更具体的文件名
    try:
        joblib.dump(final_results, results_path)
        logger.info(f"Pipeline 结果已保存到: {results_path}")
    except Exception as e:
        logger.error(f"保存最终 Pipeline 结果失败: {e}")

    total_time = time.time() - start_time
    logger.info(f"集成流程完成，总用时: {total_time:.2f} 秒")

if __name__ == "__main__":
    main()