# src/run_pipeline.py

import os
import sys
import logging
import time
import hydra
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
from src.evaluation.visualization import plot_time_series_comparison, plot_spatial_rmse
from src.utils.model_utils import get_device_from_config  
from src.dimensionality_reduction.som_pytorch import SOMTorch
# 导入集中化的数据加载函数
from src.utils.data_loader import load_raw_data, load_mask, load_scaler, load_split_indices

# --- 导入训练函数 ---
# SOM
from src.training.train_som import train_single_feature_som, train_combined_feature_som
# HMM (可能需要修改以处理连续输入)
from src.training.train_hmm import main as train_hmm_main
# LSTM (重构后的版本)
from src.training.train_lstm import train_and_predict_lstm
# PCA (新)
from src.training.train_pca import train_and_transform_pca
# Autoencoder (新)
from src.training.train_autoencoder import train_and_transform_ae
# 重建
from src.reconstruction.reconstruct_bmu import reconstruct_from_bmu # 用于基于 SOM 的重建
# 需要一个用于 PCA/AE 基于逆变换的重建函数
logger = logging.getLogger(__name__)

# run_single_feature_som_training 函数保持不变或微调以接受 feature_type
def run_single_feature_som_training(cfg: DictConfig, feature_name: str) -> Dict[str, Any]:
    """运行单个特征的 SOM 训练流程"""
    logger.info(f"开始为特征 '{feature_name}' 训练 SOM...")
    som_results = train_single_feature_som(cfg, feature_name=feature_name)
    return som_results

# run_combined_som_training 函数保持不变

def run_combined_som_training(cfg: DictConfig, output_feature_name: str = "wind_flow") -> Dict[str, Any]:
    """运行组合特征 (wind + flow) 的 SOM 训练流程"""
    logger.info(f"开始为组合特征 '{output_feature_name}' 训练 SOM...")
    som_results = train_combined_feature_som(cfg, output_feature_name=output_feature_name)
    return som_results


def run_hmm_training_and_prediction(
    cfg: DictConfig,
    state_som_model_path: str, # 状态 SOM 模型路径
    obs_som_model_path: str    # 观测 SOM 模型路径
) -> Dict[str, Any]:
    """运行 HMM 训练和预测流程"""

    logger.info("开始 HMM 训练 (Train only) 和预测 (Train, Val, Test) 阶段...")
    config = DrprConfig.from_hydra_config(cfg) # 获取配置

    # 构建 HMM 参数保存路径
    try:
         observation_features = list(cfg.model.prediction.hmm.observation_features)
         obs_feature_name = "_".join(sorted(observation_features))
    except Exception as e:
         logger.error(f"无法从配置确定观测特征名称: {e}")
         obs_feature_name = "unknown"
    hmm_param_filename = f"hmm_params_obs_{obs_feature_name}.pkl"
    param_save_path = os.path.join(config.paths.hmm_params_dir, hmm_param_filename)

    # 构建预测状态保存路径模式 (相对于 Hydra 运行目录的 predictions 目录)
    predicted_state_save_pattern = "predicted_salinity_states_{split}.npy" # 相对于 prediction_base_dir

    # 调用修改后的 train_hmm.main (不再传递 bmu_dir 参数)
    hmm_results = train_hmm_main(
        cfg,
        state_som_model_path=state_som_model_path,
        obs_som_model_path=obs_som_model_path,
        param_save_path=param_save_path,
        predicted_state_save_pattern=predicted_state_save_pattern
    )

    # (可选) 复制/链接参数文件为固定名称 hmm_params.pkl
    fixed_hmm_param_path = config.paths.hmm_params_standard_file # 获取标准路径
    try:
        import shutil
        if hmm_results and 'model_path' in hmm_results and os.path.exists(hmm_results['model_path']):
            logger.info(f"创建固定名称的HMM参数文件: {fixed_hmm_param_path}")
            os.makedirs(os.path.dirname(fixed_hmm_param_path), exist_ok=True)
            shutil.copy2(hmm_results['model_path'], fixed_hmm_param_path)
            logger.info(f"成功将 {os.path.basename(hmm_results['model_path'])} 复制为 hmm_params.pkl")
            # 将标准路径也加入结果，方便后续直接使用
            hmm_results['standard_model_path'] = fixed_hmm_param_path
        elif hmm_results: # 如果有结果但没有 model_path
             logger.warning("HMM 流程返回结果但缺少 'model_path'，无法创建标准参数文件。")
        else: # 如果没有结果
             logger.warning("HMM 流程未成功返回结果，无法创建标准参数文件。")

    except Exception as e:
        logger.warning(f"创建固定名称的HMM参数文件失败: {e}")

    return hmm_results # 返回包含预测路径的字典

def run_reconstruction_and_evaluation(
    cfg: DictConfig,
    som_results: Dict[str, Any],
    hmm_results: Dict[str, Any]
) -> Dict[str, Any]:
    logger.info("==============开始重建和评估阶段 (Train, Val, Test) ==============")
    all_eval_results = {}
    config = DrprConfig.from_hydra_config(cfg)
    device = get_device_from_config(cfg)

    # --- 检查 HMM 预测结果 ---
    if not hmm_results or 'predicted_states_paths' not in hmm_results or not hmm_results['predicted_states_paths']:
        logger.error("HMM 结果不包含有效的预测状态路径。无法评估。")
        return {}
    predicted_paths_dict = hmm_results['predicted_states_paths']

    # --- 加载 Salinity SOM 模型 ---
    salinity_som_model_path = som_results.get('model_path')
    if not salinity_som_model_path or not os.path.exists(salinity_som_model_path):
         logger.error(f"Salinity SOM 模型路径无效: {salinity_som_model_path}")
         return {}
    try:
         salinity_som_model = SOMTorch.load(salinity_som_model_path, device=device)
    except Exception as e:
         logger.error(f"加载 Salinity SOM 模型失败: {e}")
         return {}


    # --- 使用集中式加载函数加载数据 ---
    # 1. 加载原始高维盐度数据 
    raw_salt_full = load_raw_data(cfg, "salinity")
    if raw_salt_full is None:
        logger.error("加载原始盐度数据失败")
        return {}
    logger.info(f"加载原始盐度数据，形状: {raw_salt_full.shape}")

    # 2. 加载掩码
    mask = load_mask(cfg)
    if mask is None:
        logger.error("加载掩码数据失败")
        return {}
    logger.info(f"加载掩码数据，形状: {mask.shape}")

    # 3. 加载缩放器参数
    scaler_params = load_scaler(cfg, "salinity")
    if scaler_params is None:
        logger.error("加载缩放器参数失败")
        return {}
    logger.info(f"加载盐度数据缩放器参数成功")

    # 4. 加载分割索引
    split_indices = load_split_indices(cfg)
    if split_indices is None:
        logger.error("加载分割索引失败")
        return {}

    train_indices = split_indices.get('train')
    val_indices = split_indices.get('val')
    test_indices = split_indices.get('test')
    logger.info(f"加载分割索引: Train({len(train_indices) if train_indices is not None else 'N/A'}), "
                f"Val({len(val_indices) if val_indices is not None else 'N/A'}), "
                f"Test({len(test_indices) if test_indices is not None else 'N/A'})")

    # --- 确定要评估的分割集 ---
    splits_to_evaluate = []
    split_index_map = {}
    # 只有当索引存在且对应的预测文件也存在时，才加入评估列表
    if train_indices is not None and len(train_indices) > 0 and "train" in predicted_paths_dict and os.path.exists(predicted_paths_dict["train"]):
        splits_to_evaluate.append("train")
        split_index_map["train"] = train_indices
    if val_indices is not None and len(val_indices) > 0 and "val" in predicted_paths_dict and os.path.exists(predicted_paths_dict["val"]):
        splits_to_evaluate.append("val")
        split_index_map["val"] = val_indices
    if test_indices is not None and len(test_indices) > 0 and "test" in predicted_paths_dict and os.path.exists(predicted_paths_dict["test"]):
        splits_to_evaluate.append("test")
        split_index_map["test"] = test_indices

    if not splits_to_evaluate:
        logger.error("没有有效的分割（同时具有原始索引和预测状态文件）进行评估。")
        return {}
    logger.info(f"将对以下分割进行评估: {splits_to_evaluate}")


    # --- 加载 HMM 参数 (用于等级到线性索引的转换) ---
    hmm_params_path = hmm_results.get('standard_model_path') or hmm_results.get('model_path')
    state_linear_map = None
    if hmm_params_path and os.path.exists(hmm_params_path):
        try:
            hmm_params_data = joblib.load(hmm_params_path)
            if 'state_linear_map' in hmm_params_data:
                state_linear_map = hmm_params_data['state_linear_map']
                logger.info("成功加载 HMM 状态等级 -> 线性索引映射。")
            else:
                logger.warning("HMM 参数中未找到 'state_linear_map'，将假设预测状态已是线性索引。")
        except Exception as e:
            logger.warning(f"加载 HMM 参数以获取映射失败: {e}")
    else:
        logger.warning("HMM 参数文件路径无效或不存在，将假设预测状态已是线性索引。")


    # --- 循环处理每个分割 ---
    all_reconstructed_parts = {}
    all_raw_parts = {}

    for split_name in splits_to_evaluate:
        logger.info(f"\n--- 处理分割: {split_name} ---")
        split_indices = split_index_map[split_name]
        split_eval_dir = os.path.join(config.paths.evaluation_base_dir, split_name)
        os.makedirs(split_eval_dir, exist_ok=True)

        # 1. 加载预测的状态序列
        predicted_bmu_path = predicted_paths_dict[split_name] # 获取路径
        # >>>>>> 检查点 1: 文件是否存在 <<<<<<
        if not os.path.exists(predicted_bmu_path):
            logger.error(f"预测的状态文件未找到: {predicted_bmu_path}")
            continue # 如果任何一个分割的文件不存在，就会跳过该分割
        try:
            predicted_bmu_input = np.load(predicted_bmu_path) # >>>>>> 检查点 2: 文件是否能加载 <<<<<<
            # ... (flattening) ...
            logger.info(f"  加载预测状态 ({split_name})，形状: {predicted_bmu_input.shape}")
        except Exception as e:
            logger.error(f"  加载预测状态 ({split_name}) 失败: {e}")
            continue # 如果任何一个分割的文件加载失败，就会跳过该分割


        # 2. 执行重建
        reconstructed_output_path_split = os.path.join(config.paths.reconstructed_bmu_dir, f"reconstructed_{config.reconstruction.target_field}_{split_name}.npy")
        reconstructed_flat = reconstruct_from_bmu(
            cfg,
            som_model_path=salinity_som_model_path,
            predicted_bmu_path=predicted_bmu_path,
            output_path=reconstructed_output_path_split,
            hmm_params_path=hmm_params_path
        )
        if reconstructed_flat is None:
            logger.error(f"  重建失败 ({split_name})。")
            continue
        # >>>>>> 检查点 4: 重建是否成功返回 <<<<<<
        if reconstructed_flat is None:
            logger.error(f"  重建失败 ({split_name})。")
            continue # 如果任何一个分割重建失败，就会跳过该分割

        # 3. 反标准化
        reconstructed_inv_flat = reconstructed_flat # 默认值
        if scaler_params:
            try:
                mean_val = scaler_params['mean']
                std_val = scaler_params['std']
                epsilon = 1e-8
                if np.isscalar(mean_val):
                     reconstructed_inv_flat = reconstructed_flat * (std_val + epsilon) + mean_val
                elif len(mean_val) == reconstructed_flat.shape[1]:
                     reconstructed_inv_flat = reconstructed_flat * (std_val + epsilon) + mean_val
                else:
                     logger.error(f"  Scaler 特征数 ({len(mean_val)}) 与重建特征数 ({reconstructed_flat.shape[1]}) 不匹配。跳过反标准化。")
            except Exception as inv_trans_err:
                logger.error(f"  反标准化过程中发生错误 ({split_name}): {inv_trans_err}")
        else:
             logger.warning(f"  Scaler 参数无效或未加载 ({split_name})，跳过反标准化。")
        if not 'reconstructed_inv_flat' in locals(): reconstructed_inv_flat = reconstructed_flat # 确保变量存在


        # 4. 使用 Mask 重塑
        reconstructed_split = None
        try:
            n_pred_samples = reconstructed_inv_flat.shape[0]
            target_spatial_shape = raw_salt_full.shape[1:]
            reconstructed_split = np.full((n_pred_samples,) + target_spatial_shape, np.nan)
            flat_mask = mask.flatten()
            for t in range(n_pred_samples):
                temp_flat = np.full(mask.size, np.nan)
                temp_flat[flat_mask] = reconstructed_inv_flat[t]
                reconstructed_split[t] = temp_flat.reshape(target_spatial_shape)
            logger.info(f"  重建数据已重塑 ({split_name})，形状: {reconstructed_split.shape}")
            all_reconstructed_parts[split_name] = reconstructed_split # 存储起来供 'all' 使用

        except Exception as reshape_err:
            logger.error(f"  使用 mask 重塑重建数据时出错 ({split_name}): {reshape_err}", exc_info=True)
            continue
        if 'reconstructed_split' not in locals() or reconstructed_split is None: continue # 如果重塑失败则跳过

        # 5. 获取对应分割的原始数据用于评估 *** 使用正确的索引 ***
        try:
             # 检查索引边界
             if len(split_indices) == 0:
                  logger.warning(f"'{split_name}' 的索引列表为空，无法对齐原始数据。")
                  continue
             max_index_needed = np.max(split_indices)
             if max_index_needed >= raw_salt_full.shape[0]:
                  logger.error(f"错误: '{split_name}' 集所需最大索引 {max_index_needed} 超出原始数据范围 {raw_salt_full.shape[0]}。")
                  continue

             raw_salt_aligned = raw_salt_full[split_indices] # 使用正确的索引切片
             logger.info(f"  原始数据已对齐 ({split_name})，形状: {raw_salt_aligned.shape}")
             all_raw_parts[split_name] = raw_salt_aligned
        except IndexError as e:
             logger.error(f"  对齐原始数据时发生索引错误 ({split_name}): {e}. Max index needed: {np.max(split_indices) if len(split_indices)>0 else 'N/A'}, Raw shape: {raw_salt_full.shape}")
             continue
        except Exception as e:
             logger.error(f"  对齐原始数据时发生未知错误 ({split_name}): {e}", exc_info=True)
             continue
        if 'raw_salt_aligned' not in locals() or raw_salt_aligned is None: continue # 如果对齐失败则跳过


        # 6. 比较评估
        if reconstructed_split.shape == raw_salt_aligned.shape:
            # --- *** 计算指标 *** --- <--- 添加/恢复这部分代码
            diff = reconstructed_split - raw_salt_aligned
            valid_points_mask_3d = ~np.isnan(raw_salt_aligned)

            metrics = {} # 初始化 metrics 字典
            if np.any(valid_points_mask_3d):
                diff_sq_masked = np.square(diff[valid_points_mask_3d])
                mean_rmse_val = np.sqrt(np.mean(diff_sq_masked))
                mae_val = np.mean(np.abs(diff[valid_points_mask_3d]))

                rmse_field = np.full(mask.shape, np.nan)
                if reconstructed_split.shape[1:] == mask.shape:
                     mean_diff_sq_spatial = np.nanmean(np.square(diff), axis=0)
                     valid_spatial_points = mask & ~np.isnan(mean_diff_sq_spatial)
                     if np.any(valid_spatial_points):
                         rmse_field[valid_spatial_points] = np.sqrt(mean_diff_sq_spatial[valid_spatial_points])
                else:
                    logger.warning(f"RMSE 场计算 ({split_name})：形状不匹配。")

                metrics = {
                    "mean_rmse": float(mean_rmse_val),
                    "mean_mae": float(mae_val),
                    "max_rmse": float(np.nanmax(rmse_field)) if np.any(np.isfinite(rmse_field)) else np.nan,
                    "min_rmse": float(np.nanmin(rmse_field)) if np.any(np.isfinite(rmse_field)) else np.nan
                }
            else:
                logger.warning(f"'{split_name}' 分割对齐后的原始数据中没有有效点。指标为空。")
                metrics = {k: np.nan for k in ["mean_rmse", "mean_mae", "max_rmse", "min_rmse"]}
            # --- *** 结束计算指标 *** ---

            # --- 保存指标 ---
            metrics_path = os.path.join(split_eval_dir, f"metrics_{split_name}.npy")
            try:
                np.save(metrics_path, metrics)
                logger.info(f"评估指标 ({split_name}) 已保存到: {metrics_path}")
            except Exception as save_err:
                logger.error(f"保存评估指标 ({split_name}) 失败: {save_err}")

            # --- 生成图表 ---
            if np.any(valid_points_mask_3d):
                try:
                    # ... (绘图代码，使用计算出的 rmse_field) ...
                    logger.info(f"生成评估图表 ({split_name})...")
                    rec_mean_ts = np.nanmean(reconstructed_split, axis=(1, 2))
                    raw_mean_ts = np.nanmean(raw_salt_aligned, axis=(1, 2))
                    ts_save_path = os.path.join(split_eval_dir, f"time_series_comparison_{split_name}.png")
                    plot_time_series_comparison(
                        rec_mean_ts, raw_mean_ts, cfg=cfg, save_path=ts_save_path,
                        title=f"Spatial Mean Time Series Comparison ({split_name})"
                    )

                    spatial_rmse_save_path = os.path.join(split_eval_dir, f"spatial_rmse_{split_name}.png")
                    plot_spatial_rmse(
                        rmse_field, cfg=cfg, mask=mask, 
                        save_path=spatial_rmse_save_path,
                        title=f"Spatial RMSE Distribution ({split_name})"
                    )
                    logger.info(f"评估图表 ({split_name}) 已保存到: {split_eval_dir}")
                except Exception as plot_err:
                     logger.error(f"生成评估图表 ({split_name}) 时出错: {plot_err}", exc_info=True)
            else:
                 logger.warning(f"跳过 '{split_name}' 的绘图，因为没有有效数据点。")


            # --- 日志输出 ---
            logger.info(f"\n--- {split_name} 评估结果 ---")
            for metric_name, value in metrics.items():
                 if isinstance(value, (float, np.floating)):
                    logger.info(f"  {metric_name}: {value:.4f}")
                 else:
                    logger.info(f"  {metric_name}: {value}")
            logger.info(f"--- 结束 {split_name} 评估 ---")

            # --- 存储结果 ---
            split_result_data = {'metrics': metrics} # 现在 metrics 肯定已被定义
            all_eval_results[split_name] = split_result_data
            # 存储成功处理的部分
            all_reconstructed_parts[split_name] = reconstructed_split
            all_raw_parts[split_name] = raw_salt_aligned

        else:
             logger.error(f"无法评估 '{split_name}'：重建/原始数据形状不匹配。Recon: {reconstructed_split.shape}, Raw: {raw_salt_aligned.shape}")


    # --- (可选) 评估 'all' ---
    available_keys = set(all_reconstructed_parts.keys()) & set(all_raw_parts.keys())
    if len(available_keys) >= 2:
        logger.info("\n--- 开始对 'all' (拼接后) 分割进行评估 ---")
        split_name = "all"
        split_eval_dir = os.path.join(config.paths.evaluation_base_dir, split_name)
        os.makedirs(split_eval_dir, exist_ok=True)
        logger.info(f"结果将保存到: {split_eval_dir}")

        try:
            order = [s for s in ["train", "val", "test"] if s in available_keys]
            logger.info(f"  将按顺序拼接以下部分: {order}")

            reconstructed_all = np.concatenate([all_reconstructed_parts[s] for s in order], axis=0)
            raw_salt_all = np.concatenate([all_raw_parts[s] for s in order], axis=0)
            logger.info(f"  拼接后的重建数据形状 ('all'): {reconstructed_all.shape}")
            logger.info(f"  拼接后的原始数据形状 ('all'): {raw_salt_all.shape}")

            if reconstructed_all.shape == raw_salt_all.shape:
                # --- *** 计算指标 for 'all' *** --- <--- 添加/恢复这部分代码
                diff_all = reconstructed_all - raw_salt_all
                valid_points_mask_3d_all = ~np.isnan(raw_salt_all)
                metrics_all = {} # 初始化

                if np.any(valid_points_mask_3d_all):
                    diff_sq_masked_all = np.square(diff_all[valid_points_mask_3d_all])
                    mean_rmse_val_all = np.sqrt(np.mean(diff_sq_masked_all))
                    mae_val_all = np.mean(np.abs(diff_all[valid_points_mask_3d_all]))

                    rmse_field_all = np.full(mask.shape, np.nan)
                    if reconstructed_all.shape[1:] == mask.shape:
                        mean_diff_sq_spatial_all = np.nanmean(np.square(diff_all), axis=0)
                        valid_spatial_points_all = mask & ~np.isnan(mean_diff_sq_spatial_all)
                        if np.any(valid_spatial_points_all):
                             rmse_field_all[valid_spatial_points_all] = np.sqrt(mean_diff_sq_spatial_all[valid_spatial_points_all])
                    else:
                        logger.warning(f"RMSE 场计算 (all)：形状不匹配。")

                    metrics_all = {
                        "mean_rmse": float(mean_rmse_val_all),
                        "mean_mae": float(mae_val_all),
                        "max_rmse": float(np.nanmax(rmse_field_all)) if np.any(np.isfinite(rmse_field_all)) else np.nan,
                        "min_rmse": float(np.nanmin(rmse_field_all)) if np.any(np.isfinite(rmse_field_all)) else np.nan
                    }
                else:
                    logger.warning("'all' 分割拼接后的原始数据中没有有效点。指标为空。")
                    metrics_all = {k: np.nan for k in ["mean_rmse", "mean_mae", "max_rmse", "min_rmse"]}
                # --- *** 结束计算指标 for 'all' *** ---

                # --- 保存指标 for 'all' ---
                metrics_path_all = os.path.join(split_eval_dir, f"metrics_{split_name}.npy")
                try:
                    np.save(metrics_path_all, metrics_all) # 保存 metrics_all
                    logger.info(f"评估指标 (all) 已保存到: {metrics_path_all}")
                except Exception as save_err:
                    logger.error(f"保存评估指标 (all) 失败: {save_err}")

                # --- 生成图表 for 'all' ---
                if np.any(valid_points_mask_3d_all):
                    try:
                         # ... (绘图代码，使用 rmse_field_all) ...
                         logger.info(f"生成评估图表 (all)...")
                         rec_mean_ts_all = np.nanmean(reconstructed_all, axis=(1, 2))
                         raw_mean_ts_all = np.nanmean(raw_salt_all, axis=(1, 2))
                         ts_save_path_all = os.path.join(split_eval_dir, f"time_series_comparison_{split_name}.png")
                         plot_time_series_comparison(
                             rec_mean_ts_all, raw_mean_ts_all, cfg=cfg, save_path=ts_save_path_all,
                             title=f"Spatial Mean Time Series Comparison ({split_name})"
                         )

                         spatial_rmse_save_path_all = os.path.join(split_eval_dir, f"spatial_rmse_{split_name}.png")
                         plot_spatial_rmse(
                             rmse_field_all, cfg=cfg, mask=mask,
                             save_path=spatial_rmse_save_path_all,
                             title=f"Spatial RMSE Distribution ({split_name})"
                         )
                         logger.info(f"评估图表 (all) 已保存到: {split_eval_dir}")
                    except Exception as plot_err:
                         logger.error(f"生成评估图表 (all) 时出错: {plot_err}", exc_info=True)
                else:
                     logger.warning("跳过 'all' 的绘图，因为没有有效数据点。")


                # --- 日志输出指标 for 'all' ---
                logger.info(f"\n--- {split_name} 评估结果 ---")
                for metric_name, value in metrics_all.items(): # 使用 metrics_all
                    # ... (日志输出逻辑) ...
                     if isinstance(value, (float, np.floating)):
                        logger.info(f"  {metric_name}: {value:.4f}")
                     else:
                        logger.info(f"  {metric_name}: {value}")
                logger.info(f"--- 结束 {split_name} 评估 ---")

                # --- 存储 'all' 的结果 ---
                all_eval_results[split_name] = {'metrics': metrics_all} # 使用 metrics_all

            else:
                 logger.error(f"形状不匹配（拼接后） ({split_name})：重建 {reconstructed_all.shape} vs 原始 {raw_salt_all.shape}")

        except Exception as e:
             logger.error(f"处理 'all' 分割时出错: {e}", exc_info=True)
    else:
        logger.warning("未能成功处理足够的数据分割部分，跳过 'all' 的评估。")


    logger.info("==============结束重建和评估阶段 (所有可用分割) ==============")
    return all_eval_results # 返回包含所有已评估分割结果的字典

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """集成流程主函数"""
    start_time = time.time()
    config = DrprConfig.from_hydra_config(cfg)
    logger.info("开始集成流程...")
    logger.info(f"配置信息：\n{OmegaConf.to_yaml(cfg)}")

    # 确保所有必要的目录存在
    os.makedirs(config.paths.som_models_dir, exist_ok=True)
    os.makedirs(config.paths.bmu_base_dir, exist_ok=True)
    os.makedirs(config.paths.hmm_params_dir, exist_ok=True)
    os.makedirs(config.paths.predicted_states_dir, exist_ok=True)
    os.makedirs(config.paths.reconstructed_bmu_dir, exist_ok=True)
    os.makedirs(config.paths.evaluation_base_dir, exist_ok=True)

    # --- 步骤 1: 训练 Salinity SOM 并生成 BMU (状态) ---
    salinity_feature = 'salinity'
    salinity_som_results = run_single_feature_som_training(cfg, feature_name=salinity_feature)
    if not salinity_som_results or 'model_path' not in salinity_som_results:
        logger.error("Salinity SOM 训练失败，退出流程。")
        return
    salinity_som_model_path = salinity_som_results['model_path']
    
    # --- 步骤 2: 根据配置生成 HMM 观测 BMU ---
    observation_features = list(cfg.model.prediction.hmm.observation_features)
    obs_feature_name = "_".join(sorted(observation_features))
    observation_som_results = None
    if len(observation_features) == 2 and "wind" in observation_features and "flow" in observation_features:
        observation_som_results = run_combined_som_training(cfg, output_feature_name=obs_feature_name)
    elif len(observation_features) == 1:
        cfg.training.som.map_size = cfg.training.som.map_size_obs # 仅使用第一个特征的 map_size
        observation_som_results = run_single_feature_som_training(cfg, feature_name=observation_features[0])
    else:
        logger.error(f"无效的 HMM 观测特征配置: {observation_features}")
        return

    if not observation_som_results or 'model_path' not in observation_som_results:
         logger.error(f"观测特征 '{obs_feature_name}' 的 SOM 训练失败，退出流程。")
         return
    observation_som_model_path = observation_som_results['model_path']

    # --- 步骤 3: HMM 训练 (仅用 train) 和预测 (train, val, test) ---
    # 使用更新后的 run_hmm_training_and_prediction 函数 (不再传递 BMU 目录)
    hmm_results = run_hmm_training_and_prediction(
        cfg,
        state_som_model_path=salinity_som_model_path,
        obs_som_model_path=observation_som_model_path
    )
    if not hmm_results or 'predicted_states_paths' not in hmm_results:
        logger.error("HMM 训练或预测失败，或未返回预测路径，退出流程。")
        return

    # --- 步骤 4: 重建和评估 (现在会评估 train, val, test, all) ---
    eval_results = run_reconstruction_and_evaluation(
        cfg,
        salinity_som_results, # 包含 Salinity SOM 模型路径
        hmm_results           # 包含 HMM 参数路径和预测状态路径字典
    )
    if not eval_results:
        logger.error("重建或评估失败，退出流程。")
        return

    # --- 步骤 5: 保存总体结果 ---
    final_results = {
        'config': OmegaConf.to_container(cfg, resolve=True),
        'salinity_som': salinity_som_results,
        f'{obs_feature_name}_som': observation_som_results,
        'hmm': hmm_results, # 包含参数路径和预测状态路径
        'evaluation': eval_results, # 包含 train, val, test, all 的评估结果
    }
    results_path = os.path.join(config.paths.evaluation_base_dir, "pipeline_results.pkl")
    joblib.dump(final_results, results_path)
    logger.info(f"Pipeline 结果已保存到: {results_path}")
    
    total_time = time.time() - start_time
    logger.info(f"集成流程完成，总用时: {total_time:.2f} 秒")

if __name__ == "__main__":
    main()