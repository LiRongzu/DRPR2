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
    results = {'method': method, 'success': False} # Initialize success to False
    dr_results = None # For PCA/AE results
    primary_bmu_paths = {} # Store the main BMU paths needed for prediction

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
        state_bmu_positions = {}
        obs_bmu_positions = {}

        # --- Train State SOM (e.g., salinity) ---
        # Always run state SOM for now, needed for reconstruction
        logger.info(f"训练状态 SOM ({target_field_dr})...")
        som_state_results = train_single_feature_som(cfg, feature_name=target_field_dr)
        logger.info(f"状态 SOM 返回结果: {som_state_results}") # Log the result

        if som_state_results and isinstance(som_state_results, dict):
            results['som_state'] = som_state_results
            bmu_paths_state = som_state_results.get('bmu_indices_paths', {})
            if bmu_paths_state:
                 try:
                     # Ensure inner dict and 'positions' key exist
                     state_bmu_positions = {split: p['positions'] for split, p in bmu_paths_state.items() if isinstance(p, dict) and 'positions' in p and p['positions']}
                     logger.info(f"提取的状态 BMU 位置路径: {state_bmu_positions}")
                 except Exception as e:
                     logger.error(f"从状态 SOM 结果提取 BMU 位置路径时出错: {e}", exc_info=True)
            else:
                 logger.warning("状态 SOM 结果中未找到 'bmu_indices_paths' 或其为空。")
            results['low_dim_dv_paths'] = som_state_results.get('dv_paths', {})
            results['model_path'] = som_state_results.get('model_path') # State SOM model path for reconstruction
        else:
            logger.error("状态 SOM 训练失败或未返回有效字典。")
            # State SOM failure is likely fatal for reconstruction
            results['success'] = False
            logger.info(f"--- 降维完成 (方法: {method}) ---")
            return results


        # --- Train Observation SOM (if prediction is HMM) ---
        if cfg.model.prediction.method == 'hmm':
            observation_features = list(cfg.model.prediction.hmm.observation_features)
            if not observation_features:
                 logger.error("HMM 预测方法已选择，但未在配置中指定 observation_features。")
                 raise ValueError("HMM 需要 observation_features 配置。")

            obs_feature_name = "_".join(sorted(observation_features)) # e.g., "flow_wind"
            logger.info(f"训练观测 SOM ({obs_feature_name}) for HMM...")

            # Temporarily set map size for observation SOM from config if different
            original_map_size = cfg.training.som.map_size
            map_size_obs = cfg.training.som.get("map_size_obs", original_map_size) # Get obs size or default
            if map_size_obs != original_map_size:
                 logger.info(f"临时为观测 SOM 设置地图大小: {map_size_obs}")
                 cfg.training.som.map_size = map_size_obs # Use obs map size

            if len(observation_features) == 1:
                som_obs_results = train_single_feature_som(cfg, feature_name=observation_features[0])
            else: # Combined features
                som_obs_results = train_combined_feature_som(cfg, output_feature_name=obs_feature_name)

            # Restore original map size in config if it was changed
            if map_size_obs != original_map_size:
                 cfg.training.som.map_size = original_map_size
                 logger.info(f"恢复 SOM 地图大小为: {original_map_size}")

            logger.info(f"观测 SOM 返回结果: {som_obs_results}") # Log the result

            if som_obs_results and isinstance(som_obs_results, dict):
                results['som_observation'] = som_obs_results
                bmu_paths_obs = som_obs_results.get('bmu_indices_paths', {})
                if bmu_paths_obs:
                    try:
                        # Ensure inner dict and 'positions' key exist
                        obs_bmu_positions = {split: p['positions'] for split, p in bmu_paths_obs.items() if isinstance(p, dict) and 'positions' in p and p['positions']}
                        logger.info(f"提取的观测 BMU 位置路径: {obs_bmu_positions}")
                    except Exception as e:
                        logger.error(f"从观测 SOM 结果提取 BMU 位置路径时出错: {e}", exc_info=True)
                else:
                    logger.warning("观测 SOM 结果中未找到 'bmu_indices_paths' 或其为空。")
                results['som_observation_model_path'] = som_obs_results.get('model_path') # Store obs SOM model path
            else:
                # If HMM is the predictor, observation SOM is essential
                logger.error("观测 SOM 训练失败或未返回有效字典 (HMM 需要)。")
                results['success'] = False # Mark as failed
                logger.info(f"--- 降维完成 (方法: {method}) ---") # Log completion before returning
                return results


        if cfg.model.prediction.method == 'hmm':
            if obs_bmu_positions:
                primary_bmu_paths = obs_bmu_positions
                logger.info("将观测 SOM BMU 位置设置为主要低维数据路径 (for HMM)。")
                results['success'] = True
            else:
                logger.error("HMM 模式下未能获取观测 BMU 位置路径。")
                # Failure should have been caught above, but set success=False just in case
                results['success'] = False
        else: # Not HMM (e.g., LSTM using state BMUs)
            if state_bmu_positions:
                primary_bmu_paths = state_bmu_positions
                logger.info("将状态 SOM BMU 位置设置为主要低维数据路径。")
            else:
                logger.error("非 HMM 模式下未能获取状态 BMU 位置路径。")
                # If state SOM failed earlier, success should already be False
                results['success'] = False


        if primary_bmu_paths and results.get('success', True) is not False : # Check if paths exist AND no fatal error occurred
             results['low_dim_data_paths'] = primary_bmu_paths
             logger.info(f"最终设置的 low_dim_data_paths: {primary_bmu_paths}")
             results['success'] = True # Explicitly set success if paths are assigned
        elif results.get('success', True) is not False: # If no fatal error, but paths are missing
             logger.error("未能确定主要的低维数据路径 (BMU positions)。")
             results['success'] = False # Mark failure if paths are missing


    elif method == 'pca':
        # ... (PCA logic remains the same) ...
        # Ensure PCA logic sets results['success'] and results['low_dim_data_paths']
        if 'train' not in high_dim_paths:
             logger.error("缺少用于 PCA 的高维训练数据路径。")
             results['success'] = False
             return results
        model_path = os.path.join(dr_model_dir, f"pca_model_{target_field_dr}.pkl")
        dr_scaler_path = os.path.join(config.paths.processed_data_dir, f"pca_{target_field_dr}_input_scaler.pkl")
        dr_results = train_and_transform_pca(
            cfg, high_dim_paths, model_path, dr_scaler_path, transformed_data_dir
        )
        if dr_results:
            results.update(dr_results)
            # Check if train_and_transform_pca sets 'low_dim_data_paths' and 'success'
            if 'low_dim_data_paths' in dr_results and dr_results.get('success', False):
                results['success'] = True
            else:
                results['success'] = False
        else:
            results['success'] = False
    else:
        logger.error(f"未知的降维方法: {method}")
        results['success'] = False


    # --- Final Check and Logging ---
    if results.get('success', False):
         if 'low_dim_data_paths' in results and results['low_dim_data_paths']:
              logger.info(f"成功生成低维数据路径: {results['low_dim_data_paths']}")
         else:
              logger.error("降维标记为成功，但 low_dim_data_paths 为空或缺失！")
              results['success'] = False # Correct the status
    else:
         logger.error("降维步骤未能生成低维数据路径或遇到致命错误。")


    logger.info(f"--- 降维完成 (方法: {method}) ---")
    return results

def run_prediction(cfg: DictConfig, dr_results: Dict[str, Any]) -> Dict[str, Any]:
    """运行所选的预测方法。"""
    config = DrprConfig.from_hydra_config(cfg)
    method = cfg.model.prediction.method
    logger.info(f"--- 阶段：预测 (方法: {method}) ---")
    results = {'method': method}
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
        # 如果 DR 是 SOM，HMM 需要状态和观测 SOM 模型
        # 如果 DR 是 PCA/AE，则需要修改 HMM
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

        elif dr_method in ['pca', 'autoencoder']:
            # HMM 处理连续输入的挑战点
            if hmm_input_type == 'continuous':
                logger.error(f"使用 GaussianHMM 处理来自 {dr_method} 的连续低维输入的 HMM 逻辑尚未实现。需要修改 train_hmm.py。")
                results['success'] = False; return results
                # 如果已实现：
                # 1. 修改 train_hmm.py 以使用 GaussianHMM 并在 continuous low_dim_data_paths 上拟合
                # 2. 调用修改后的 HMM 训练函数
                # 3. HMM 将预测低维向量序列 (隐藏状态的均值)

            elif hmm_input_type == 'discrete':
                 logger.error(f"将来自 {dr_method} 的连续低维输入离散化以用于 CategoricalHMM 的逻辑尚未实现。需要在 Pipeline 中添加 K-Means 或类似步骤。")
                 results['success'] = False; return results
                 # 如果已实现：
                 # 1. 在此步骤之前添加一个离散化步骤（例如 K-Means）
                 # 2. 将离散化后的状态序列传递给现有的 train_hmm_main
            else:
                 logger.error(f"未知的 HMM 输入类型配置: {hmm_input_type}")
                 results['success'] = False; return results

        else:
             logger.error(f"不支持的降维方法 ({dr_method}) 与 HMM 组合。")
             results['success'] = False; return results

    elif method == 'lstm':
        dr_method = dr_results['method']
        lstm_input_type = cfg.model.prediction.lstm.get('input_type')
        model_save_dir_lstm = os.path.join(pred_model_dir) # 在此保存 LSTM 模型
        scaler_save_path_lstm = os.path.join(config.paths.processed_data_dir, f"{dr_results['method']}_lstm_input_scaler.pkl")
        pred_output_dir_lstm = os.path.join(pred_output_dir) # 在此保存预测

        if lstm_input_type == 'bmu_rank':
            pred_results = train_and_predict_lstm(
                cfg,
                low_dim_data_paths=low_dim_data_paths, # 来自降维步骤
                model_save_dir=model_save_dir_lstm,
                scaler_save_path=scaler_save_path_lstm,
                prediction_save_dir=pred_output_dir_lstm
                # prediction_filename_pattern 可以使用默认值
            )
            if pred_results: # 合并结果
                results.update(pred_results)

                
        elif lstm_input_type == 'dv':
            logger.error(f"使用 LSTM 处理来自 {dr_method} 的离散低维输入的逻辑尚未实现。需要修改相关代码。")
            results['success'] = False; return results
            

    else:
        logger.error(f"未知的预测方法: {method}")
        results['success'] = False; return results

    if 'predicted_low_dim_paths' not in results or not results['predicted_low_dim_paths']:
         logger.error("预测步骤未能生成预测的低维数据路径。")
         results['success'] = False
    else:
         results['success'] = True

    logger.info(f"--- 预测完成 (方法: {method}) ---")
    return results

def run_reconstruction(cfg: DictConfig, dr_results: Dict[str, Any], pred_results: Dict[str, Any]) -> Dict[str, Any]:
    """根据降维方法和预测的低维数据运行重建。"""
    config = DrprConfig.from_hydra_config(cfg)
    dr_method = dr_results['method']
    pred_method = pred_results['method']
    logger.info(f"--- 阶段：重建 (DR: {dr_method}, Pred: {pred_method}) ---")
    results = {}

    predicted_low_dim_paths = pred_results.get('predicted_low_dim_paths')
    dr_model_path = dr_results.get('model_path')
    # 原始高维数据的 Scaler 路径 (来自 PCA/AE)
    high_dim_scaler_path = dr_results.get('scaler_path')

    if not predicted_low_dim_paths:
        logger.error("缺少预测的低维数据路径，无法重建。")
        results['success'] = False; return results

    recon_output_dir = os.path.join(config.paths.reconstructions_base_dir, f"{dr_method}_{pred_method}")
    os.makedirs(recon_output_dir, exist_ok=True)
    results['reconstructed_high_dim_paths'] = {}

    if dr_method == 'som':
        # 重建需要预测的状态 BMU 秩/索引和状态 SOM 模型
        state_som_model_path = dr_results.get('som_state', {}).get('model_path')
        hmm_params_path = pred_results.get('model_path') # HMM 参数包含秩映射

        if not state_som_model_path:
             logger.error("缺少状态 SOM 模型路径，无法进行 BMU 重建。")
             results['success'] = False; return results
        if not hmm_params_path and cfg.model.prediction.hmm.input_type == 'bmu_rank':
             # 只有在输入是秩时才需要 HMM 参数进行映射
             logger.warning("缺少 HMM 参数路径，并且 HMM 输入类型是 'bmu_rank'。无法将预测的秩转换为索引进行重建。")
             results['success'] = False; return results
        elif not hmm_params_path:
             logger.warning("缺少 HMM 参数路径，将假设预测的 BMU 是线性索引。")


        for split, pred_path in predicted_low_dim_paths.items():
            logger.info(f"  重建 {split} 分割 (BMU)...")
            output_path = os.path.join(recon_output_dir, f"reconstructed_high_dim_{split}.npy")
            try:
                reconstructed_flat = reconstruct_from_bmu( 
                    cfg,
                    som_model_path=state_som_model_path,
                    predicted_bmu_path=pred_path, # 指向预测的状态秩/索引的路径
                    output_path=output_path,
                    # 只有当预测的是秩时才需要 HMM 参数
                    hmm_params_path=hmm_params_path if cfg.model.prediction.hmm.input_type == 'bmu_rank' else None
                )
                if reconstructed_flat is not None:
                    # *** 反标准化需要高维数据的 Scaler ***
                    salinity_scaler_path = os.path.join(config.paths.processed_data_dir, "salinity_scaler.npy")
                    salinity_scaler_params = None
                    if os.path.exists(salinity_scaler_path):
                        try:
                            # 使用现有函数加载 scaler (假设它处理按维度)
                            salinity_scaler_params = load_scaler(cfg, "salinity")
                        except Exception as e:
                            logger.warning(f"加载 Salinity Scaler ({salinity_scaler_path}) 失败: {e}。跳过反标准化。")

                    if salinity_scaler_params:
                        logger.info(f"  对 {split} 重建结果执行反标准化...")
                        # 假设 scaler_params['mean'] 和 ['std'] 是向量
                        mean_vec = salinity_scaler_params['mean']
                        std_vec = salinity_scaler_params['std']
                        epsilon = 1e-8
                        # 检查维度是否匹配
                        if isinstance(mean_vec, np.ndarray) and mean_vec.ndim == 1 and len(mean_vec) == reconstructed_flat.shape[1]:
                            reconstructed_inv_flat = reconstructed_flat * (std_vec + epsilon) + mean_vec
                        elif np.isscalar(mean_vec): # 处理全局 scaler 的情况
                            reconstructed_inv_flat = reconstructed_flat * (std_vec + epsilon) + mean_vec
                        else:
                             logger.error(f" Salinity Scaler 维度 ({mean_vec.shape if isinstance(mean_vec, np.ndarray) else type(mean_vec)}) 与重建特征 ({reconstructed_flat.shape[1]}) 不匹配。")
                             reconstructed_inv_flat = reconstructed_flat # 回退
                    else:
                        reconstructed_inv_flat = reconstructed_flat # 没有 scaler

                    # 保存最终的（可能已反标准化的）扁平数据
                    np.save(output_path, reconstructed_inv_flat) # 如果适用，用反标准化的结果覆盖
                    results['reconstructed_high_dim_paths'][split] = output_path
                    logger.info(f"  {split} 重建的高维数据 (flat) 已保存: {output_path}")
                else:
                     logger.error(f"  {split} BMU 重建失败。")

            except Exception as e:
                logger.error(f"  重建 {split} (BMU) 失败: {e}", exc_info=True)

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
                 logger.warning(f"{split}: 重建样本数 ({n_recon_samples}) 小于原始样本数 ({n_orig_samples})。将使用前 {n_recon_samples} 个原始样本进行比较。")
                 orig_split_raw = orig_split_raw[:n_recon_samples]


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
            if cfg.evaluation.get("calculate_correlation", True): # 添加配置开关
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
            if cfg.evaluation.visualization.get("plot_instantaneous", True): # 添加配置开关
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