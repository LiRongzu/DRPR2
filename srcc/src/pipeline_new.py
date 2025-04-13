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
from src.evaluation.visualization import plot_time_series_comparison, plot_spatial_rmse
from src.utils.model_utils import get_device_from_config  
from src.dimensionality_reduction.som_pytorch import SOMTorch
# 导入集中化的数据加载函数
from src.utils.data_loader import load_raw_data, load_mask, load_scaler, load_split_indices

# --- 导入训练函数 ---
from src.training.train_som import train_single_feature_som, train_combined_feature_som
from src.training.train_hmm import main as train_hmm_main
from src.training.train_lstm import train_and_predict_lstm
from src.training.train_pca import train_and_transform_pca
from src.training.train_autoencoder import train_and_transform_ae
from src.reconstruction.reconstruct_bmu import reconstruct_from_bmu # 用于基于 SOM 的重建
# 需要一个用于 PCA/AE 基于逆变换的重建函数
logger = logging.getLogger(__name__)

# --- 设置全局随机种子函数 ---
def set_global_seeds(seed):
    """
    设置所有随机数生成器的种子，确保结果可重现
    
    Args:
        seed: 随机种子值
    """
    if seed is None:
        logger.warning("未提供随机种子，结果可能不可重现")
        return
        
    logger.info(f"设置全局随机种子: {seed}")
    
    # Python 内置 random 模块
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    
    # 如果有CUDA可用，也设置CUDA种子
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
    results = {'method': method}
    dr_results = None

    # --- 定义降维所需的路径 ---
    # 指向处理后的高维数据路径 (PCA/AE 的输入)
    # 假设盐度是降维目标 (如果可变，需要修改)
    target_field_dr = "salinity"
    high_dim_paths = {}
    for split in ["train", "val", "test"]:
         # 构建路径，例如使用 cfg.paths.processed_data_dir 和命名约定
         # 假设存在像 train_salinity_processed.npy 这样的预处理数据
         try:
             path = os.path.join(config.paths.processed_data_dir, f"{split}_{target_field_dr}_processed.npy")
             if os.path.exists(path):
                 high_dim_paths[split] = path
         except Exception as e:
             logger.warning(f"无法构造 {split} 的高维数据路径: {e}")

    if 'train' not in high_dim_paths:
        logger.error("缺少用于降维的高维训练数据路径。")
        return results # 返回部分结果表示失败

    # --- 降维模型输出的通用路径 ---
    dr_model_dir = os.path.join(config.paths.models_base_dir, method) # 例如 models/som, models/pca
    os.makedirs(dr_model_dir, exist_ok=True)
    dr_scaler_path = os.path.join(config.paths.processed_data_dir, f"{method}_{target_field_dr}_input_scaler.pkl") # 高维输入的 Scaler
    transformed_data_dir = os.path.join(config.paths.processed_data_dir, f"{method}_{target_field_dr}_low_dim") # 在此存储低维数据
    os.makedirs(transformed_data_dir, exist_ok=True)

    if method == 'som':
        # 如果使用 HMM，SOM 需要特殊处理状态与观测
        # 状态 SOM (Salinity)
        som_state_results = train_single_feature_som(cfg, feature_name=target_field_dr)
        if not som_state_results: raise RuntimeError("状态 SOM 训练失败")
        results['som_state'] = som_state_results

        # 生成距离向量 (如果 LSTM 使用) 或 BMU 索引/秩
        # 此逻辑可能需要根据 LSTM/HMM 的消耗进行细化
        # 目前假设 BMU 索引是 HMM 的主要输出
        # 如果 LSTM 需要，可能需要单独生成距离向量
        # HMM 输入需要 BMU 文件路径
        # bmu_paths 字典在 som_state_results['bmu_paths'] 中
        bmu_paths_state = som_state_results.get('bmu_paths', {})
        results['low_dim_data_paths'] = {split: p.get('positions') for split, p in bmu_paths_state.items() if p.get('positions')}
        results['model_path'] = som_state_results.get('model_path')
        # 在此设置中，SOM 不使用单独的高维输入缩放器

        # 处理 HMM 的观测 SOM
        if cfg.model.prediction.method == 'hmm':
            observation_features = list(cfg.model.prediction.hmm.observation_features)
            obs_feature_name = "_".join(sorted(observation_features))
            if len(observation_features) == 2:
                som_obs_results = train_combined_feature_som(cfg, output_feature_name=obs_feature_name)
            elif len(observation_features) == 1:
                cfg.training.som.map_size = cfg.training.som.map_size_obs
                som_obs_results = train_single_feature_som(cfg, feature_name=observation_features[0])
            else:
                raise ValueError("无效的 HMM 观测特征配置")
            if not som_obs_results: raise RuntimeError("观测 SOM 训练失败")
            results['som_observation'] = som_obs_results

    elif method == 'pca':
        model_path = os.path.join(dr_model_dir, f"pca_model_{target_field_dr}.pkl")
        dr_results = train_and_transform_pca(
            cfg, high_dim_paths, model_path, dr_scaler_path, transformed_data_dir
        )
    elif method == 'autoencoder':
        model_path = os.path.join(dr_model_dir, f"ae_model_{target_field_dr}.pt")
        dr_results = train_and_transform_ae(
            cfg, high_dim_paths, model_path, dr_scaler_path, transformed_data_dir
        )
    else:
        logger.error(f"未知的降维方法: {method}")
        return results

    if dr_results: # 合并 PCA/AE 调用的结果
        results.update(dr_results)

    if 'low_dim_data_paths' not in results or not results['low_dim_data_paths']:
         logger.error("降维步骤未能生成低维数据路径。")
         results['success'] = False
    else:
         results['success'] = True

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
                # pred_results = run_gaussian_hmm_training(...)
                # results['predicted_low_dim_paths'] = pred_results['predicted_vector_paths']
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
        model_save_dir_lstm = os.path.join(pred_model_dir) # 在此保存 LSTM 模型
        scaler_save_path_lstm = os.path.join(config.paths.processed_data_dir, f"{dr_results['method']}_lstm_input_scaler.pkl")
        pred_output_dir_lstm = os.path.join(pred_output_dir) # 在此保存预测

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
    mask = load_mask(cfg) # 假设盐度 mask 具有代表性
    if mask is None: logger.warning("无法加载 Mask，空间评估可能不准确。"); flat_mask=None; target_spatial_shape=None
    else: flat_mask = mask.flatten(); target_spatial_shape = mask.shape

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
            if orig_split_raw.ndim > 2 and flat_mask is not None:
                 T = orig_split_raw.shape[0]
                 # 从原始数据中提取有效点
                 orig_valid_flat = orig_split_raw.reshape(T, -1)[:, flat_mask]
                 # 检查特征数是否匹配
                 if orig_valid_flat.shape[1] != recon_flat.shape[1]:
                     logger.error(f"{split}: 原始数据有效点数 ({orig_valid_flat.shape[1]}) 与重建数据特征数 ({recon_flat.shape[1]}) 不匹配！")
                     continue
                 orig_for_comparison = orig_valid_flat
            elif orig_split_raw.ndim == 2:
                 # 假设原始数据已经是扁平的并且匹配
                 if orig_split_raw.shape[1] != recon_flat.shape[1]:
                     logger.error(f"{split}: 原始数据特征数 ({orig_split_raw.shape[1]}) 与重建数据特征数 ({recon_flat.shape[1]}) 不匹配！")
                     continue
                 orig_for_comparison = orig_split_raw
            else:
                 logger.error(f"{split}: 原始数据维度 ({orig_split_raw.ndim}) 无法处理。")
                 continue

            # --- 计算指标 ---
            # ... (与之前版本类似的指标计算逻辑) ...
            # 需要注意：现在比较的是 recon_flat 和 orig_for_comparison (两者都应是反标准化/原始尺度)
            metrics = {}
            rmse_field = np.full(mask.shape, np.nan) if mask is not None else None

            if mask is not None and target_spatial_shape is not None:
                 # --- 重塑回空间网格进行比较 ---
                 recon_spatial = np.full((n_recon_samples,) + target_spatial_shape, np.nan)
                 orig_spatial_masked = np.full((n_recon_samples,) + target_spatial_shape, np.nan)
                 temp_flat = np.full(mask.size, np.nan)

                 for t in range(n_recon_samples):
                      temp_flat[flat_mask] = recon_flat[t]
                      recon_spatial[t] = temp_flat.reshape(target_spatial_shape)
                      temp_flat[flat_mask] = orig_for_comparison[t] # 用对齐后的扁平原始数据填充
                      orig_spatial_masked[t] = temp_flat.reshape(target_spatial_shape)


                 diff = recon_spatial - orig_spatial_masked # 与填充 NaN 的原始数据比较
                 valid_points_mask_3d = ~np.isnan(orig_spatial_masked) # 有效点是原始数据非 NaN 的地方

                 if np.any(valid_points_mask_3d):
                      # ... (计算 RMSE, MAE, rmse_field 等指标) ...
                      diff_sq_masked = np.square(diff[valid_points_mask_3d])
                      mean_rmse_val = np.sqrt(np.mean(diff_sq_masked))
                      mae_val = np.mean(np.abs(diff[valid_points_mask_3d]))

                      mean_diff_sq_spatial = np.nanmean(np.square(diff), axis=0)
                      valid_spatial_points = mask & ~np.isnan(mean_diff_sq_spatial)
                      if np.any(valid_spatial_points):
                           rmse_field[valid_spatial_points] = np.sqrt(mean_diff_sq_spatial[valid_spatial_points])

                      metrics = {
                           "mean_rmse": float(mean_rmse_val), "mean_mae": float(mae_val),
                           "rmse_map": rmse_field,
                           "max_rmse": float(np.nanmax(rmse_field)) if np.any(np.isfinite(rmse_field)) else np.nan,
                           "min_rmse": float(np.nanmin(rmse_field)) if np.any(np.isfinite(rmse_field)) else np.nan
                      }
                 else:
                      # ... (处理没有有效点的情况) ...
                      metrics = {k: np.nan for k in ["mean_rmse", "mean_mae", "max_rmse", "min_rmse"]}
            else: # 无法进行空间指标计算
                 # ... (计算扁平化数据的 RMSE/MAE) ...
                 diff_flat = recon_flat - orig_for_comparison
                 valid_flat = ~np.isnan(orig_for_comparison)
                 if np.any(valid_flat):
                      metrics = {"mean_rmse": np.sqrt(np.mean(np.square(diff_flat[valid_flat]))),
                                 "mean_mae": np.mean(np.abs(diff_flat[valid_flat]))}
                 else: metrics = {"mean_rmse": np.nan, "mean_mae": np.nan}

            # --- 保存指标和生成图表 ---
            eval_output_dir = os.path.join(config.paths.evaluation_base_dir, f"{dr_method}_{pred_method}", split)
            os.makedirs(eval_output_dir, exist_ok=True)
            metrics_path = os.path.join(eval_output_dir, f"metrics_{split}.npy")
            np.save(metrics_path, metrics)
            logger.info(f"  评估指标 ({split}) 已保存到: {metrics_path}")

            # ... (生成图表的代码，使用新的标题包含方法组合) ...
            if metrics.get("mean_rmse") is not np.nan:
                 rec_mean_ts = np.nanmean(recon_flat, axis=1)
                 orig_mean_ts = np.nanmean(orig_for_comparison, axis=1)
                 ts_save_path = os.path.join(eval_output_dir, f"time_series_comparison_{split}.png")
                 plot_time_series_comparison(rec_mean_ts, orig_mean_ts, cfg=cfg, save_path=ts_save_path,
                                               title=f"Spatial Mean TS ({split} - {dr_method}_{pred_method})")

                 if rmse_field is not None and np.any(np.isfinite(rmse_field)):
                     spatial_rmse_save_path = os.path.join(eval_output_dir, f"spatial_rmse_{split}.png")
                     plot_spatial_rmse(rmse_field, cfg=cfg, mask=mask, save_path=spatial_rmse_save_path,
                                       title=f"Spatial RMSE ({split} - {dr_method}_{pred_method})")
                 logger.info(f"  评估图表 ({split}) 已保存到: {eval_output_dir}")

            # 存储此分割的结果
            all_eval_results[split] = {'metrics': metrics}
            logger.info(f"--- 评估 {split} 完成 ---")

        except Exception as e:
            logger.error(f"评估 {split} 时出错: {e}", exc_info=True)

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
    # 设置 logger 可能?

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