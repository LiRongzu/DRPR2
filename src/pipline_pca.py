# src/run_pipeline.py
import os
import sys
import logging
import time
import hydra
import random
from omegaconf import DictConfig, OmegaConf, ListConfig
from typing import Optional, Dict, Any, List
import numpy as np
import joblib  # Make sure joblib is imported
import torch
import shutil

# --- 项目设置 和 import ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils.hydra_config import DrprConfig
from src.evaluation.visualization import (
    plot_time_series_comparison,
    plot_spatial_rmse,
    plot_spatial_comparison_at_timestep,
    plot_spatial_difference,
    plot_spatial_statistic,
    calculate_correlation_map,
    generate_evaluation_report,
    plot_spatial_distribution
)
from src.utils.model_utils import get_device_from_config
from src.dimensionality_reduction.som_pytorch import SOMTorch
# --- 导入数据加载器 ---
from src.utils.data_loader import load_raw_data, load_mask, load_scaler, load_split_indices # Keep load_scaler if used elsewhere

# --- 导入训练和重建函数 ---
from src.training.train_som import train_single_feature_som, train_combined_feature_som
from src.training.train_hmm import main as train_hmm_main
from src.training.train_lstm import train_and_predict_lstm
# --- >>> 导入 PCA 相关的 LSTM 训练函数 <<< ---
from src.training.train_lstm_pca import train_and_predict_lstm_pca

from src.reconstruction.reconstruct_bmu import reconstruct_from_bmu # 用于基于 SOM 的重建

logger = logging.getLogger(__name__)

# --- Utility function needed in run_reconstruction ---
def load_scaler_info(scaler_path: str) -> (Optional[Any], Optional[str]):
    """从 npy 或 pkl 文件加载 scaler 信息。"""
    if not os.path.exists(scaler_path):
        logger.error(f"Scaler 文件未找到: {scaler_path}")
        return None, None
    try:
        suffix = os.path.splitext(scaler_path)[1].lower()
        if suffix == '.pkl':
            scaler_obj = joblib.load(scaler_path)
            logger.debug(f"从 {scaler_path} 加载 scaler 对象 (pkl)")
            # 检查是否是有效的 scaler (可选)
            if not hasattr(scaler_obj, 'inverse_transform'):
                logger.error(f"加载的 pkl 对象 ({scaler_path}) 没有 inverse_transform 方法。")
                return None, None
            return scaler_obj, 'pkl'
        elif suffix == '.npy':
            scaler_dict = np.load(scaler_path, allow_pickle=True).item()
            logger.debug(f"从 {scaler_path} 加载 scaler 字典 (npy)")
            # --- >>> 修改这里的键检查：将 'scale' 改为 'std' (或你找到的实际键名) <<< ---
            if not isinstance(scaler_dict, dict) or ('mean' not in scaler_dict or 'std' not in scaler_dict):
                logger.error(f"npy scaler 文件 {scaler_path} 未包含有效的带有 'mean' 和 'std' 的字典。") # 更新错误消息
                return None, None
            # --- >>> 结束修改 <<< ---
            return scaler_dict, 'npy'
        else:
            logger.error(f"不支持的 scaler 文件类型: {suffix}")
            return None, None
    except Exception as e:
        logger.error(f"从 {scaler_path} 加载 scaler 失败: {e}", exc_info=True)
        return None, None


def set_global_seeds(seed = 42):
    """设置全局随机种子以确保结果可重现。"""
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
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        logger.info("已设置CUDA随机种子")
    logger.info("全局随机种子设置完成")


def run_dimensionality_reduction(cfg: DictConfig) -> Dict[str, Any]:
    """运行所选的降维方法 (或加载 PCA/AE 结果)。"""
    config = DrprConfig.from_hydra_config(cfg)
    method = cfg.model.dimensionality_reduction.method
    logger.info(f"--- 阶段：维度降低 (方法: {method}) ---")
    results = {'method': method, 'success': False} # 初始化 success 为 False

    # --- 如果是 PCA，加载预处理结果 ---
    if method == 'pca':
        logger.info("方法: PCA - 加载预处理结果")
        # 获取配置中定义了 PCA 的特征列表
        pca_features = cfg.model.dimensionality_reduction.pca.get("target_features", [])
        if not isinstance(pca_features, (list, ListConfig)) or not pca_features:
            logger.error("配置 'model.dimensionality_reduction.pca.target_features' 未定义、为空或不是列表。")
            return results

        # --- 从配置获取 PCA 输出和原始 Scaler 路径 ---
        try:
            pca_model_dir = config.paths.pca_models_dir # 使用 DrprConfig 解析后的路径
            pca_components_dir = config.paths.pca_components_dir
            # 原始高维 scaler 所在的目录 (用于重建)
            scaler_dir = config.paths.processed_data_dir # 假设原始 scaler 在 params_base_dir
        except AttributeError as e:
             logger.error(f"无法从配置的 paths 中获取 PCA 或 Scaler 目录: {e}。请检查 conf/paths/default.yaml 配置。")
             return results
        except Exception as e:
             logger.error(f"从配置中获取 PCA 或 Scaler 目录路径时出错: {e}")
             return results

        low_dim_data_paths = {} # {feature: {split: path}}
        model_paths = {}        # {feature: path}
        scaler_paths = {}       # {feature: path} to ORIGINAL high-dim scalers
        n_components_map = {}   # {feature: n_components}
        all_features_processed = True

        for feature in pca_features:
            logger.info(f"  检查特征 '{feature}' 的 PCA 文件...")
            feature_paths_found = True

            # 1. 定位 PCA 模型
            pca_model_path = os.path.join(pca_model_dir, f"pca_model_{feature}.pkl")
            if not os.path.exists(pca_model_path):
                logger.error(f"  未找到预计算的 PCA 模型: {pca_model_path}")
                feature_paths_found = False
            else:
                model_paths[feature] = pca_model_path
                # 从模型加载 n_components (更准确)
                try:
                    pca_model_loaded = joblib.load(pca_model_path)
                    n_components_map[feature] = pca_model_loaded.n_components_
                    logger.info(f"  从加载的 PCA 模型 '{feature}' 获取 n_components: {n_components_map[feature]}")
                except Exception as e:
                    logger.warning(f"  无法从 PCA 模型 '{feature}' 加载 n_components ({e})，将使用配置值。")
                    # 从配置获取 n_components 作为备选
                    comp_cfg = cfg.model.dimensionality_reduction.pca.get("components", {})
                    n_comp = comp_cfg.get(feature, cfg.model.dimensionality_reduction.pca.get("n_components"))
                    if n_comp is None:
                         logger.error(f"  配置中也找不到特征 '{feature}' 的 n_components。")
                         feature_paths_found = False
                    else:
                         n_components_map[feature] = n_comp


            # 2. 定位原始高维 Scaler (用于重建目标变量)
            # 尝试 .npy 和 .pkl
            scaler_path_found = False
            for ext in ['.npy', '.pkl']:
                 scaler_path_try = os.path.join(scaler_dir, f"{feature}_scaler{ext}")
                 if os.path.exists(scaler_path_try):
                      scaler_paths[feature] = scaler_path_try
                      logger.info(f"  找到原始高维 Scaler ({ext}): {scaler_paths[feature]}")
                      scaler_path_found = True
                      break # 找到一个即可
            if not scaler_path_found:
                 logger.error(f"  未找到特征 '{feature}' 的原始高维 Scaler 文件 (尝试了 .npy 和 .pkl): {scaler_dir}")
                 feature_paths_found = False # Scaler 对重建至关重要

            # 3. 定位 PCA 成分文件
            feature_component_paths = {}
            splits_found_for_feature = True
            for split in ["train", "val", "test"]:
                component_path = os.path.join(pca_components_dir, f"pca_components_{feature}_{split}.npy")
                if os.path.exists(component_path):
                    feature_component_paths[split] = component_path
                else:
                    logger.error(f"  未找到预计算的 PCA 成分文件 ({feature} - {split}): {component_path}")
                    splits_found_for_feature = False
            if not splits_found_for_feature:
                feature_paths_found = False # 必须所有分割都存在

            if feature_paths_found:
                low_dim_data_paths[feature] = feature_component_paths
            else:
                all_features_processed = False # 标记整体失败

        # --- 组装结果 ---
        if all_features_processed and low_dim_data_paths:
            results['low_dim_data_paths'] = low_dim_data_paths # {feature: {split: path}}
            results['model_path'] = model_paths              # {feature: path}
            results['scaler_path'] = scaler_paths            # {feature: path} to original scaler
            results['n_components'] = n_components_map         # {feature: n_components}
            # 确定主目标特征，例如 LSTM 的目标或列表中的第一个
            lstm_target = cfg.model.prediction.lstm_pca.get('target_feature') if cfg.model.prediction.method == 'lstm' else None
            results['target_feature'] = lstm_target or list(pca_features)[0] # 提供一个默认值
            results['success'] = True
            logger.info(f"成功定位所有特征 {pca_features} 的预计算 PCA 文件。")
        else:
            logger.error("未能成功定位所有必需特征的 PCA 文件。")
            results['success'] = False # 确保是 False

    # --- Autoencoder 逻辑 (Placeholder) ---
    elif method == 'autoencoder':
        # 此处应加载 AE 预处理结果，类似于 PCA
        logger.warning("Autoencoder 降维加载逻辑尚未实现。")
        # results['success'] 保持 False
        pass

    else:
        logger.error(f"未知的降维方法: {method}")
        # results['success'] is already False

    # --- Final Check and Logging ---
    if not results.get('success'):
        logger.error("降维步骤未能成功完成。")
    elif 'low_dim_data_paths' not in results or not results['low_dim_data_paths']:
        logger.error(f"降维步骤 ({method}) 标记为成功，但 low_dim_data_paths 丢失或为空！")
        results['success'] = False

    logger.info(f"--- 降维完成 (方法: {method}, 成功: {results.get('success')}) ---")
    return results


def run_prediction(cfg: DictConfig, dr_results: Dict[str, Any]) -> Dict[str, Any]:
    """运行所选的预测方法。"""
    config = DrprConfig.from_hydra_config(cfg)
    method = cfg.model.prediction.method
    dr_method = dr_results.get('method') # 获取降维方法

    logger.info(f"--- 阶段：预测 (方法: {method}, DR: {dr_method}) ---")
    results = {'method': method, 'success': False} # 初始化 success 为 False

    # 检查降维是否成功
    if not dr_results.get('success'):
        logger.error("由于降维步骤失败，无法继续进行预测。")
        return results

    # --- 检查低维数据路径 ---
    low_dim_data_paths = dr_results.get('low_dim_data_paths')
    if not low_dim_data_paths or not isinstance(low_dim_data_paths, dict):
        logger.error("缺少来自降维步骤的低维数据路径或格式不正确，无法进行预测。")
        return results

    # LSTM 预测逻辑
    if method == 'lstm':
        logger.info(f"方法: LSTM - 使用降维方法: {dr_method}")

        # --- LSTM with SOM BMUs ---
        if dr_method == 'som':
            logger.info("调用 LSTM (BMU 版本)...")
            # --- 获取 LSTM (BMU) 特定配置 ---
            lstm_cfg = cfg.model.prediction.get('lstm', {}) # 通用 LSTM 配置
            # 假设 train_and_predict_lstm 处理 BMU 输入
            # 确认 low_dim_data_paths 结构是 {feature: {split: {'positions': path}}}
            if not all(isinstance(split_data, dict) and 'positions' in split_data
                       for feature_data in low_dim_data_paths.values()
                       for split_data in feature_data.values()):
                logger.error(f"传递给 LSTM (BMU) 的 low_dim_data_paths 结构不正确: {low_dim_data_paths}")
                return results

            # --- 定义 LSTM (BMU) 输出子目录和模式 ---
            model_subdir = "lstm_bmu_model"
            prediction_subdir = "lstm_bmu_predictions"
            target_field_name = dr_results.get('target_feature', 'salinity') # BMU 预测的目标通常是状态特征
            prediction_filename_pattern = f"predicted_lstm_target_bmu_{target_field_name}_{{split}}.npy"

            try:
                pred_results_lstm_bmu = train_and_predict_lstm( # 调用 BMU 版本
                    cfg=cfg,
                    low_dim_data_paths=low_dim_data_paths, # BMU 路径
                    # --- 传递子目录名称 ---
                    model_save_subdir=model_subdir,
                    prediction_save_subdir=prediction_subdir,
                    # --- 结束传递子目录名称 ---
                    prediction_filename_pattern=prediction_filename_pattern
                )
                # --- 处理 BMU 版本 LSTM 的返回结果 ---
                if pred_results_lstm_bmu and pred_results_lstm_bmu.get('success') and 'predicted_target_low_dim_paths' in pred_results_lstm_bmu:
                    # 这个路径指向预测的目标 BMU 序列
                    results['predicted_low_dim_paths'] = pred_results_lstm_bmu['predicted_target_low_dim_paths']
                    results['model_path'] = pred_results_lstm_bmu.get('model_path') # LSTM 模型路径
                    results['target_feature'] = target_field_name # 记录预测的目标是哪个特征的 BMU
                    results['success'] = True
                    logger.info(f"LSTM (BMU) 预测成功。")
                else:
                    logger.error("train_and_predict_lstm (BMU) 未返回预期结果或失败。")
                    results['success'] = False
            except Exception as e:
                logger.error(f"调用 train_and_predict_lstm (BMU) 时出错: {e}", exc_info=True)
                results['success'] = False


        # --- LSTM with PCA Components ---
        elif dr_method == 'pca':
            logger.info("调用 LSTM (PCA 版本)...")
            try:
                # --- 获取 LSTM (PCA) 特定配置 ---
                lstm_pca_cfg = cfg.model.prediction.lstm_pca
                # --- >>> 获取输入和目标特征列表 <<< ---
                input_features = list(lstm_pca_cfg.input_features) # e.g., ['wind'] or ['wind', 'salinity']
                target_feature = lstm_pca_cfg.target_feature     # e.g., 'salinity'

                # --- 从 dr_results 获取 PCA 信息 ---
                # low_dim_data_paths 已经是 {feature: {split: path}} 结构
                pca_component_paths = dr_results.get('low_dim_data_paths', {})
                pca_n_components = dr_results.get('n_components', {}) # 已经是 {feature: n_components} 结构

                # --- 验证所需信息是否存在 ---
                if target_feature not in pca_component_paths or target_feature not in pca_n_components:
                     raise ValueError(f"目标特征 '{target_feature}' 的 PCA 成分路径或 n_components 未在 dr_results 中找到。")
                for f in input_features:
                     if f not in pca_component_paths or f not in pca_n_components:
                         raise ValueError(f"输入特征 '{f}' 的 PCA 成分路径或 n_components 未在 dr_results 中找到。")

                # --- 定义 LSTM (PCA) 输出子目录和模式 ---
                model_subdir = "lstm_pca_model"
                prediction_subdir = "lstm_pca_predictions"
                prediction_filename_pattern = f"predicted_pca_{target_feature}_lstm_{{split}}.npy"

                # --- 调用 LSTM (PCA) 训练/预测函数 ---
                pred_results_lstm_pca = train_and_predict_lstm_pca(
                    cfg=cfg, # Pass the whole config
                    # --- >>> 传递输入和目标特征 <<< ---
                    input_features=input_features,
                    target_feature=target_feature,
                    # --- >>> 结束传递 <<< ---
                    low_dim_data_paths=pca_component_paths, # PCA 成分路径
                    pca_n_components=pca_n_components,      # 各特征的成分数
                    # --- >>> 传递子目录名称 <<< ---
                    model_save_subdir=model_subdir,
                    prediction_save_subdir=prediction_subdir,
                    # --- >>> 结束传递 <<< ---
                    prediction_filename_pattern=prediction_filename_pattern
                )

                # --- 处理 PCA 版本 LSTM 的返回结果 ---
                if pred_results_lstm_pca and pred_results_lstm_pca.get('success') and 'predicted_target_low_dim_paths' in pred_results_lstm_pca:
                    # 这个路径指向预测的目标 PCA 成分序列
                    results['predicted_low_dim_paths'] = pred_results_lstm_pca['predicted_target_low_dim_paths']
                    results['model_path'] = pred_results_lstm_pca.get('model_path') # LSTM 模型路径
                    results['target_feature'] = target_feature # 记录预测的目标是哪个特征的 PCA 成分
                    results['success'] = True
                    logger.info(f"LSTM (PCA) 预测成功。")
                else:
                    logger.error("train_and_predict_lstm_pca 未返回预期结果或失败。")
                    results['success'] = False

            except KeyError as e:
                 logger.error(f"配置 'model.prediction.lstm_pca' 中缺少关键参数: {e}", exc_info=True)
                 results['success'] = False
            except ValueError as e:
                 logger.error(f"调用 LSTM (PCA) 时发生值错误: {e}", exc_info=True)
                 results['success'] = False
            except Exception as e:
                logger.error(f"运行 LSTM (PCA) 预测时发生意外错误: {e}", exc_info=True)
                results['success'] = False

        # --- [Placeholder for LSTM with Autoencoder latent space] ---
        elif dr_method == 'autoencoder':
             logger.error("尚未实现 LSTM 与 Autoencoder 结合的预测。")
             results['success'] = False

        else:
             logger.error(f"不支持的降维方法 '{dr_method}' 与 LSTM 结合使用。")
             results['success'] = False

    else:
        logger.error(f"未知的预测方法: {method}")
        results['success'] = False

    # --- 最终检查和日志 ---
    if not results.get('success'):
        logger.error(f"预测阶段 ({method}) 失败。")
    elif 'predicted_low_dim_paths' not in results or not results['predicted_low_dim_paths']:
        logger.error(f"预测阶段 ({method}) 标记为成功，但缺少 'predicted_low_dim_paths'！")
        results['success'] = False # 修正状态

    logger.info(f"--- 预测完成 (方法: {method}, DR: {dr_method}, 成功: {results.get('success')}) ---")
    return results


def run_reconstruction(cfg: DictConfig, dr_results: Dict[str, Any], pred_results: Dict[str, Any]) -> Dict[str, Any]:
    """根据降维方法和预测的低维数据运行重建。"""
    config = DrprConfig.from_hydra_config(cfg)
    dr_method = dr_results['method']
    pred_method = pred_results['method']
    logger.info(f"--- 阶段：重建 (DR: {dr_method}, Pred: {pred_method}) ---")
    results = {'success': False, 'reconstructed_high_dim_paths': {}} # 初始化

    # 检查预测是否成功并且有路径
    predicted_low_dim_paths = pred_results.get('predicted_low_dim_paths')
    if not pred_results.get('success') or not predicted_low_dim_paths:
        logger.error("预测步骤失败或缺少预测路径，无法进行重建。")
        return results

    # 定义重建输出目录 (现在基于 Hydra 输出目录构建)
    hydra_output_dir = os.getcwd() # 获取当前 Hydra 输出目录
    recon_output_subdir = f"reconstruction_{dr_method}_{pred_method}"
    recon_output_dir = os.path.join(hydra_output_dir, recon_output_subdir)
    os.makedirs(recon_output_dir, exist_ok=True)
    logger.info(f"重建结果将保存在: {recon_output_dir}")


    if dr_method == 'som':
        # --- SOM 重建逻辑 (保持你原来的逻辑) ---
        state_som_model_path = dr_results.get('som_state', {}).get('model_path')
        if not state_som_model_path: logger.error("缺少状态 SOM 模型路径..."); return results

        hmm_params_path_for_recon = None
        if pred_method == 'hmm': hmm_params_path_for_recon = pred_results.get('model_path')

        # 加载目标特征的原始 scaler
        target_feature_name = cfg.reconstruction.target_field # 例如 "salinity"
        target_scaler_path = os.path.join(config.paths.processed_data_dir, f"{target_feature_name}_scaler.npy")
        target_scaler_info, target_scaler_type = load_scaler_info(target_scaler_path) # 使用辅助函数

        recon_success_all_splits_som = True
        for split, pred_path in predicted_low_dim_paths.items(): # pred_path 指向预测的 BMU 序列
            logger.info(f"  重建 {split} 分割 (BMU)...")
            output_path = os.path.join(recon_output_dir, f"reconstructed_high_dim_{split}.npy")
            try:
                reconstructed_flat = reconstruct_from_bmu(
                    cfg=cfg,
                    som_model_path=state_som_model_path,
                    predicted_bmu_path=pred_path,
                    output_path=output_path, # reconstruct_from_bmu 可能不需要 output_path
                    hmm_params_path=hmm_params_path_for_recon
                )
                if reconstructed_flat is None: raise RuntimeError("reconstruct_from_bmu 返回 None")

                # --- 反标准化 ---
                reconstructed_inv_flat = reconstructed_flat # 默认值
                if target_scaler_info:
                    logger.info(f"  对 {split} 重建结果执行反标准化...")
                    epsilon = 1e-8
                    if target_scaler_type == 'npy':
                        mean_vec = target_scaler_info['mean']
                        scale_vec = target_scaler_info['scale'] # 使用 'scale'
                        if isinstance(mean_vec, np.ndarray) and mean_vec.ndim == 1 and len(mean_vec) == reconstructed_flat.shape[1]:
                            reconstructed_inv_flat = reconstructed_flat * scale_vec + mean_vec # 修正：先乘 scale，再加 mean
                        elif np.isscalar(mean_vec):
                            reconstructed_inv_flat = reconstructed_flat * scale_vec + mean_vec
                        else: logger.error(f"目标特征 Scaler (npy) 维度不匹配。跳过反标准化。")
                    elif target_scaler_type == 'pkl':
                         # 假设 scaler_info 是一个 fitted scaler object
                         try:
                              reconstructed_inv_flat = target_scaler_info.inverse_transform(reconstructed_flat)
                         except Exception as e_inv:
                              logger.error(f"使用 pkl scaler 进行 inverse_transform 失败: {e_inv}")
                    else: logger.warning(f"未知的 Scaler 类型 ({target_scaler_type})，跳过反标准化。")
                else: logger.warning(f"未加载目标特征 Scaler，跳过反标准化。")

                # 保存最终（可能反标准化的）结果
                np.save(output_path, reconstructed_inv_flat)
                results['reconstructed_high_dim_paths'][split] = output_path
                logger.info(f"  {split} 重建的高维数据 (flat) 已保存: {output_path}")
            except Exception as e:
                logger.error(f"  重建 {split} (BMU) 出错: {e}", exc_info=True)
                recon_success_all_splits_som = False

        if recon_success_all_splits_som and results['reconstructed_high_dim_paths']:
            results['success'] = True

    elif dr_method == 'pca': # 不需要 'autoencoder' 合并，逻辑相似但模型不同
        # --- PCA 重建逻辑 ---
        logger.info("方法: PCA 重建")
        # 获取 LSTM 预测的目标特征名称
        target_feature = pred_results.get('target_feature')
        if not target_feature:
            logger.error("无法从预测结果确定要重建的目标特征。")
            return results
        logger.info(f"将重建目标特征: {target_feature}")

        # 获取该目标特征的 PCA 模型路径和原始 Scaler 路径
        pca_model_path = dr_results.get('model_path', {}).get(target_feature)
        scaler_path = dr_results.get('scaler_path', {}).get(target_feature) # Path to ORIGINAL scaler

        if not pca_model_path or not os.path.exists(pca_model_path):
            logger.error(f"未找到目标特征 '{target_feature}' 的 PCA 模型路径 ({pca_model_path})。")
            return results
        if not scaler_path or not os.path.exists(scaler_path):
            logger.error(f"未找到目标特征 '{target_feature}' 的原始高维 Scaler 路径 ({scaler_path})。")
            return results

        try:
            # 加载目标特征的 PCA 模型
            pca_model = joblib.load(pca_model_path)
            logger.info(f"已加载目标特征 '{target_feature}' 的 PCA 模型。")

            # 加载目标特征的原始高维 Scaler
            scaler_info, scaler_type = load_scaler_info(scaler_path) # 使用辅助函数
            if scaler_info is None: raise ValueError(f"无法从 '{scaler_path}' 加载 Scaler 信息。")
            logger.info(f"已加载目标特征 '{target_feature}' 的原始 Scaler ({scaler_type} 类型)。")

            recon_success_all_splits_pca = True
            for split, pred_path in predicted_low_dim_paths.items(): # pred_path 指向预测的 PCA 成分
                logger.info(f"  重建 {split} 分割 (PCA)...")
                output_path = os.path.join(recon_output_dir, f"reconstructed_high_dim_{split}.npy") # 保存到 Hydra 子目录
                try:
                    predicted_low_dim = np.load(pred_path) # 加载 LSTM 的预测 (PCA 成分)

                    # 1. PCA 逆变换
                    reconstructed_scaled = pca_model.inverse_transform(predicted_low_dim)
                    logger.debug(f"  PCA 逆变换完成 ({split})，形状: {reconstructed_scaled.shape}")

                    # 2. Scaler 逆变换 (反标准化)
                    reconstructed_high_dim = None
                    epsilon = 1e-8
                    if scaler_type == 'pkl':
                        reconstructed_high_dim = scaler_info.inverse_transform(reconstructed_scaled)
                        logger.debug(f"  使用 pkl Scaler 对象进行反标准化 ({split})")
                    elif scaler_type == 'npy':
                        mean_vec = scaler_info['mean']
                        # --- >>> 修改这里的键名：将 'scale' 改为 'std' (或你找到的实际键名) <<< ---
                        std_vec = scaler_info['std'] # 获取 std 而不是 scale
                        epsilon = 1e-8
                        # 检查维度匹配
                        if isinstance(mean_vec, np.ndarray) and mean_vec.ndim == 1 and len(mean_vec) == reconstructed_scaled.shape[1]:
                            # --- >>> 修改这里的计算：使用 std_vec <<< ---
                            reconstructed_high_dim = reconstructed_scaled * (std_vec + epsilon) + mean_vec # 使用 std_vec
                        elif np.isscalar(mean_vec): # 处理全局 scaler
                            reconstructed_high_dim = reconstructed_scaled * (std_vec + epsilon) + mean_vec # 使用 std_vec
                        else:
                            logger.error(f"  npy Scaler 维度不匹配。跳过反标准化。")
                    else: logger.error(f"  未知的 Scaler 类型 '{scaler_type}'，无法反标准化。")

                    # 3. 保存结果
                    if reconstructed_high_dim is not None:
                        np.save(output_path, reconstructed_high_dim)
                        results['reconstructed_high_dim_paths'][split] = output_path
                        logger.info(f"  {split} 重建的高维数据 ({target_feature}) 已保存: {output_path}，形状: {reconstructed_high_dim.shape}")
                    else:
                         logger.error(f"  {split} 反标准化失败。")
                         recon_success_all_splits_pca = False

                except Exception as e:
                    logger.error(f"  重建 {split} (PCA) 失败: {e}", exc_info=True)
                    recon_success_all_splits_pca = False

            if recon_success_all_splits_pca and results['reconstructed_high_dim_paths']:
                 results['success'] = True

        except Exception as e:
            logger.error(f"加载 PCA 模型或 Scaler 或执行重建时出错: {e}", exc_info=True)
            results['success'] = False

    elif dr_method == 'autoencoder':
         logger.error("Autoencoder 重建尚未实现。")
         results['success'] = False

    else:
         logger.error(f"未知的降维方法 '{dr_method}'，无法重建。")
         results['success'] = False

    # 更新最终成功状态
    results['success'] = results.get('success', False) and bool(results.get('reconstructed_high_dim_paths'))
    logger.info(f"--- 重建完成 (DR: {dr_method}, Pred: {pred_method}, 成功: {results['success']}) ---")
    return results


def run_evaluation(cfg: DictConfig, recon_results: Dict[str, Any], dr_method: str, pred_method: str):
    """运行评估，比较重建数据与原始数据。"""
    config = DrprConfig.from_hydra_config(cfg)
    logger.info(f"--- 阶段：评估 (DR: {dr_method}, Pred: {pred_method}) ---")
    results = {'success': False} # 初始化

    reconstructed_paths = recon_results.get('reconstructed_high_dim_paths', {})
    if not recon_results.get('success') or not reconstructed_paths:
        logger.error("重建失败或缺少重建路径，无法进行评估。")
        return results

    # --- 确定评估的目标特征 ---
    target_feature_eval = None
    split_example = next(iter(reconstructed_paths))
    filename = os.path.basename(reconstructed_paths[split_example])
    # 尝试从文件名推断，例如 reconstructed_high_dim_salinity_test.npy 或 reconstructed_salinity_test.npy
    parts = filename.replace('reconstructed_', '').replace('_high_dim', '').split('_')
    if len(parts) >= 2: # 期望是 feature_split.npy
         potential_feature = parts[0]
         # 可以根据 cfg.reconstruction.target_field 或 cfg.model.prediction.*.target_feature 验证
         # 简单起见，我们直接使用它，或使用配置中的值
         target_feature_eval = cfg.reconstruction.get('target_field', potential_feature) # 优先用配置
         logger.info(f"将评估目标特征: {target_feature_eval}")
    else:
         logger.error(f"无法从重建文件名 '{filename}' 推断评估的目标特征。")
         return results


    # --- 加载 Mask ---
    mask = load_mask(cfg)
    if mask is None: logger.warning("无法加载 Mask，空间评估可能不准确。"); return results
    boolean_mask = (mask == 0) # True 表示有效点
    boolean_flat_mask = boolean_mask.flatten()
    target_spatial_shape = mask.shape
    logger.info(f"加载 Mask 并创建布尔掩码 (True=有效)，形状: {mask.shape}, 有效点数: {np.sum(boolean_mask)}")


    # --- 加载原始高维数据以供比较 (使用原始 raw 数据) ---
    original_high_dim = {}
    split_indices = load_split_indices(cfg)
    if split_indices is None: logger.error("无法加载 split_indices 用于评估。"); return results
    try:
        raw_target_full = load_raw_data(cfg, target_feature_eval) # 加载评估目标的原始数据
        if raw_target_full is None: raise RuntimeError(f"无法加载原始 {target_feature_eval} 数据进行评估。")

        for split, indices in split_indices.items():
            if indices is not None and len(indices) > 0:
                original_high_dim[split] = raw_target_full[indices]
                logger.info(f"加载原始高维数据 ({target_feature_eval} - {split}) 形状: {original_high_dim[split].shape}")

    except Exception as e:
        logger.error(f"加载原始高维数据进行评估时出错: {e}", exc_info=True)
        return results


    # --- 对每个分割执行评估 ---
    all_eval_results = {}
    # --- 获取 Hydra 输出目录用于保存评估结果 ---
    hydra_output_dir = os.getcwd()
    eval_base_dir = os.path.join(hydra_output_dir, f"evaluation_{dr_method}_{pred_method}")
    os.makedirs(eval_base_dir, exist_ok=True)
    logger.info(f"评估结果将保存在: {eval_base_dir}")

    for split, recon_path in reconstructed_paths.items():
        if split not in original_high_dim:
            logger.warning(f"跳过评估 {split}，因为缺少对应的原始数据。")
            continue

        logger.info(f"\n--- 评估分割: {split} ---")
        eval_output_dir_split = os.path.join(eval_base_dir, split) # 为每个 split 创建子目录
        os.makedirs(eval_output_dir_split, exist_ok=True)
        figure_paths_split = [] # 存储当前分割生成的图表路径

        try:
            recon_flat = np.load(recon_path) # 加载重建的扁平数据
            orig_split_raw = original_high_dim[split] # 获取对应的原始数据切片 (未缩放)

            # --- 对齐时间步长 ---
            n_recon_samples = recon_flat.shape[0]
            n_orig_samples = orig_split_raw.shape[0]
            if n_recon_samples == 0: continue

            if n_recon_samples > n_orig_samples:
                logger.warning(f"{split}: 重建样本数 ({n_recon_samples}) > 原始 ({n_orig_samples})。截断重建。")
                recon_flat = recon_flat[:n_orig_samples]
            elif n_recon_samples < n_orig_samples:
                logger.warning(f"{split}: 重建样本数 ({n_recon_samples}) < 原始 ({n_orig_samples})。使用后 {n_recon_samples} 个原始样本。")
                orig_split_raw = orig_split_raw[-n_recon_samples:]

            # --- 比较数据 (原始数据展平并应用 mask) ---
            if orig_split_raw.ndim != 3: # 期望 T, H, W
                 logger.error(f"{split}: 原始数据维度不是 3 ({orig_split_raw.shape})，无法处理。")
                 continue

            T = orig_split_raw.shape[0]
            orig_flat_full = orig_split_raw.reshape(T, -1)
            if orig_flat_full.shape[1] != len(boolean_flat_mask):
                 logger.error(f"{split}: 原始数据展平后特征数 ({orig_flat_full.shape[1]}) 与 Mask ({len(boolean_flat_mask)}) 不匹配！")
                 continue
            orig_valid_flat = orig_flat_full[:, boolean_flat_mask] # 提取有效点

            num_valid_points_expected = np.sum(boolean_mask)
            if recon_flat.shape[1] != num_valid_points_expected:
                 logger.error(f"{split}: 原始数据有效点数 ({num_valid_points_expected}) 与重建数据特征数 ({recon_flat.shape[1]}) 不匹配！")
                 continue
            orig_for_comparison = orig_valid_flat

            # --- 计算指标 ---
            logger.info(f"  计算 RMSE 和 MAE ({split})...")
            metrics = {}
            rmse_field = np.full(target_spatial_shape, np.nan)
            recon_spatial = np.full((n_recon_samples,) + target_spatial_shape, np.nan)
            orig_spatial_masked = np.full((n_recon_samples,) + target_spatial_shape, np.nan)
            diff_spatial = np.full((n_recon_samples,) + target_spatial_shape, np.nan)

            # 重塑回空间网格
            temp_flat_base = np.full(boolean_mask.size, np.nan)
            for t in range(n_recon_samples):
                temp_flat = temp_flat_base.copy()
                temp_flat[boolean_flat_mask] = recon_flat[t]
                recon_spatial[t] = temp_flat.reshape(target_spatial_shape)

                temp_flat = temp_flat_base.copy()
                temp_flat[boolean_flat_mask] = orig_for_comparison[t]
                orig_spatial_masked[t] = temp_flat.reshape(target_spatial_shape)

            # 计算差值 (只在有效点)
            diff_spatial[:, boolean_mask] = recon_spatial[:, boolean_mask] - orig_spatial_masked[:, boolean_mask]

            if np.any(boolean_mask):
                diff_sq_masked = np.square(diff_spatial[:, boolean_mask])
                mean_rmse_val = np.sqrt(np.mean(diff_sq_masked))
                mae_val = np.mean(np.abs(diff_spatial[:, boolean_mask]))

                mean_diff_sq_spatial = np.nanmean(np.square(diff_spatial[:, boolean_mask]), axis=0)
                rmse_field[boolean_mask] = np.sqrt(mean_diff_sq_spatial)

                metrics = {
                    "mean_rmse": float(mean_rmse_val), "mean_mae": float(mae_val),
                    "rmse_map": rmse_field, # 保存 numpy 数组
                    "max_rmse": float(np.nanmax(rmse_field)) if np.any(np.isfinite(rmse_field)) else np.nan,
                    "min_rmse": float(np.nanmin(rmse_field)) if np.any(np.isfinite(rmse_field)) else np.nan
                }
            else: # 没有有效点
                metrics = {k: np.nan for k in ["mean_rmse", "mean_mae", "max_rmse", "min_rmse"]}
                metrics["rmse_map"] = rmse_field

            # --- 保存指标和生成图表 ---
            # 保存指标 (移除 numpy 数组以保存为 json/yaml，或单独保存 npy)
            metrics_to_save = {k: v for k, v in metrics.items() if k != "rmse_map"}
            metrics_file = os.path.join(eval_output_dir_split, f"metrics_{split}.yaml") # 保存为 yaml
            try:
                 # 使用 OmegaConf 保存更安全
                 OmegaConf.save(config=OmegaConf.create(metrics_to_save), f=metrics_file)
                 logger.info(f"  评估指标 ({split}) 已保存到: {metrics_file}")
            except Exception as e_save:
                 logger.error(f"  保存指标文件失败 ({split}): {e_save}")
            # 单独保存 RMSE 图
            rmse_map_file = os.path.join(eval_output_dir_split, f"rmse_map_{split}.npy")
            np.save(rmse_map_file, metrics["rmse_map"])


            # --- 绘制图表 (如果指标有效) ---
            if metrics.get("mean_rmse") is not np.nan:
                # 时间序列对比图
                rec_mean_ts = np.nanmean(recon_flat, axis=1)
                orig_mean_ts = np.nanmean(orig_for_comparison, axis=1)
                ts_save_path = os.path.join(eval_output_dir_split, f"time_series_comparison_{split}.png")
                try:
                    plot_time_series_comparison(rec_mean_ts, orig_mean_ts, cfg=cfg, save_path=ts_save_path,
                                                title=f"Spatial Mean TS ({split} - {dr_method}_{pred_method})")
                    figure_paths_split.append(ts_save_path)
                except Exception as e_plot: logger.error(f"绘制时间序列图失败 ({split}): {e_plot}")

                # 空间 RMSE 图
                if rmse_field is not None and np.any(np.isfinite(rmse_field)):
                    spatial_rmse_save_path = os.path.join(eval_output_dir_split, f"spatial_rmse_{split}.png")
                    try:
                        # 注意 plot_spatial_rmse 可能需要 mask=True表示无效点
                        plot_spatial_rmse(rmse_field, cfg=cfg, mask=boolean_mask, save_path=spatial_rmse_save_path,
                                            title=f"Spatial RMSE ({split} - {dr_method}_{pred_method})")
                        figure_paths_split.append(spatial_rmse_save_path)
                    except Exception as e_plot: logger.error(f"绘制空间 RMSE 图失败 ({split}): {e_plot}")

                # --- (可选) 空间相关性图 ---
                if cfg.evaluation.get("calculate_correlation", False):
                    logger.info(f"  计算空间相关性图 ({split})...")
                    try:
                        # 注意 calculate_correlation_map 需要 mask=True 表示无效
                        corr_map = calculate_correlation_map(recon_spatial, orig_spatial_masked, (mask != 0))
                        metrics['correlation_map'] = corr_map # 添加到运行时字典，但不保存到 yaml
                        mean_corr = float(np.nanmean(corr_map)) if np.any(np.isfinite(corr_map)) else np.nan
                        metrics_to_save['mean_correlation'] = mean_corr # 保存平均相关性
                        logger.info(f"  {split} - Mean Correlation: {mean_corr:.6f}")
                        # 保存相关性图的 npy 文件
                        corr_map_file = os.path.join(eval_output_dir_split, f"correlation_map_{split}.npy")
                        np.save(corr_map_file, corr_map)

                        if cfg.evaluation.visualization.get("plot_spatial_correlation", True) and np.any(np.isfinite(corr_map)):
                            corr_save_path = os.path.join(eval_output_dir_split, f"spatial_correlation_{split}.png")
                            # plot_spatial_statistic 需要 mask=True 表示无效
                            plot_spatial_statistic(corr_map, cfg, (mask != 0), corr_save_path,
                                                 title=f"Spatial Correlation ({split} - {dr_method}_{pred_method})",
                                                 cmap='coolwarm', vmin=-1, vmax=1, cbar_label="Correlation Coeff.")
                            figure_paths_split.append(corr_save_path)
                    except Exception as e: logger.error(f"  计算或绘制空间相关性图失败 ({split}): {e}", exc_info=True)

                # --- (可选) 瞬时对比图 ---
                if cfg.evaluation.visualization.get("plot_instantaneous", False):
                    time_indices_to_plot = cfg.evaluation.visualization.get("time_indices_to_plot", [0, n_recon_samples // 2, n_recon_samples - 1])
                    logger.info(f"  绘制特定时间点的空间图 ({split}, indices={time_indices_to_plot})...")
                    for t_idx in time_indices_to_plot:
                         if 0 <= t_idx < n_recon_samples:
                              try:
                                   comp_save_path = os.path.join(eval_output_dir_split, f"spatial_comparison_{split}_t{t_idx}.png")
                                   # plot 函数需要 mask=True 表示无效
                                   plot_spatial_comparison_at_timestep(
                                       orig_spatial_masked[t_idx], recon_spatial[t_idx], diff_spatial[t_idx],
                                       cfg, (mask != 0), t_idx, comp_save_path,
                                       title_prefix=f"Spatial Comparison ({split} - {dr_method}_{pred_method})"
                                   )
                                   figure_paths_split.append(comp_save_path)

                                   if cfg.evaluation.visualization.get("plot_instantaneous_difference_only", True):
                                       diff_save_path = os.path.join(eval_output_dir_split, f"spatial_difference_{split}_t{t_idx}.png")
                                       plot_spatial_difference(
                                           diff_spatial[t_idx], cfg, (mask != 0), t_idx, diff_save_path,
                                           title=f"Spatial Difference ({split} - {dr_method}_{pred_method})"
                                       )
                                       figure_paths_split.append(diff_save_path)
                              except Exception as e: logger.error(f"  绘制瞬时图失败 (t={t_idx}, split={split}): {e}", exc_info=True)

            # 存储此分割的结果和图表路径
            all_eval_results[split] = {'metrics': metrics_to_save, 'figures': figure_paths_split}
            logger.info(f"--- 评估 {split} 完成 ---")

        except Exception as e:
            logger.error(f"评估分割 {split} 时出错: {e}", exc_info=True)
            all_eval_results[split] = {"error": str(e)}

    results['evaluation_details'] = all_eval_results
    results['success'] = any('metrics' in v for v in all_eval_results.values()) # 如果任何分割有指标则为成功

    for split, eval_data in all_eval_results.items():
        if 'metrics' in eval_data and 'mean_rmse' in eval_data['metrics']:
            logger.info(f"--- {split} Mean RMSE: {eval_data['metrics']['mean_rmse']:.6f}")

    logger.info(f"--- 评估完成 (DR: {dr_method}, Pred: {pred_method}) ---")
    return results

# === 主 Pipeline ===
@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """主集成流程"""
    start_run_time = time.time()
    # --- 配置和日志初始化 ---
    log_level = cfg.get('log_level', 'INFO').upper()
    logging.basicConfig(level=log_level,
                       format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
                       datefmt='%Y-%m-%d %H:%M:%S')
    # 确保 DrprConfig 在最开始解析，路径问题能尽早发现
    try:
         config = DrprConfig.from_hydra_config(cfg)
         logger.info("DrprConfig 解析成功。")
    except Exception as e:
         logger.error(f"解析 DrprConfig 失败: {e}. 请检查 Hydra 配置和路径。")
         sys.exit(1) # 配置失败则退出

    logger.info("==== 开始集成流程 ====")
    logger.debug(f"完整配置:\n{OmegaConf.to_yaml(cfg, resolve=True)}") # DEBUG 级别打印完整配置

    # --- 创建基础目录 (使用解析后的 config.paths) ---
    try:
        os.makedirs(config.paths.models_base_dir, exist_ok=True)
        os.makedirs(config.paths.processed_data_dir, exist_ok=True) # params dir
        # os.makedirs(config.paths.bmu_base_dir, exist_ok=True) # BMU dir (if needed)
        os.makedirs(config.paths.predictions_base_dir, exist_ok=True) # Base predictions dir
        os.makedirs(config.paths.reconstructions_base_dir, exist_ok=True) # Base reconstructions dir
        os.makedirs(config.paths.evaluation_base_dir, exist_ok=True) # Base evaluation dir
    except AttributeError as e:
         logger.error(f"创建基础目录失败，缺少路径配置: {e}. 请检查 conf/paths/default.yaml。")
         sys.exit(1)
    except Exception as e:
         logger.error(f"创建基础目录时出错: {e}")
         sys.exit(1)


    # 设置全局随机种子
    set_global_seeds(cfg.get('random_seed', 42)) # 使用 None 作为默认值更好

    # === 阶段 1: 降维 ===
    dr_results = run_dimensionality_reduction(cfg)
    if not dr_results or not dr_results.get('success'):
        logger.error("维度降低阶段失败，流程终止。")
        sys.exit(1) # 失败则退出

    # === 阶段 2: 预测 ===
    pred_results = run_prediction(cfg, dr_results)
    if not pred_results or not pred_results.get('success'):
        logger.error("预测阶段失败，流程终止。")
        sys.exit(1) # 失败则退出

    # === 阶段 3: 重建 ===
    recon_results = run_reconstruction(cfg, dr_results, pred_results)
    if not recon_results or not recon_results.get('success'):
        logger.error("重建阶段失败，流程终止。")
        sys.exit(1) # 失败则退出

    # === 阶段 4: 评估 ===
    eval_results = run_evaluation(cfg, recon_results, dr_results['method'], pred_results['method'])
    if not eval_results or not eval_results.get('success'):
        logger.warning("评估阶段失败或未生成任何结果。")
        # 评估失败不一定需要终止流程，但可以记录

    # --- 保存总体 Pipeline 结果到 Hydra 输出目录 ---
    final_summary = {
        'config': OmegaConf.to_container(cfg, resolve=True), # 保存解析后的配置
        'dimensionality_reduction_summary': {k: v for k, v in dr_results.items() if k not in ['low_dim_data_paths']}, # 移除大数据路径
        'prediction_summary': {k: v for k, v in pred_results.items() if k not in ['predicted_low_dim_paths']},
        'reconstruction_summary': {k: v for k, v in recon_results.items() if k != 'reconstructed_high_dim_paths'},
        'evaluation_summary': eval_results.get('evaluation_details', {}) # 保存详细评估指标
    }
    # 使用 Hydra 的当前工作目录 (即输出目录)
    hydra_output_dir = os.getcwd()
    summary_path = os.path.join(hydra_output_dir, "pipeline_summary.yaml") # 保存为 yaml 更易读
    try:
        OmegaConf.save(config=OmegaConf.create(final_summary), f=summary_path)
        logger.info(f"Pipeline 总结结果已保存到: {summary_path}")
    except Exception as e:
        logger.error(f"保存 Pipeline 总结失败: {e}")

    total_run_time = time.time() - start_run_time
    logger.info(f"==== 集成流程完成 (DR: {dr_results['method']}, Pred: {pred_results['method']}) ====")
    logger.info(f"总用时: {total_run_time:.2f} 秒")
    logger.info(f"Hydra 输出目录: {hydra_output_dir}")

if __name__ == "__main__":
    main()