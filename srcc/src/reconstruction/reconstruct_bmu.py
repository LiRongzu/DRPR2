# src/reconstruction/reconstruct_bmu.py


import numpy as np
import os
import logging
import joblib # 用于加载 hmm_params (通常是 .pkl)
import torch
from omegaconf import DictConfig # 导入 DictConfig
from typing import Optional, Any, Union # 导入 Union

# --- 项目相关的导入 ---
# *** 导入你实际的 SOMTorch 类 ***
from src.dimensionality_reduction.som_pytorch import SOMTorch
# *** 从 model_utils 导入通用加载器和设备获取函数 ***
from src.utils.model_utils import load_model as generic_load_model, get_device_from_config
# *** 导入其他需要的工具函数 ***
from src.utils.bmu_utils import convert_rank_to_linear, convert_grid_to_linear # <--- 修改这里

logger = logging.getLogger(__name__)

# --- 大幅简化后的 load_som_model 函数 ---
def load_som_model(model_path: str, device: Union[str, torch.device]) -> Optional[SOMTorch]:
    """
    加载 SOM 模型，直接调用 SOMTorch 类的 load 方法。
    """
    logger.info(f"尝试使用 SOMTorch.load 加载模型: {model_path}")
    try:
        # 直接调用 SOMTorch 类自带的 load 方法
        model = SOMTorch.load(model_path, device=device)
        if model is None:
             # SOMTorch.load 内部应该处理了错误日志
             return None
        # 验证加载的模型是否正确 (可选)
        if not hasattr(model, 'weights') or not hasattr(model, 'map_size'):
             logger.error("加载的 SOM 模型对象不完整。")
             return None
        # 现在 model.feature_name 应该由 SOMTorch.load 设置好了
        logger.info(f"通过 SOMTorch.load 成功加载模型。特征名: {model.feature_name})")
        return model
    except FileNotFoundError:
         logger.error(f"模型文件未找到: {model_path}")
         return None
    except Exception as e:
        logger.error(f"使用 SOMTorch.load 加载模型 {model_path} 时发生意外错误: {e}", exc_info=True)
        return None

# --- reconstruct_from_bmu 函数保持不变 ---
# 这个函数现在调用上面简化的 load_som_model，并且能正确获取 som_model 实例
# 其内部获取 feature_name 的逻辑 (som_model.feature_name) 也能工作了
def reconstruct_from_bmu(
    cfg: DictConfig,
    som_model_path: str,
    predicted_bmu_path: str,
    output_path: str,
    hmm_params_path: str = None
) -> np.ndarray | None:
    """
    根据预测的 BMU 序列重建高维数据。
    支持使用 SOM 权重或簇类平均值进行重建，通过 cfg 控制。
    *** 使用简化后的 load_som_model ***
    """
    try:
        # --- 0. 获取配置和设备 ---
        recon_method = cfg.reconstruction.bmu.get('som_reconstruction_method', 'weights')
        logger.info(f"开始 BMU 重建，方法: '{recon_method}'")
        device = get_device_from_config(cfg) # 从 utils 获取设备

        # --- 1. 加载 SOM 模型 (调用简化后的加载函数) ---
        som_model = load_som_model(som_model_path, device) # 调用我们上面修改的函数
        if som_model is None:
             logger.error(f"无法加载 SOM 模型: {som_model_path}，重建中止。")
             return None

        # --- 从加载成功的模型实例中获取信息 ---
        map_size = som_model.map_size
        n_features = som_model.input_dim
        feature_name = som_model.feature_name # 从模型实例获取特征名
        n_nodes = map_size[0] * map_size[1]

        # 检查 cluster_average 方法是否可行
        if recon_method == 'cluster_average' and feature_name is None:
             logger.error("SOM 模型加载后未包含 'feature_name'，无法在 'cluster_average' 模式下确定平均值文件路径。")
             logger.warning("将回退到使用 SOM 权重进行重建。")
             recon_method = 'weights'

        # --- 2. 加载预测的 BMU 序列并转换为线性索引 ---
        # ... (逻辑不变) ...
        predicted_bmus = np.load(predicted_bmu_path)
        if hmm_params_path and os.path.exists(hmm_params_path):
            # ... (加载 rank_map 并调用 convert_rank_to_linear) ...
             try:
                  hmm_params = joblib.load(hmm_params_path)
                  state_rank_map = hmm_params.get('state_rank_map')
                  if state_rank_map is None: logger.error(...); return None
                  bmu_indices = convert_rank_to_linear(predicted_bmus.flatten(), {rank: linear for linear, rank in state_rank_map.items()}) # 注意这里需要 rank->linear 的映射
             except Exception as e: logger.error(...); return None
        else:
            bmu_indices = predicted_bmus.flatten().astype(int)

        # --- 3. 检查 BMU 索引有效性 ---
        # (这部分逻辑保持不变)
        if np.any(bmu_indices < 0) or np.any(bmu_indices >= n_nodes):
             min_idx, max_idx = np.min(bmu_indices), np.max(bmu_indices)
             logger.error(f"预测的 BMU 索引越界 (应在 [0, {n_nodes-1}] 范围内)。检测到范围: [{min_idx}, {max_idx}]")
             # ... (记录无效索引的日志) ...
             return None

        # --- 4. 执行重建 ---
        reconstructed_flat = None

        # --- 尝试使用簇类平均值 ---
        if recon_method == 'cluster_average':
            # (这部分逻辑保持不变，使用 feature_name 来查找文件)
            logger.info("尝试使用簇类平均值进行重建...")
            averages_filename = f"som_cluster_averages_{feature_name}.npy"
            averages_path = os.path.join(os.path.dirname(som_model_path), averages_filename)
            # ... (检查备用路径 cfg.paths.processed_data_dir) ...
            if not os.path.exists(averages_path):
                 try:
                      processed_dir = cfg.paths.processed_data_dir
                      alt_averages_path = os.path.join(processed_dir, averages_filename)
                      if os.path.exists(alt_averages_path): averages_path = alt_averages_path
                      else: logger.error(...); recon_method = 'weights' # 回退
                 except Exception as path_e: logger.error(...); recon_method = 'weights'

            if recon_method == 'cluster_average':
                try:
                    cluster_averages = np.load(averages_path)
                    logger.info(f"成功加载簇类平均值，形状: {cluster_averages.shape}")
                    # ... (检查形状和 NaN) ...
                    if cluster_averages.shape != (n_nodes, n_features):
                        logger.error(...); recon_method = 'weights'
                    else:
                        # ... NaN 检查 ...
                        reconstructed_flat = cluster_averages[bmu_indices]
                        logger.info(f"使用簇类平均值重建完成...")
                except Exception as e:
                    logger.error(f"加载或使用簇类平均值时出错: {e}", exc_info=True)
                    logger.warning("将回退到使用 SOM 权重进行重建。")
                    recon_method = 'weights'

        # --- 如果方法是 'weights' (或从 'cluster_average' 回退) ---
        if recon_method == 'weights': # 注意这里是 if 而不是 elif，因为可能从上面回退过来
            logger.info("使用 SOM 权重进行重建...")
            try:
                if hasattr(som_model, 'weights') and isinstance(som_model.weights, torch.Tensor):
                     weights_tensor = som_model.weights
                     weights_np = weights_tensor.cpu().numpy() # 转到 CPU 并转为 NumPy
                     weights_flat = weights_np.reshape(-1, n_features) # (H*W, Feat)
                     reconstructed_flat = weights_flat[bmu_indices]
                     logger.info(f"使用权重重建完成，输出形状: {reconstructed_flat.shape}")
                else:
                     logger.error("加载的 SOM 模型对象缺少 'weights' 属性或类型不正确。")
                     return None # 无法使用权重重建

            except Exception as e:
                logger.error(f"使用 SOM 权重重建时出错: {e}", exc_info=True)
                return None # 如果权重重建失败，则中止

        # --- 5. 保存结果 ---
        # (这部分逻辑保持不变)
        if reconstructed_flat is not None:
            try:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                np.save(output_path, reconstructed_flat)
                logger.info(f"重建结果 (扁平化) 已保存到: {output_path}")
                return reconstructed_flat
            except Exception as e:
                logger.error(f"保存重建结果失败: {e}")
                return None
        else:
            logger.error("重建失败，未生成有效结果。")
            return None

    # --- 外部错误处理 ---
    except FileNotFoundError as e:
        logger.error(f"文件未找到错误: {e}")
        return None
    except Exception as e:
        logger.error(f"BMU 重建过程中发生未知错误: {e}", exc_info=True)
        return None