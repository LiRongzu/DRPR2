# src/training/train_som.py

import os
import sys
import numpy as np
import logging
import time
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from typing import Optional, Dict, Any, List, Tuple
import torch.nn.functional as F
from src.utils.data_loader import load_processed_data

# --- 项目设置 ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if (project_root not in sys.path):
    sys.path.insert(0, project_root)

from src.utils.hydra_config import DrprConfig
from src.utils.model_utils import  get_device_from_config
from src.dimensionality_reduction.som_pytorch import SOMTorch
from src.utils.generate_BMU import generate_BMU
from src.utils.bmu_utils import calculate_som_cluster_averages

logger = logging.getLogger(__name__)


def train_single_feature_som(
    cfg: DictConfig,
    feature_name: str = "salinity" # 将 feature_type 改为 feature_name
) -> Dict[str, Any]:
    """
    训练单个特征的SOM模型并生成BMU索引

    参数:
        cfg: 配置对象
        feature_name: 特征类型（'salinity', 'wind', 或 'flow'）

    返回:
        Dict: 包含训练结果信息的字典 (model_path, bmu_paths 等)
    """
    from src.utils.data_loader import load_processed_data  # 确保从集中化data_loader导入

    config = DrprConfig.from_hydra_config(cfg)
    device = get_device_from_config(cfg)
    results = {}  # 用于存储返回信息

    logger.info(f"开始训练 {feature_name} 特征的 SOM 模型...")
    start_time = time.time()

    # --- 1. 准备数据 ---
    all_features = {}
    for split in ['train', 'val', 'test']:
        try:
            # 加载处理过的数据
            try:
                # 使用集中化的数据加载函数
                features = load_processed_data(cfg, feature_name, split)
                if features is not None:
                    original_shape = features.shape
                    if features.ndim > 2:
                        # 如果特征是3维或更高，将时间维(通常是第一维)之外的维度展平
                        features = features.reshape(features.shape[0], -1)
                    logger.info(f"将 {split} {feature_name} 数据从 {original_shape} 重塑为 {features.shape}")
                    all_features[split] = features
                else:
                    logger.warning(f"{split} 分割的 {feature_name} 数据未找到，跳过。")
                    continue
            except FileNotFoundError:
                logger.warning(f"{split} 分割的 {feature_name} 数据文件未找到，跳过。")
                continue
            except Exception as e:
                logger.error(f"加载 {split} {feature_name} 数据时发生未知错误: {e}")
                continue
        except Exception as e:
            logger.error(f"准备 {split} {feature_name} 数据时发生错误: {e}")
            continue

    if 'train' not in all_features:
        logger.error(f"训练集数据不存在，无法训练 {feature_name} SOM 模型")
        return results

    # --- 2. 初始化SOM模型 ---
    train_data = all_features['train']
    try:
        som_config = cfg.training.som
        map_size = (som_config.map_size[0], som_config.map_size[1])  # SOM地图大小
        sigma = som_config.get("sigma", 1.0)
        learning_rate = som_config.get("learning_rate", 0.5)
        n_iterations = som_config.get("n_iterations", 200)
        model = SOMTorch(
            input_dim=train_data.shape[1],
            map_size=map_size,
            sigma=sigma,
            learning_rate=learning_rate,
            device=device, # 传递设备信息以支持GPU
            n_iterations=n_iterations,
            feature_name=feature_name # 将特征名称传递给模型,
        )
        logger.info(f"{feature_name} SOM 模型初始化完成，地图大小: {map_size}")
    except Exception as e:
        logger.error(f"{feature_name} SOM 模型初始化失败: {e}")
        return results
    # print(f"初始化后 model.feature_name: {model.feature_name}")
    # --- 3. 训练模型 ---
    logger.info(f"开始训练 {feature_name} SOM 模型...")
    try:
        train_start_time = time.time()
        model.fit(train_data)
        
        training_time = time.time() - train_start_time
        logger.info(f"{feature_name} 模型训练完成，用时: {training_time:.2f}秒")
    except Exception as e:
        logger.error(f"{feature_name} 模型训练失败: {e}")
        return results
    
    # --- 4. 保存模型 ---
    # 确保保存目录存在
    if not hasattr(config.paths, "som_models_dir"):
        logger.warning("配置中缺少 'paths.som_models_dir'，使用当前目录作为替代。")
        model_save_dir = "."
    else:
        model_save_dir = config.paths.som_models_dir
        os.makedirs(model_save_dir, exist_ok=True)
    
    model_name = f"som_model_{feature_name}.pt"
    model_path = os.path.join(model_save_dir, model_name)
    

    model.save(model_path)
    logger.info(f"{feature_name} SOM 模型已保存到: {model_path}")
    results['model_path'] = model_path # 将 model_path 加入结果

    logger.info(f"开始为 {feature_name} 计算 Cluster Averages (调用 bmu_utils)...")
    # 确保 train_data 是正确的训练数据
    train_data_for_avg = all_features.get('train')
    if train_data_for_avg is not None:
        cluster_averages = calculate_som_cluster_averages(model, train_data_for_avg)

        if cluster_averages is not None:
            # 构建保存路径 (与模型保存在同一目录)
            averages_filename = f"som_cluster_averages_{feature_name}.npy"
            averages_save_path = os.path.join(os.path.dirname(model_path), averages_filename)

            # 保存 Cluster Averages
            try:
                np.save(averages_save_path, cluster_averages)
                logger.info(f"  Cluster Averages (由 bmu_utils 计算) 已保存到: {averages_save_path}")
                if 'post_train_outputs' not in results: results['post_train_outputs'] = {}
                results['post_train_outputs']['cluster_averages_path'] = averages_save_path
            except Exception as e:
                logger.error(f"保存 Cluster Averages 时出错: {e}")
        else:
            logger.error(f"未能从 bmu_utils 成功计算 Cluster Averages。")
    else:
        logger.error("无法计算 Cluster Averages，因为训练数据不可用。")



    # --- 5. 生成BMU (Best Matching Units) ---
    # --- MODIFICATION START: Call the new function ---
    # 确保BMU目录存在
    if not hasattr(config.paths, "bmu_base_dir"):
        logger.warning("配置中缺少 'paths.bmu_base_dir'，使用当前目录作为替代。")
        bmu_base_dir = "."
    else:
        bmu_base_dir = config.paths.bmu_base_dir

    # 调用新函数生成并保存 BMU
    bmu_paths = generate_and_save_bmus(
        model=model,
        all_features=all_features,
        feature_name=feature_name,
        bmu_base_dir=bmu_base_dir,
        map_size=map_size # Pass the map_size determined earlier
    )


    # --- 6. 生成DV (Distance Vector) ---
    # 确保DV目录存在
    if not hasattr(config.paths, "dv_base_dir"):
        logger.warning("配置中缺少 'paths.dv_base_dir'，使用当前目录作为替代。")
        dv_base_dir = "."
    else:
        dv_base_dir = config.paths.dv_base_dir
        os.makedirs(dv_base_dir, exist_ok=True) # Ensure directory exists

    # 调用新函数生成并保存 DV
    dv_paths = generate_and_save_dvs(
        model=model,
        all_features=all_features,
        feature_name=feature_name,
        dv_base_dir=dv_base_dir,
        map_size=map_size # Pass the map_size determined earlier
    )
    # --- MODIFICATION END ---

    total_run_time = time.time() - start_time
    logger.info(f"{feature_name} 特征的 SOM 训练和 数据生成完成，总用时：{total_run_time:.2f}秒")


    # 结束训练，返回结果
    results = {
        'feature_name': feature_name,
        'model_path': model_path,
        'bmu_indices_paths': bmu_paths,
        'dv_paths': dv_paths # Add DV paths to results
    }
    return results

def combine_features(
        cfg: DictConfig,
        wind_data: np.ndarray,
        flow_data: np.ndarray,
        # --- 加载预处理后的 wind 和 flow 数据 ---
    # 处理所有分割
    ):
    splits = cfg.training.spilt_group
    combined_features_dict = {}
    
    for split in splits:
        try:
            wind_data = load_processed_data(cfg, "wind", split)
            
            flow_data = load_processed_data(cfg, "flow", split)
            
            # 确保时间步长一致
            min_T = min(wind_data.shape[0], flow_data.shape[0])
            if wind_data.shape[0] != min_T or flow_data.shape[0] != min_T:
                logger.warning(f"{split}: Wind ({wind_data.shape[0]}) 和 Flow ({flow_data.shape[0]}) 时间步长不匹配，截断为 {min_T}")
                wind_data = wind_data[:min_T,:]
                try:
                    flow_data = flow_data[:min_T,:]
                except Exception as e:
                    logger.error(f"处理 {split} 分割的 flow 数据时发生错误,数据可能是一维: {e}")
                    flow_data = flow_data[:min_T]
                    continue

            # 特征组合
            # 1. 风场降采样或展平
            if wind_data.ndim > 2:
                T, C, H, W = wind_data.shape
                compressed_rate = cfg.training.wind_compression
                target_H, target_W = H // compressed_rate, W // compressed_rate
                # 转换为 Tensor 进行插值
                wind_tensor = torch.tensor(wind_data, dtype=torch.float32)
                # 使用 'bilinear' 或 'nearest' 等模式
                wind_downsampled_tensor = F.interpolate(wind_tensor, size=(target_H, target_W), mode='bilinear', align_corners=False)
                wind_downsampled = wind_downsampled_tensor.numpy()
                # 展平空间维度
                wind_downsampled_flat = wind_downsampled.reshape(T, -1)
                logger.info(f"{split}: Wind 数据降采样后形状: {wind_downsampled.shape}, 展平后: {wind_downsampled_flat.shape}")
            else:
                wind_downsampled_flat = wind_data
            
            # 2. 处理流量数据 (如果是1D则重塑，基于特征数量处理)
            if flow_data.ndim == 1:
                flow_data = flow_data.reshape(-1, 1) # Reshape to (T, 1)

            target_feature_dim = wind_downsampled_flat.shape[1] # Wind features
            flow_feature_dim = flow_data.shape[1] # Flow features

            if flow_feature_dim == 1:
                # 情景: 平均流量 (1 个特征). 广播以匹配风特征.
                flow_data_processed = np.broadcast_to(flow_data, (flow_data.shape[0], target_feature_dim))
                logger.info(f"{split}: 检测到单个流量特征. 从 {flow_data.shape} 广播到 {flow_data_processed.shape} 以匹配风特征 ({target_feature_dim}).")
            elif flow_feature_dim > 1:
                # 情景: 独立流量 (例如, 3 个特征). 直接用于拼接.
                target_feature_dim = cfg.training.flow_boradcast
                flow_data_processed = np.broadcast_to(flow_data, (flow_data.shape[0], target_feature_dim ))
                logger.info(f"{split}: 检测到多个流量特征 ({flow_feature_dim}). 直接用于拼接.")
            else: # flow_feature_dim is 0?
                 logger.error(f"{split}: 流量数据在处理后特征数量为零. 形状: {flow_data.shape}. 跳过此分割.")
                 continue # Skip this split if flow data is invalid

            # 3. 特征拼接 (Wind + 处理过的 Flow)
            combined_features = np.concatenate((wind_downsampled_flat, flow_data_processed), axis=1)
            logger.info(f"{split}: 组合 (Wind + Flow) 特征形状: {combined_features.shape}")

            
            # 存储组合特征
            combined_features_dict[split] = combined_features

        except Exception as e:
            logger.error(f"{split} 分割的特征组合失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
        
        return combined_features_dict

def train_combined_feature_som(cfg: DictConfig, output_feature_name: str = "wind_flow") -> Dict[str, Any]:
    """
    加载 wind 和 flow 数据，合并后训练 SOM，生成 BMU。

    Args:
        cfg: Hydra 配置对象
        output_feature_name: 输出 BMU 文件中使用的特征名称 (例如 "wind_flow")

    Returns:
        包含模型路径和 BMU 路径的字典
    """
    start_run_time = time.time()
    config = DrprConfig.from_hydra_config(cfg)
    device = get_device_from_config(cfg)
    map_size = cfg.training.som.map_size_obs

    logger.info(f"开始为组合特征 '{output_feature_name}' 训练 SOM...")

    # --- 加载预处理后的 wind 和 flow 数据 ---
    # 处理所有分割
    splits = cfg.training.spilt_group
    combined_features_dict = {}
    
    combined_features_dict = combine_features(cfg, "wind", "flow")
    
    # 确保训练集数据存在
    if 'train' not in combined_features_dict:
        logger.error(f"训练集数据不存在，无法训练组合特征 SOM 模型")
        return {}
    
    # 获取训练集数据
    train_combined = combined_features_dict['train']
    input_dim_combined = train_combined.shape[1]

    # --- 初始化和训练 SOM ---
    try:
        # 初始化模型
        model = SOMTorch(
            input_dim=input_dim_combined,
            map_size=map_size,
            learning_rate=cfg.training.som.learning_rate,
            n_iterations=cfg.training.som.n_iterations,
            device=device,
            feature_name=output_feature_name,
        )
        logger.info(f"组合特征 SOM 模型初始化完成，地图大小: {map_size}, 输入维度: {input_dim_combined}")

        # 转换为 Tensor
        train_combined_tensor = torch.tensor(train_combined, dtype=torch.float32).to(device)

        # 训练模型
        logger.info(f"开始训练组合特征 '{output_feature_name}' 的 SOM...")
        train_start_time = time.time()
        model.fit(train_combined_tensor) 
        training_time = time.time() - train_start_time
        logger.info(f"组合特征 SOM 训练完成，用时: {training_time:.2f}秒")

    except Exception as e:
        logger.error(f"组合特征 SOM 初始化或训练失败: {e}")
        return {}

    # --- 保存 SOM 模型 ---
    # 使用从配置中获取的路径
    model_save_dir = os.path.join(config.paths.som_models_dir)
    os.makedirs(model_save_dir, exist_ok=True)
    
    model_name = f"som_model_{output_feature_name}.pt" # 例如 som_model_wind_flow.pt
    model_path = os.path.join(model_save_dir, model_name)
    
    model.save(model_path)
        
    logger.info(f"组合特征 SOM 模型已保存到: {model_path}")

    # --- 生成 BMU ---
    logger.info(f"开始为组合特征 '{output_feature_name}' 生成 BMU...")
    bmu_start_time = time.time()
    
    # 创建BMU目录
    bmu_dir = os.path.join(config.paths.bmu_base_dir)
    os.makedirs(bmu_dir, exist_ok=True)

    bmu_paths_combined = {}
    
    for split in splits:
        # 跳过没有加载的分割
        if split not in combined_features_dict:
            continue
            
        bmu_positions_path = os.path.join(bmu_dir, f"bmu_positions_{output_feature_name}_{split}.npy")
        bmu_map_path = os.path.join(bmu_dir, f"bmu_map_{output_feature_name}_{split}.npy")
        bmu_freq_path = os.path.join(bmu_dir, f"bmu_freq_{output_feature_name}_{split}.npy")

        try:
            bmu_positions = generate_BMU(
                model, 
                combined_features_dict[split],
                bmu_positions_path=bmu_positions_path,
                bmu_map_path=bmu_map_path,
                bmu_freq_path=bmu_freq_path
            )
            
            if bmu_positions is None:
                logger.error(f"组合特征 {split} 分割的 BMU 生成失败")
                continue

            bmu_paths_combined[split] = {
                'positions': bmu_positions_path,
                'map': bmu_map_path,
                # --- MODIFICATION START ---
                'freq': bmu_freq_path # Change key from 'frequency' to 'freq'
                # --- MODIFICATION END ---
            }
            
            logger.info(f"组合特征 {split} 分割的 BMU 已生成")
            
        except Exception as e:
            logger.error(f"组合特征 {split} 分割的 BMU 生成失败: {e}")

    bmu_generation_time = time.time() - bmu_start_time
    logger.info(f"组合特征 BMU 生成完成，用时: {bmu_generation_time:.2f}秒")

    total_run_time = time.time() - start_run_time
    logger.info(f"组合特征 '{output_feature_name}' 的 SOM 训练和 BMU 生成完成，总用时：{total_run_time:.2f}秒")

    return {
        'feature_name': output_feature_name,
        'model_path': model_path,
        'map_size': map_size,
        'training_time': training_time,
        'bmu_generation_time': bmu_generation_time,
        # --- MODIFICATION START ---
        'bmu_indices_paths': bmu_paths_combined # Change key from 'bmu_paths' to 'bmu_indices_paths'
        # --- MODIFICATION END ---
    }

def generate_and_save_bmus(
    model: SOMTorch,
    all_features: Dict[str, np.ndarray],
    feature_name: str,
    bmu_base_dir: str,
    map_size: Tuple[int, int]
) -> Dict[str, Dict[str, str]]:
    """
    使用训练好的 SOM 模型为不同数据分割生成 BMU 并保存。

    Args:
        model: 训练好的 SOMTorch 模型。
        all_features: 包含各分割数据的字典 {split: data_array}。
        feature_name: 当前特征的名称。
        bmu_base_dir: 保存 BMU 文件的根目录。
        map_size: SOM 地图的大小 (height, width)。

    Returns:
        一个字典，包含每个分割对应的 BMU 文件路径 {'split': {'positions': path, 'map': path, 'freq': path}}。
    """
    logger.info(f"开始为 {feature_name} 生成 BMU...")
    bmu_start_time = time.time()

    # 确保BMU目录存在
    bmu_dir = os.path.join(bmu_base_dir) # Removed redundant join
    os.makedirs(bmu_dir, exist_ok=True)

    bmu_paths = {}  # 存储各个分割的BMU路径

    for split, data in all_features.items():
        logger.info(f"计算 {split} 数据的 BMUs...")
        try:
            if data is None or data.size == 0:
                logger.warning(f"{split} 数据为空，跳过 BMU 计算。")
                continue

            # 使用模型计算 BMU 索引 (坐标)
            # Assuming model.transform returns (n_samples, 2) array of (row, col)
            bmu_indices = model.transform(data)

            if bmu_indices is None or bmu_indices.size == 0:
                 logger.warning(f"模型未能为 {split} 数据生成 BMU 索引。")
                 continue

            # 定义保存路径
            bmu_positions_path = os.path.join(bmu_dir, f"bmu_positions_{feature_name}_{split}.npy")
            bmu_map_path = os.path.join(bmu_dir, f"bmu_map_{feature_name}_{split}.npy")
            bmu_freq_path = os.path.join(bmu_dir, f"bmu_freq_{feature_name}_{split}.npy")

            # 1. 保存 BMU 坐标 (row, col)
            np.save(bmu_positions_path, bmu_indices)

            # 2. 计算和保存 BMU map (每个节点最后映射到的样本索引)
            bmu_map = np.full(map_size, -1, dtype=int) # Use np.full for clarity
            # Ensure bmu_indices is a 2D array (N, 2)
            if bmu_indices.ndim == 2 and bmu_indices.shape[1] == 2:
                for i, (row, col) in enumerate(bmu_indices):
                    # Check bounds just in case transform returns invalid indices
                    if 0 <= row < map_size[0] and 0 <= col < map_size[1]:
                        bmu_map[row, col] = i
                    else:
                        logger.warning(f"样本 {i} 的 BMU 索引 ({row}, {col}) 超出地图边界 {map_size}，已忽略。")
            else:
                 logger.error(f"BMU 索引形状不符合预期 ({bmu_indices.shape})，无法计算 BMU map。")
                 # Handle error - perhaps skip saving map?
                 bmu_map_path = None # Indicate map wasn't saved

            if bmu_map_path: # Only save if path is valid
                np.save(bmu_map_path, bmu_map)

            # 3. 计算和保存 BMU 频率分布
            bmu_freq = np.zeros(map_size, dtype=int)
            if bmu_indices.ndim == 2 and bmu_indices.shape[1] == 2:
                valid_indices = bmu_indices[(bmu_indices[:, 0] >= 0) & (bmu_indices[:, 0] < map_size[0]) &
                                            (bmu_indices[:, 1] >= 0) & (bmu_indices[:, 1] < map_size[1])]
                # Use np.add.at for efficient counting at specific indices
                np.add.at(bmu_freq, (valid_indices[:, 0], valid_indices[:, 1]), 1)
            else:
                 logger.error(f"BMU 索引形状不符合预期 ({bmu_indices.shape})，无法计算 BMU 频率。")
                 # Handle error - perhaps skip saving freq?
                 bmu_freq_path = None # Indicate freq wasn't saved

            if bmu_freq_path: # Only save if path is valid
                np.save(bmu_freq_path, bmu_freq)

            logger.info(f"  {split} BMU 已保存: positions={bmu_positions_path}, map={bmu_map_path}, freq={bmu_freq_path}")

            # 记录到结果字典 (only add paths if they were successfully created)
            split_paths = {}
            if os.path.exists(bmu_positions_path): split_paths['positions'] = bmu_positions_path
            if bmu_map_path and os.path.exists(bmu_map_path): split_paths['map'] = bmu_map_path
            if bmu_freq_path and os.path.exists(bmu_freq_path): split_paths['freq'] = bmu_freq_path

            if split_paths: # Only add if at least one file was saved
                bmu_paths[split] = split_paths

        except Exception as e:
            logger.error(f"为 {split} 生成 BMU 时发生错误: {e}", exc_info=True) # Add traceback
            continue # Continue with the next split

    bmu_generation_time = time.time() - bmu_start_time
    logger.info(f"{feature_name} BMU 生成完成，用时: {bmu_generation_time:.2f}秒")
    return bmu_paths

def generate_and_save_dvs(
    model: SOMTorch,
    all_features: Dict[str, np.ndarray],
    feature_name: str,
    dv_base_dir: str,
    map_size: Tuple[int, int] # map_size might not be strictly needed here but kept for consistency
) -> Dict[str, str]:
    """
    使用训练好的 SOM 模型为不同数据分割生成距离向量 (DV) 并保存。

    Args:
        model: 训练好的 SOMTorch 模型。
        all_features: 包含各分割数据的字典 {split: data_array}。
        feature_name: 当前特征的名称。
        dv_base_dir: 保存 DV 文件的根目录。
        map_size: SOM 地图的大小 (height, width).

    Returns:
        一个字典，包含每个分割对应的 DV 文件路径 {'split': path}。
    """
    logger.info(f"开始为 {feature_name} 生成 Distance Vectors (DV)...")
    dv_start_time = time.time()

    # 确保DV目录存在
    os.makedirs(dv_base_dir, exist_ok=True)

    dv_paths = {}  # 存储各个分割的DV路径

    for split, data in all_features.items():
        logger.info(f"计算 {split} 数据的 DVs...")
        try:
            if data is None or data.size == 0:
                logger.warning(f"{split} 数据为空，跳过 DV 计算。")
                continue

            # 使用模型计算 Distance Vectors
            distance_vectors = model.compute_distance_vectors(data)

            if distance_vectors is None or distance_vectors.size == 0:
                 logger.warning(f"模型未能为 {split} 数据生成 Distance Vectors。")
                 continue

            # 定义保存路径
            dv_save_path = os.path.join(dv_base_dir, f"dv_{feature_name}_{split}.npy")

            # 保存 Distance Vectors
            np.save(dv_save_path, distance_vectors)
            logger.info(f"  {split} DV 已保存: {dv_save_path}")

            # 记录到结果字典
            dv_paths[split] = dv_save_path

        except Exception as e:
            logger.error(f"为 {split} 生成 DV 时发生错误: {e}", exc_info=True) # Add traceback
            continue # Continue with the next split

    dv_generation_time = time.time() - dv_start_time
    logger.info(f"{feature_name} DV 生成完成，用时: {dv_generation_time:.2f}秒")
    return dv_paths


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """主函数入口"""
    # 解析命令行参数，确定要训练哪些特征的 SOM
    feature_to_train = cfg.get("feature_to_train", "salinity")

    # 确保路径目录存在
    config = DrprConfig.from_hydra_config(cfg)
    os.makedirs(config.paths.som_models_dir, exist_ok=True)
    os.makedirs(config.paths.bmu_base_dir, exist_ok=True)
    # Ensure dv_base_dir exists as well
    if hasattr(config.paths, "dv_base_dir"):
        os.makedirs(config.paths.dv_base_dir, exist_ok=True)
    else:
        logger.warning("配置中缺少 'paths.dv_base_dir'，DV 文件将保存在当前目录或由函数内部处理。")


    # 根据指定特征选择训练函数
    if (feature_to_train == "all"):
        logger.info("训练所有特征的 SOM 模型")
        results = train_multiple_soms(cfg)
    elif (feature_to_train == "wind_flow"):
        logger.info("训练 wind_flow 组合特征的 SOM 模型")
        results = train_combined_feature_som(cfg, "wind_flow")
    else:
        logger.info(f"训练 {feature_to_train} 特征的 SOM 模型")
        results = train_single_feature_som(cfg, feature_to_train)
    
    # 检查结果
    if not results:
        logger.error("SOM 模型训练失败")
        sys.exit(1)
    
    logger.info("SOM 模型训练完成")
    sys.exit(0)

if __name__ == "__main__":
    main()