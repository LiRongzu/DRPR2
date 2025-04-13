# src/training/train_som.py

import os
import sys
import numpy as np
import logging
import time
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from typing import Optional, Dict, Any, List
# 假设你有一个插值或降采样库，例如 scipy 或 torch.nn.functional
from scipy.interpolate import interpn # 或者其他降采样方法
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

    # --- 修改：调用新函数计算并保存 Cluster Averages ---

    logger.info(f"开始为 {feature_name} 计算 Cluster Averages (调用 bmu_utils)...")
    # 确保 train_data 是正确的训练数据
    train_data_for_avg = all_features.get('train')
    if train_data_for_avg is not None:
        # 调用封装好的函数
        cluster_averages = calculate_som_cluster_averages(model, train_data_for_avg)

        if cluster_averages is not None:
            # 构建保存路径 (与模型保存在同一目录)
            averages_filename = f"som_cluster_averages_{feature_name}.npy"
            averages_save_path = os.path.join(os.path.dirname(model_path), averages_filename)

            # 保存 Cluster Averages
            try:
                np.save(averages_save_path, cluster_averages)
                logger.info(f"  Cluster Averages (由 bmu_utils 计算) 已保存到: {averages_save_path}")
                # （可选）将路径添加到返回结果中
                if 'post_train_outputs' not in results: results['post_train_outputs'] = {}
                results['post_train_outputs']['cluster_averages_path'] = averages_save_path
            except Exception as e:
                logger.error(f"保存 Cluster Averages 时出错: {e}")
        else:
            logger.error(f"未能从 bmu_utils 成功计算 Cluster Averages。")
    else:
        logger.error("无法计算 Cluster Averages，因为训练数据不可用。")



    # --- 5. 生成BMU (Best Matching Units) ---
    logger.info(f"开始为 {feature_name} 生成 BMU...")
    
    # 确保BMU目录存在
    if not hasattr(config.paths, "bmu_base_dir"):
        logger.warning("配置中缺少 'paths.bmu_base_dir'，使用当前目录作为替代。")
        bmu_base_dir = "."
    else:
        bmu_base_dir = config.paths.bmu_base_dir
    
    # 每个特征类型有自己的子目录————舍弃
    bmu_dir = os.path.join(bmu_base_dir)
    os.makedirs(bmu_dir, exist_ok=True)
    
    bmu_paths = {}  # 存储各个分割的BMU路径

    for split, data in all_features.items():
        logger.info(f"计算 {split} 数据的 BMUs...")
        try:
            if data is None or data.size == 0:
                logger.warning(f"{split} 数据为空，跳过 BMU 计算。")
                continue
                
            bmu_indices = model.transform(data)  # shape: (n_samples, 2), (n_samples,)
            
            # 保存 BMU 坐标 (row, col)
            bmu_positions_path = os.path.join(bmu_dir, f"bmu_positions_{feature_name}_{split}.npy")
            bmu_map_path = os.path.join(bmu_dir, f"bmu_map_{feature_name}_{split}.npy")
            bmu_freq_path = os.path.join(bmu_dir, f"bmu_freq_{feature_name}_{split}.npy")
            
            # 保存 BMU 坐标
            np.save(bmu_positions_path, bmu_indices)
            
            # 计算和保存 BMU map (每个节点最近的样本)
            map_height, map_width = map_size
            bmu_map = np.zeros(map_size, dtype=int) - 1  # 默认为 -1 (无样本)
            for i, (row, col) in enumerate(bmu_indices):
                # 如果多个样本映射到同一个节点，保留最后一个
                bmu_map[row, col] = i
            np.save(bmu_map_path, bmu_map)
            
            # 计算和保存 BMU 频率分布
            bmu_freq = np.zeros(map_size, dtype=int)
            for row, col in bmu_indices:
                bmu_freq[row, col] += 1
            np.save(bmu_freq_path, bmu_freq)
            
            logger.info(f"  {split} BMU 已保存: positions={bmu_positions_path}, map={bmu_map_path}, freq={bmu_freq_path}")
            
            # 记录到结果字典
            bmu_paths[split] = {
                'positions': bmu_positions_path,
                'map': bmu_map_path,
                'freq': bmu_freq_path # Key is already 'freq', OK
            }
            
        except Exception as e:
            logger.error(f"为 {split} 生成 BMU 时发生错误: {e}")
            continue
    
    total_run_time = time.time() - start_time
    logger.info(f"{feature_name} 特征的 SOM 训练和 BMU 生成完成，总用时：{total_run_time:.2f}秒")
    
    # 返回结果
    results = {
        'feature_name': feature_name,
        'model_path': model_path,
        # --- MODIFICATION START ---
        'bmu_indices_paths': bmu_paths # Change key from 'bmu_paths' to 'bmu_indices_paths'
        # --- MODIFICATION END ---
    }
    return results

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
    splits = ['train', 'val', 'test']
    combined_features_dict = {}
    
    for split in splits:
        try:
            # 使用 data_loader 加载风场数据
            wind_data = load_processed_data(cfg, "wind", split)
            
            # 使用 data_loader 加载径流数据
            flow_data = load_processed_data(cfg, "flow", split)
            
            # 确保时间步长一致
            min_T = min(wind_data.shape[0], flow_data.shape[0])
            if wind_data.shape[0] != min_T or flow_data.shape[0] != min_T:
                logger.warning(f"{split}: Wind ({wind_data.shape[0]}) 和 Flow ({flow_data.shape[0]}) 时间步长不匹配，截断为 {min_T}")
                wind_data = wind_data[:min_T]
                flow_data = flow_data[:min_T]

            # 特征组合
            # 1. 风场降采样或展平
            if wind_data.ndim > 2:
                T, C, H, W = wind_data.shape
                target_H, target_W = H // 2, W // 2
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
            
            # 2. 径流拼接
            combined_features = np.concatenate((wind_downsampled_flat, flow_data), axis=1)
            logger.info(f"{split}: 组合 (Wind + Flow) 特征形状: {combined_features.shape}")
            
            # 存储组合特征
            combined_features_dict[split] = combined_features

        except Exception as e:
            logger.error(f"{split} 分割的特征组合失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
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

# 实用函数：整合多个特征 SOM 的训练
def train_multiple_soms(
    cfg: DictConfig, 
    feature_types: List[str] = ["salinity", "wind", "flow"],
    train_combined: bool = True
) -> Dict[str, Dict[str, Any]]:
    """
    训练多个特征的 SOM 模型并生成 BMU
    
    Args:
        cfg: Hydra 配置对象
        feature_types: 需要训练 SOM 的特征类型列表
        train_combined: 是否训练 wind_flow 组合特征的 SOM
        
    Returns:
        每个特征 SOM 训练结果的字典
    """
    results = {}
    
    # 训练单一特征的 SOM 模型
    for feature_type in feature_types:
        logger.info(f"开始训练 {feature_type} 特征的 SOM 模型...")
        result = train_single_feature_som(cfg, feature_type)
        if result:
            results[feature_type] = result
        else:
            logger.error(f"{feature_type} 特征的 SOM 模型训练失败")
    
    # 训练组合特征的 SOM 模型
    if train_combined and "wind" in feature_types and "flow" in feature_types:
        logger.info("开始训练 wind_flow 组合特征的 SOM 模型...")
        combined_result = train_combined_feature_som(cfg, "wind_flow")
        if combined_result:
            results["wind_flow"] = combined_result
        else:
            logger.error("wind_flow 组合特征的 SOM 模型训练失败")
    
    return results

@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """主函数入口"""
    # 解析命令行参数，确定要训练哪些特征的 SOM
    feature_to_train = cfg.get("feature_to_train", "salinity")
    
    # 确保路径目录存在
    config = DrprConfig.from_hydra_config(cfg)
    os.makedirs(config.paths.som_models_dir, exist_ok=True)
    os.makedirs(config.paths.bmu_base_dir, exist_ok=True)
    
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