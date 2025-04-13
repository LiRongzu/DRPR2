# src/utils/bmu_utils.py
import numpy as np
import torch
import joblib
import logging
from typing import Dict, Tuple, Callable, Optional, List 

try:
    from src.dimensionality_reduction.som_pytorch import SOMTorch
except ImportError:
    logging.error("无法导入 SOMTorch 类，请确保路径设置正确。")
    # 可以定义一个占位符类或引发错误，取决于你的错误处理策略
    class SOMTorch: pass # 占位符

logger = logging.getLogger(__name__)

# --- Rank Map Creation (从 train_hmm.py 移动) ---
def calculate_l2_norm(prototype_vector: np.ndarray) -> float:
    """计算原型向量的 L2 范数"""
    return np.linalg.norm(prototype_vector)

def create_ranking_map(som_model, # 预期是 SOMTorch 实例
                      property_func: Callable = calculate_l2_norm) -> Dict[int, int]:
    """
    根据属性函数创建从 SOM 节点线性索引到等级的映射。

    Args:
        som_model: 训练好的 SOM 模型 (需要 .weights 和 .map_size 属性)。
        property_func: 用于计算节点属性值的函数 (输入原型向量，输出标量值)。

    Returns:
        rank_map: 字典 {linear_index: rank}。
    """
    if not hasattr(som_model, 'weights') or not hasattr(som_model, 'map_size'):
         logger.error("输入的 SOM 模型缺少 'weights' 或 'map_size' 属性。")
         raise AttributeError("SOM 模型对象不符合预期。")

    map_size = tuple(som_model.map_size)
    map_height, map_width = map_size[0], map_size[1]
    num_nodes = map_height * map_width

    logger.info(f"为 {num_nodes} 个节点计算属性值 (使用 {property_func.__name__})...")

    # 确保权重在 CPU 上并是 NumPy 数组
    try:
         with torch.no_grad():
              prototypes_flat = som_model.weights.reshape(num_nodes, -1).cpu().numpy()
    except Exception as e:
         logger.error(f"访问或处理 SOM 权重时出错: {e}")
         raise

    # 计算属性值
    try:
         property_values = np.array([property_func(p) for p in prototypes_flat])
    except Exception as e:
         logger.error(f"使用 property_func ({property_func.__name__}) 计算属性值时出错: {e}")
         raise

    logger.info(f"属性值计算完成，范围: [{np.min(property_values):.4f}, {np.max(property_values):.4f}]")

    # 处理 NaN/Inf (保持 train_hmm.py 中的逻辑)
    if np.any(~np.isfinite(property_values)):
         num_non_finite = np.sum(~np.isfinite(property_values))
         logger.warning(f"在属性值计算中发现 {num_non_finite} 个非有限值 (NaN/Inf)。将尝试用中位数替换。")
         median_val = np.nanmedian(property_values)
         if np.isfinite(median_val):
              property_values[~np.isfinite(property_values)] = median_val
              logger.info(f"非有限值已替换为中位数: {median_val:.4f}")
         else:
              property_values[~np.isfinite(property_values)] = 0.0
              logger.warning("中位数也是非有限值，非有限属性值已替换为 0.0")

    # 获取排序后的索引
    sorted_indices = np.argsort(property_values)

    # 创建映射: {linear_index: rank}
    rank_map = {int(original_idx): rank for rank, original_idx in enumerate(sorted_indices)}
    logger.info(f"为 {len(rank_map)} 个节点创建了等级映射。")
    return rank_map

# --- Index Conversion Functions ---
def convert_grid_to_linear(grid_indices: np.ndarray, map_width: int) -> np.ndarray:
    """将 2D 网格索引 (row, col) 转换为 1D 线性索引"""
    if grid_indices.ndim != 2 or grid_indices.shape[1] != 2:
         raise ValueError(f"输入 grid_indices 应为 (N, 2) 形状, 得到 {grid_indices.shape}")
    # 假设 grid_indices 是 [row, col]
    # 线性索引 = row * map_width + col
    return (grid_indices[:, 0] * map_width + grid_indices[:, 1]).astype(int)

def convert_linear_to_grid(linear_indices: np.ndarray, map_width: int) -> np.ndarray:
    """将 1D 线性索引转换为 2D 网格索引 (row, col)"""
    if linear_indices.ndim != 1:
         raise ValueError(f"输入 linear_indices 应为 1D 数组, 得到 {linear_indices.ndim}D")
    rows = linear_indices // map_width # 行索引
    cols = linear_indices % map_width  # 列索引
    return np.stack([rows, cols], axis=1).astype(int)

def convert_rank_to_linear(ranks: np.ndarray, rank_to_linear_map: Dict[int, int]) -> np.ndarray:
    """将等级索引转换为线性索引"""
    if ranks.ndim != 1:
         raise ValueError(f"输入 ranks 应为 1D 数组, 得到 {ranks.ndim}D")
    try:
        linear_indices = np.array([rank_to_linear_map[rank] for rank in ranks])
        return linear_indices
    except KeyError as e:
        missing_ranks = set(rank for rank in ranks if rank not in rank_to_linear_map)
        logger.error(f"等级反向映射中存在 KeyError。缺失的等级索引示例: {list(missing_ranks)[:10]}. 总缺失数: {len(missing_ranks)}")
        raise # 重新抛出异常

def convert_linear_to_rank(linear_indices: np.ndarray, rank_map: Dict[int, int]) -> np.ndarray:
     """将线性索引转换为等级索引"""
     if linear_indices.ndim != 1:
          raise ValueError(f"输入 linear_indices 应为 1D 数组, 得到 {linear_indices.ndim}D")
     try:
          ranks = np.array([rank_map[idx] for idx in linear_indices])
          return ranks
     except KeyError as e:
          missing_indices = set(idx for idx in linear_indices if idx not in rank_map)
          logger.error(f"等级映射中存在 KeyError。缺失的线性索引示例: {list(missing_indices)[:10]}. 总缺失数: {len(missing_indices)}")
          raise # 重新抛出异常

# --- Utility to get maps from HMM params ---
def get_rank_maps_from_hmm_params(hmm_params_path: str) -> Tuple[Optional[Dict[int, int]], Optional[Dict[int, int]]]:
    """
    从 HMM 参数文件中加载 rank_map 和 rank_to_linear_map。

    Args:
        hmm_params_path: HMM 参数文件路径 (.pkl)。

    Returns:
        Tuple[rank_map, rank_to_linear_map]:
            - rank_map: {linear_index: rank} 或 None
            - rank_to_linear_map: {rank: linear_index} 或 None
    """
    rank_map = None
    rank_to_linear_map = None
    if not os.path.exists(hmm_params_path):
        logger.warning(f"HMM 参数文件未找到: {hmm_params_path}。无法加载 Rank Maps。")
        return rank_map, rank_to_linear_map
    try:
        hmm_params_data = joblib.load(hmm_params_path)
        if 'state_rank_map' in hmm_params_data:
            rank_map = hmm_params_data['state_rank_map'] # {linear: rank}
            # 创建反向映射
            rank_to_linear_map = {rank: linear for linear, rank in rank_map.items()}
            logger.info(f"从 HMM 参数加载了 rank_map 和 rank_to_linear_map (大小: {len(rank_map)})")
        elif 'state_linear_map' in hmm_params_data: # 兼容只保存了反向映射的情况
             rank_to_linear_map = hmm_params_data['state_linear_map'] # {rank: linear}
             # 创建正向映射
             rank_map = {linear: rank for rank, linear in rank_to_linear_map.items()}
             logger.info(f"从 HMM 参数加载了 rank_to_linear_map 并创建了 rank_map (大小: {len(rank_map)})")
        else:
             logger.warning(f"在 HMM 参数文件 {hmm_params_path} 中未找到 'state_rank_map' 或 'state_linear_map'。")

        # (可选) 检查地图大小是否一致
        map_size_from_params = hmm_params_data.get('map_size', None)
        if map_size_from_params:
             num_nodes_params = map_size_from_params[0] * map_size_from_params[1]
             if rank_map and len(rank_map) != num_nodes_params:
                  logger.warning(f"HMM 参数中的 Rank Map 大小 ({len(rank_map)}) 与地图大小计算的节点数 ({num_nodes_params}) 不符!")

        return rank_map, rank_to_linear_map

    except Exception as e:
        logger.error(f"加载或处理 HMM 参数文件失败: {e}")
        return None, None # 返回 None 表示失败

# --- Combined Conversion Utilities ---
import os # 为了支持 os.path.exists

def convert_bmu_to_ranks(bmu_raw: np.ndarray, rank_map: Dict[int, int], map_width: int) -> Optional[np.ndarray]:
    """
    将 2D BMU 坐标转换为等级索引序列。
    这是一个便捷函数，结合了 convert_grid_to_linear 和 convert_linear_to_rank。

    Args:
        bmu_raw: 形状为 (N, 2) 的 BMU 坐标数组，格式为 [row, col]
        rank_map: 从线性索引到等级的映射字典 {linear_index: rank}
        map_width: SOM 模型的网格宽度

    Returns:
        等级索引数组，如果转换失败则返回 None
    """
    if bmu_raw is None:
        return None
    
    if bmu_raw.ndim != 2 or bmu_raw.shape[1] != 2:
        logger.error(f"期望 2D BMU 坐标数组形状为 (N, 2)，但得到 {bmu_raw.shape}")
        return None
    
    try:
        # 1. 网格索引转线性索引
        linear_indices = convert_grid_to_linear(bmu_raw, map_width)
        # 2. 线性索引转等级
        ranks = convert_linear_to_rank(linear_indices, rank_map)
        return ranks
    except KeyError:
        # convert_linear_to_rank 已经记录了错误
        return None
    except Exception as e:
        logger.error(f"将 BMU 坐标转换为等级时出错: {e}")
        return None
    
def calculate_som_cluster_averages(som_model: SOMTorch, train_data: np.ndarray) -> Optional[np.ndarray]:
    """
    计算 SOM 每个节点对应的训练样本的平均值 (Cluster Averages)。

    Args:
        som_model: 训练好的 SOMTorch 模型实例。
        train_data: 用于训练该 SOM 模型的高维数据 (Numpy array, shape: [n_samples, n_features])。
                    应该是模型 fit 时使用的数据。

    Returns:
        Numpy array (shape: [n_nodes, n_features]) 包含每个节点的平均向量，
        如果计算失败则返回 None。
        非活动节点将使用节点自身的权重向量填充。
    """
    logger = logging.getLogger(__name__) # 使用模块级 logger
    logger.info("开始计算 SOM Cluster Averages...")

    if som_model is None or not hasattr(som_model, 'weights') or not hasattr(som_model, 'map_size'):
        logger.error("无效的 SOM 模型对象。")
        return None
    if train_data is None or train_data.size == 0:
        logger.error("训练数据为空，无法计算 Cluster Averages。")
        return None

    try:
        n_samples_train, n_features = train_data.shape
        # 从模型获取必要信息
        map_size = som_model.map_size
        map_height, map_width = map_size[0], map_size[1]
        n_nodes = map_height * map_width
        if som_model.input_dim != n_features:
             logger.error(f"训练数据特征维度 ({n_features}) 与 SOM 模型输入维度 ({som_model.input_dim}) 不匹配。")
             return None

        # 1. 计算训练数据的 BMU (获取网格坐标)
        # som_model.transform 应该能处理 numpy 输入并返回 numpy grid indices (N, 2)
        logger.info("  计算训练数据的 BMU...")
        train_bmu_grid_indices = som_model.transform(train_data)
        if train_bmu_grid_indices is None or train_bmu_grid_indices.shape[0] != n_samples_train:
             logger.error("  计算训练数据 BMU 失败或返回形状不正确。")
             return None

        # 2. 将网格坐标转换为线性索引
        logger.info("  转换 BMU 网格坐标为线性索引...")
        # 确保 convert_grid_to_linear 在此文件中或已正确导入
        train_bmu_linear_indices = convert_grid_to_linear(train_bmu_grid_indices, map_width)

        # 3. 初始化存储和计算平均值
        logger.info("  分配训练样本并计算节点平均值...")
        node_samples: List[List[np.ndarray]] = [[] for _ in range(n_nodes)]
        cluster_averages = np.full((n_nodes, n_features), np.nan, dtype=np.float32) # 初始化为 NaN

        # 分配样本
        for i in range(n_samples_train):
            node_idx = train_bmu_linear_indices[i]
            # 添加检查确保 node_idx 在有效范围内
            if 0 <= node_idx < n_nodes:
                node_samples[node_idx].append(train_data[i])
            else:
                logger.warning(f"  样本 {i} 的 BMU 线性索引 {node_idx} 超出范围 [0, {n_nodes-1}]，已忽略。")


        # 计算平均值，处理非活动节点
        inactive_nodes = 0
        # 确保权重在 CPU 上并是 NumPy 数组
        try:
            weights_flat = som_model.weights.reshape(n_nodes, n_features).detach().cpu().numpy()
        except Exception as e:
            logger.error(f"  无法获取或处理 SOM 权重: {e}")
            return None

        for node_idx in range(n_nodes):
            samples = node_samples[node_idx]
            if len(samples) > 0:
                # 计算有效节点的平均值
                cluster_averages[node_idx] = np.mean(np.stack(samples), axis=0)
            else:
                # 处理非活动节点：使用节点权重填充
                inactive_nodes += 1
                cluster_averages[node_idx] = weights_flat[node_idx]

        if inactive_nodes > 0:
             logger.warning(f"  发现 {inactive_nodes} 个非活动节点，已使用节点权重填充平均值。")

        # 最终检查 NaN (理论上如果用权重填充了就不会有)
        if np.isnan(cluster_averages).any():
             num_nan = np.sum(np.isnan(cluster_averages))
             logger.warning(f"  计算出的 Cluster Averages 中仍包含 {num_nan} 个 NaN 值！可能需要检查数据或填充逻辑。")

        logger.info(f"Cluster Averages 计算完成，形状: {cluster_averages.shape}")
        return cluster_averages

    except Exception as e:
        logger.error(f"计算 Cluster Averages 时发生意外错误: {e}", exc_info=True)
        return None