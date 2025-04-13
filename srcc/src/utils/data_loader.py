# src/utils/data_loader.py
import os
import numpy as np
import joblib
import logging
from omegaconf import DictConfig

logger = logging.getLogger(__name__)

# 内部辅助函数，用于根据配置安全地获取路径
def _get_path(cfg: DictConfig, *path_keys, filename: str = None) -> str:
    """根据 cfg.paths 中的键或直接使用 filename 构建完整路径"""
    base_dir = cfg.paths # 假设所有路径配置都在 cfg.paths 下
    for key in path_keys:
        if hasattr(base_dir, key):
            base_dir = getattr(base_dir, key)
        else:
            logger.error(f"在 cfg.paths 中未找到路径键: {'.'.join(path_keys)}")
            raise AttributeError(f"Config path key not found: {'.'.join(path_keys)}")

    if not isinstance(base_dir, str):
         logger.error(f"Config path for {'.'.join(path_keys)} is not a string: {base_dir}")
         raise TypeError(f"Config path is not a string: {'.'.join(path_keys)}")

    if filename:
        return os.path.join(base_dir, filename)
    return base_dir # 如果没有 filename，返回目录路径

# --- 加载原始数据 ---
def load_raw_data(cfg: DictConfig, feature_name: str) -> np.ndarray:
    """加载原始数据 (例如 'salt', 'wind', 'flow')"""
    try:
        # 假设原始数据文件名在 cfg.data.<feature_name>.data 中定义
        if feature_name == 'salinity':
            raw_path = cfg.data.salt.data # 假设它已经解析为完整路径
        elif feature_name == 'wind':
            raw_path = cfg.data.wind.data
        elif feature_name == 'flow':
            raw_path = cfg.data.flow
        else:
             # 尝试通用模式
             raw_path = _get_path(cfg, 'raw_data_dir', filename=f"{feature_name}.npy") # 假设命名约定

        logger.info(f"尝试加载原始 {feature_name} 数据: {raw_path}")
        if not os.path.exists(raw_path):
            logger.error(f"原始数据文件未找到: {raw_path}")
            raise FileNotFoundError(f"原始数据文件未找到: {raw_path}")
        data = np.load(raw_path)
        logger.info(f"成功加载原始 {feature_name} 数据，形状: {data.shape}")
        return data
    except AttributeError as e:
         logger.error(f"加载原始 {feature_name} 数据时配置错误: {e}")
         raise
    except Exception as e:
        logger.error(f"加载原始 {feature_name} 数据失败: {e}")
        raise # 重新抛出异常，让调用者处理

def load_coordinates(cfg: DictConfig, feature_name: str) -> np.ndarray:
    """加载坐标数据 (例如 'salt_lonlat', 'wind_lonlat')"""
    try:
        if feature_name == 'salt':
            coord_path = cfg.data.salt.lonlat # 假设它已经解析为完整路径
        elif feature_name == 'wind':
             coord_path = cfg.data.wind.lonlat
        # ... 可以添加更多坐标类型
        else:
             # 尝试通用模式
             coord_path = _get_path(cfg, 'raw_data_dir', filename=f"{feature_name}_lonlat.npy") # 假设命名约定

        logger.info(f"尝试加载 {feature_name} 坐标数据: {coord_path}")
        if not os.path.exists(coord_path):
            logger.warning(f"坐标数据文件未找到: {coord_path}")
            return None # 坐标可能是可选的
        data = np.load(coord_path)
        logger.info(f"成功加载 {feature_name} 坐标数据，形状: {data.shape}")
        return data
    except AttributeError as e:
         logger.error(f"加载 {feature_name} 坐标数据时配置错误: {e}")
         return None # 配置错误也返回 None
    except Exception as e:
        logger.error(f"加载 {feature_name} 坐标数据失败: {e}")
        return None # 加载失败返回 None


def load_mask(cfg: DictConfig, feature_name: str = "salinity") -> np.ndarray:
     """加载 Mask 数据"""
     # 假设 mask 文件路径在 cfg.data.<feature_name>.mask 或 cfg.data.mesh.mask
     mask_path = None
     try:
         # 优先尝试特定特征的 mask
         if hasattr(cfg.data, feature_name) and hasattr(cfg.data[feature_name], 'mask'):
             mask_path = cfg.data[feature_name].mask
         # 否则尝试通用的 mesh mask
         elif hasattr(cfg.data, 'mesh') and hasattr(cfg.data.mesh, 'mask'):
              mask_path = cfg.data.mesh.mask
         # 否则尝试默认命名
         else:
              mask_path = _get_path(cfg, 'raw_data_dir', filename="mask.npy")

         logger.info(f"尝试加载 Mask 数据: {mask_path}")
         if not os.path.exists(mask_path):
              logger.warning(f"Mask 文件未找到: {mask_path}")
              return None # Mask 可能是可选的
         mask = np.load(mask_path).astype(bool) # 确保是布尔类型

         logger.info(f"成功加载 Mask 数据，形状: {mask.shape}")
         return mask
     except AttributeError as e:
          logger.error(f"加载 Mask 数据时配置错误: {e}")
          return None
     except Exception as e:
          logger.error(f"加载 Mask 数据失败: {e}")
          return None

# --- 加载处理后数据 ---
def load_processed_data(cfg: DictConfig, feature_name: str, split: str) -> np.ndarray:
    """加载处理后的数据 (例如 'train_salinity_processed.npy')"""
    filename = f"{split}_{feature_name}_processed.npy"
    try:
        processed_path = _get_path(cfg, 'processed_data_dir', filename=filename)
        logger.info(f"尝试加载处理后的 {split} {feature_name} 数据: {processed_path}")
        if not os.path.exists(processed_path):
             logger.error(f"处理后数据文件未找到: {processed_path}")
             raise FileNotFoundError(f"处理后数据文件未找到: {processed_path}")
        data = np.load(processed_path)
        logger.info(f"成功加载处理后的 {split} {feature_name} 数据，形状: {data.shape}")
        return data
    except Exception as e:
        logger.error(f"加载处理后的 {split} {feature_name} 数据失败: {e}")
        raise

# --- 加载 BMU 位置数据 ---
def load_bmu_positions(cfg: DictConfig, feature_name: str, split: str) -> np.ndarray:
    """加载 BMU 位置数据 (例如 'bmu_positions_salinity_train.npy')"""
    filename = f"bmu_positions_{feature_name}_{split}.npy"
    try:
        # BMU 位置文件通常保存在 cfg.paths.bmu_base_dir 下
        bmu_path = _get_path(cfg, 'bmu_base_dir', filename=filename)
        logger.info(f"尝试加载 {split} {feature_name} BMU 位置: {bmu_path}")
        if not os.path.exists(bmu_path):
             logger.error(f"BMU 位置文件未找到: {bmu_path}")
             raise FileNotFoundError(f"BMU 位置文件未找到: {bmu_path}")
        data = np.load(bmu_path)
        logger.info(f"成功加载 {split} {feature_name} BMU 位置，形状: {data.shape}")
        return data
    except Exception as e:
        logger.error(f"加载 {split} {feature_name} BMU 位置失败: {e}")
        raise

# --- 加载距离向量数据 ---
def load_distance_vectors(cfg: DictConfig, feature_name: str, split: str) -> np.ndarray:
     """加载距离向量数据"""
     # 使用 cfg.data.feature_loading 中定义的模式
     try:
          pattern = cfg.data.feature_loading.distance_vector.file_pattern
          # 注意：这里的 pattern 可能需要 cfg.paths.distance_vectors_dir，确保它已正确解析
          dv_path = pattern.format(field=feature_name, split=split)
          logger.info(f"尝试加载 {split} {feature_name} 距离向量: {dv_path}")
          if not os.path.exists(dv_path):
               logger.error(f"距离向量文件未找到: {dv_path}")
               raise FileNotFoundError(f"距离向量文件未找到: {dv_path}")
          data = np.load(dv_path)
          logger.info(f"成功加载 {split} {feature_name} 距离向量，形状: {data.shape}")
          return data
     except AttributeError as e:
          logger.error(f"加载距离向量时配置错误 (检查 cfg.data.feature_loading.distance_vector.file_pattern): {e}")
          raise
     except Exception as e:
          logger.error(f"加载 {split} {feature_name} 距离向量失败: {e}")
          raise

# --- 加载 Scaler ---
def load_scaler(cfg: DictConfig, feature_name: str) -> dict:
    """加载 Scaler 文件 (通常保存在 processed_data_dir)"""
    # Scaler 文件名可能不同，优先从配置读取，否则使用约定
    scaler_filename = f"{feature_name}_scaler.npy" # 或 .pkl
    # LSTM 可能有自己的 scaler，例如 lstm_scaler.pkl
    if feature_name == 'lstm':
         scaler_filename = "lstm_scaler.pkl" # 从 LSTM 配置读取更佳

    scaler_path = _get_path(cfg, 'processed_data_dir', filename=scaler_filename)
    logger.info(f"尝试加载 {feature_name} Scaler: {scaler_path}")

    if not os.path.exists(scaler_path):
         logger.error(f"Scaler 文件未找到: {scaler_path}")
         raise FileNotFoundError(f"Scaler 文件未找到: {scaler_path}")
    try:
         if scaler_path.endswith('.npy'):
              # NPY scaler 通常是包含 'mean' 和 'std' 的字典
              scaler_data = np.load(scaler_path, allow_pickle=True).item()
              if not isinstance(scaler_data, dict) or 'mean' not in scaler_data or 'std' not in scaler_data:
                   raise ValueError("NPY Scaler 文件格式错误，应为包含 'mean' 和 'std' 的字典")
         elif scaler_path.endswith('.pkl'):
              # PKL scaler 通常是 scikit-learn 的 scaler 对象
              with open(scaler_path, 'rb') as f:
                   scaler_data = joblib.load(f) # 或者 pickle.load(f)
         else:
              raise ValueError(f"未知的 Scaler 文件格式: {scaler_path}")

         logger.info(f"成功加载 {feature_name} Scaler")
         return scaler_data
    except Exception as e:
         logger.error(f"加载 {feature_name} Scaler 失败: {e}")
         raise

# --- 加载分割索引 ---
def load_split_indices(cfg: DictConfig) -> dict:
     """加载 train/val/test 的原始索引"""
     split_indices_path = _get_path(cfg, 'processed_data_dir', filename="split_indices.npz")
     logger.info(f"尝试加载分割索引: {split_indices_path}")
     if not os.path.exists(split_indices_path):
          logger.error(f"分割索引文件未找到: {split_indices_path}")
          raise FileNotFoundError(f"分割索引文件未找到: {split_indices_path}")
     try:
          split_data = np.load(split_indices_path)
          indices = {
               'train': split_data.get('train_indices'),
               'val': split_data.get('val_indices'),
               'test': split_data.get('test_indices')
          }
          logger.info(f"成功加载分割索引: Train({len(indices['train']) if indices['train'] is not None else 'N/A'}), Val({len(indices['val']) if indices['val'] is not None else 'N/A'}), Test({len(indices['test']) if indices['test'] is not None else 'N/A'})")
          return indices
     except Exception as e:
          logger.error(f"加载分割索引失败: {e}")
          raise