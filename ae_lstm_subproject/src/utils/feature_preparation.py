# src/utils/feature_preparation.py
import numpy as np
import os
import logging
from omegaconf import DictConfig
from typing import Optional, Tuple, Dict

logger = logging.getLogger(__name__)

def load_feature_data(config: DictConfig, feature_type: str, target_field: str, split: str) -> Optional[np.ndarray]:
    """
    加载指定类型的特征数据，使用配置文件中定义的模式
    
    参数:
        config: 配置对象，包含paths和feature_loading信息
        feature_type: 特征类型 ("distance_vector" 或 "bmu_index")
        target_field: 目标字段 ("salinity", "wind" 等)
        split: 数据分割 ("train", "val", "test" 等)
    
    返回:
        加载的数据或None（如果加载失败）
    """
    if feature_type not in ["distance_vector", "bmu_index"]:
        logger.error(f"不支持的特征类型: {feature_type}")
        return None

    try:
        # 使用配置文件中定义的文件模式
        file_pattern = config.data.feature_loading[feature_type].file_pattern
        
        # 替换模式中的占位符
        file_path = file_pattern.format(field=target_field, split=split)
        
        # 判断路径是否存在
        if not os.path.exists(file_path):
            logger.error(f"特征文件未找到: {file_path}")
            return None
            
        # 加载数据
        data = np.load(file_path)
        logger.info(f"成功加载 {split} {target_field} {feature_type} 特征，形状: {data.shape}")
        return data
        
    except Exception as e:
        logger.error(f"加载特征数据失败: {e}")
        return None

def load_flow_data(config: DictConfig, split: str) -> Optional[np.ndarray]:
    """加载并分割径流数据"""
    flow_path = config.paths.flow # 假设配置中有径流文件路径
    if not os.path.exists(flow_path):
         logger.error(f"径流数据文件未找到: {flow_path}")
         return None
    try:
         flow_data_full = np.load(flow_path)
         if flow_data_full.ndim == 1:
             flow_data_full = flow_data_full.reshape(-1, 1)
         
         # 数据分割 (需要与训练时的分割一致)
         n_total = flow_data_full.shape[0]
         train_ratio = config.training.data_split.train_test_split
         val_ratio_of_train = config.training.data_split.validation_split
         
         train_size = int(train_ratio * n_total)
         val_size = int(val_ratio_of_train * train_size)
         train_only_size = train_size - val_size
         test_size = n_total - train_size

         if split == 'train':
             data = flow_data_full[:train_only_size]
         elif split == 'val':
              data = flow_data_full[train_only_size:train_size]
         elif split == 'test':
              data = flow_data_full[train_size:]
         elif split == 'full':
              data = flow_data_full
         else:
              logger.error(f"未知的数据分割类型: {split}")
              return None
              
         logger.info(f"成功加载并分割径流数据 ({split}), 形状: {data.shape}")
         return data

    except Exception as e:
         logger.error(f"加载或分割径流数据失败: {e}")
         return None


def prepare_prediction_features(config: DictConfig, scenario: str, feature_type: str, split: str = 'test') -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    根据场景加载、组合并返回用于预测的特征。

    参数:
        config: 配置对象 (DrprConfig)
        scenario: 预测场景 ("salinity_only", "full_conditional", "observable_conditional")
        feature_type: 低维特征类型 ("distance_vector" or "bmu_index")
        split: 数据分割 ('train', 'val', 'test', 'full')

    返回:
        features: 组合后的特征数组 (n_samples, n_combined_features)
        last_sequence_history: 用于启动预测的最后一个序列 (seq_len, n_combined_features) 或 None
    """
    logger.info(f"准备预测特征: 场景='{scenario}', 特征类型='{feature_type}', 数据分割='{split}'")
    
    seq_len = config.model.prediction.lstm.get("sequence_length", 10) # 或从 HMM 配置获取
    
    salinity_features = None
    wind_features = None
    flow_features = None
    
    # 1. 加载所需特征
    if scenario in ["salinity_only", "full_conditional"]:
        salinity_features = load_feature_data(config, feature_type, "salinity", split)
        if salinity_features is None: return None, None
        
    if scenario in ["full_conditional", "observable_conditional"]:
        wind_features = load_feature_data(config, feature_type, "wind", split)
        if wind_features is None: return None, None
        
        # 加载对应分割的径流数据
        flow_features = load_flow_data(config, split)
        if flow_features is None: return None, None
        # 确保流数据长度与风场特征匹配 (如果适用)
        if wind_features is not None and flow_features.shape[0] != wind_features.shape[0]:
             min_len = min(flow_features.shape[0], wind_features.shape[0])
             logger.warning(f"风场特征 ({wind_features.shape[0]}) 和径流 ({flow_features.shape[0]}) 长度不匹配，截断为 {min_len}")
             wind_features = wind_features[:min_len]
             flow_features = flow_features[:min_len]

    if scenario == "full_conditional":
         # 确保所有数据长度一致
         if salinity_features is None or wind_features is None or flow_features is None:
             logger.error("全条件模式下缺少必要的特征数据。")
             return None, None
         min_len = min(salinity_features.shape[0], wind_features.shape[0], flow_features.shape[0])
         if salinity_features.shape[0] != min_len or wind_features.shape[0] != min_len or flow_features.shape[0] != min_len:
             logger.warning("全条件模式下特征长度不匹配，截断为最小值。")
             salinity_features = salinity_features[:min_len]
             wind_features = wind_features[:min_len]
             flow_features = flow_features[:min_len]
         
    # 2. 组合特征
    features_to_combine = []
    if scenario == "salinity_only":
        if salinity_features is not None: features_to_combine.append(salinity_features)
    elif scenario == "full_conditional":
        if salinity_features is not None: features_to_combine.append(salinity_features)
        if wind_features is not None: features_to_combine.append(wind_features)
        if flow_features is not None: features_to_combine.append(flow_features)
    elif scenario == "observable_conditional":
        if wind_features is not None: features_to_combine.append(wind_features)
        if flow_features is not None: features_to_combine.append(flow_features)
    else:
        logger.error(f"未知的预测场景: {scenario}")
        return None, None

    if not features_to_combine:
        logger.error("没有可用于组合的特征。")
        return None, None
        
    # 检查并处理BMU索引（如果是BMU，维度可能不同）
    if feature_type == "bmu_index":
         # BMU 索引是 (n_samples, 2)，流数据是 (n_samples, n_flow_features)
         # 拼接时需要注意，可能需要将 BMU 转换为其他表示（如 one-hot）或分别处理
         logger.warning("BMU索引与径流的特征组合逻辑需要仔细设计（例如 embedding 或 one-hot）。当前实现仅简单拼接。")
         # 确保所有数组至少是2D
         features_to_combine = [f.reshape(f.shape[0], -1) if f.ndim > 1 else f.reshape(-1, 1) for f in features_to_combine]

    # 确保所有待拼接数组至少是2D
    features_2d = []
    for f in features_to_combine:
         if f.ndim == 1:
             features_2d.append(f.reshape(-1, 1))
         elif f.ndim > 1:
             features_2d.append(f.reshape(f.shape[0], -1)) # 展平除时间外的维度
         else:
              logger.error(f"特征数组维度不正确: {f.ndim}")
              return None, None
              
    try:
         combined_features = np.concatenate(features_2d, axis=1)
         logger.info(f"组合后的特征形状: {combined_features.shape}")
    except ValueError as e:
         logger.error(f"特征拼接失败: {e}. 各特征形状: {[f.shape for f in features_2d]}")
         return None, None

    # 3. 获取最后一个序列作为历史启动预测 (仅用于迭代预测，如 LSTM)
    last_sequence_history = None
    if combined_features.shape[0] >= seq_len:
        last_sequence_history = combined_features[-seq_len:]
        logger.info(f"提取的最后一个序列历史形状: {last_sequence_history.shape}")
    else:
        logger.warning(f"数据长度 ({combined_features.shape[0]}) 不足序列长度 ({seq_len})，无法提取历史序列。")

    return combined_features, last_sequence_history