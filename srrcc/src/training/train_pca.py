# src/training/train_pca.py
import numpy as np
import joblib # 用于保存 sklearn 模型
import os
import logging
from omegaconf import DictConfig
from typing import Dict, Any
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from src.utils.data_loader import load_processed_data # 使用集中的加载器

logger = logging.getLogger(__name__)

def train_and_transform_pca(
    cfg: DictConfig,
    high_dim_data_paths: Dict[str, str], # 高维 train/val/test 数据路径
    model_save_path: str,
    scaler_save_path: str, # 用于高维数据的 Scaler
    transformed_save_dir: str, # 保存低维 PCA 组件的目录
    transformed_filename_pattern: str = "pca_components_{split}.npy"
) -> Dict[str, Any]:
    """训练 PCA，保存模型/scaler，转换数据，保存组件。"""
    results = {}
    pca_cfg = cfg.model.dimensionality_reduction.pca # 获取 PCA 特定配置

    # 1. 加载高维训练数据 (拟合只需要训练集)
    train_path = high_dim_data_paths.get('train')
    if not train_path or not os.path.exists(train_path):
        logger.error("PCA 训练需要高维训练数据路径。")
        return results
    try:
        data_train_high_dim = np.load(train_path)
        # 如果需要，展平 (假设是空间数据)
        if data_train_high_dim.ndim > 2:
            original_shape = data_train_high_dim.shape
            data_train_high_dim = data_train_high_dim.reshape(original_shape[0], -1)
            logger.info(f"PCA 输入 (训练集) 从 {original_shape} 展平为 {data_train_high_dim.shape}")
        logger.info(f"加载高维训练数据，形状: {data_train_high_dim.shape}")
    except Exception as e:
        logger.error(f"加载高维训练数据失败: {train_path} - {e}")
        return results

    # 2. 缩放高维训练数据 (推荐按维度)
    logger.info("缩放高维训练数据 (PCA)...")
    # 默认使用 StandardScaler 进行按维度缩放
    scaler_pca_input = StandardScaler()
    data_train_scaled = scaler_pca_input.fit_transform(data_train_high_dim)
    logger.info("  高维数据缩放完成。")

    # 3. 保存高维数据的 Scaler
    os.makedirs(os.path.dirname(scaler_save_path), exist_ok=True)
    try:
         joblib.dump(scaler_pca_input, scaler_save_path) # 保存 scaler 对象
         logger.info(f"PCA 输入 Scaler 已保存到: {scaler_save_path}")
         results['scaler_path'] = scaler_save_path
    except Exception as e:
         logger.error(f"保存 PCA 输入 Scaler 失败: {e}")
         return results

    # 4. 训练 PCA
    n_components = pca_cfg.get("n_components", 50) # 从配置获取 n_components
    logger.info(f"训练 PCA 模型，n_components={n_components}...")
    pca_model = PCA(n_components=n_components, random_state=cfg.training.random_seed)
    pca_model.fit(data_train_scaled) # 在缩放数据上拟合
    logger.info("PCA 模型训练完成。")
    logger.info(f"  解释方差比: {pca_model.explained_variance_ratio_}")
    logger.info(f"  累计解释方差: {np.cumsum(pca_model.explained_variance_ratio_)}")

    # 5. 保存 PCA 模型
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    try:
        joblib.dump(pca_model, model_save_path)
        logger.info(f"PCA 模型已保存到: {model_save_path}")
        results['model_path'] = model_save_path
    except Exception as e:
        logger.error(f"保存 PCA 模型失败: {e}")
        return results

    # 6. 转换所有分割并保存
    logger.info("转换所有数据分割到 PCA 空间...")
    results['low_dim_data_paths'] = {}
    os.makedirs(transformed_save_dir, exist_ok=True)
    for split in ["train", "val", "test"]:
        path = high_dim_data_paths.get(split)
        if path and os.path.exists(path):
            try:
                data_high_dim = np.load(path)
                # 如果需要，展平
                if data_high_dim.ndim > 2:
                    data_high_dim = data_high_dim.reshape(data_high_dim.shape[0], -1)

                # 使用已拟合的 scaler 进行缩放
                data_scaled = scaler_pca_input.transform(data_high_dim)
                # 使用已拟合的 PCA 模型进行转换
                data_low_dim = pca_model.transform(data_scaled)
                logger.info(f"  转换 {split} 数据，低维形状: {data_low_dim.shape}")

                # 保存转换后的数据
                save_path = os.path.join(transformed_save_dir, transformed_filename_pattern.format(split=split))
                np.save(save_path, data_low_dim)
                logger.info(f"  PCA 组件 ({split}) 已保存到: {save_path}")
                results['low_dim_data_paths'][split] = save_path
            except Exception as e:
                logger.error(f"  转换或保存 {split} PCA 组件失败: {e}")
        else:
            logger.warning(f"  未找到 {split} 高维数据，跳过转换。")

    return results