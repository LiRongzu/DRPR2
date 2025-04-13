# src/training/train_manifold_model.py

import os
import sys
import numpy as np
import logging
import time
import hydra
from omegaconf import DictConfig, OmegaConf
from typing import Optional, Tuple
import joblib

# Project specific imports (adjust paths if necessary)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils.hydra_config import DrprConfig
from src.utils.logger import setup_logger
# Import Manifold Reconstruction model and factory function
from src.dimensionality_reduction.manifold_reconstruction import (
    ManifoldReconstruction,
    create_manifold_reconstruction_model
)
# 导入集中化的数据加载函数
from src.utils.data_loader import load_processed_data, load_distance_vectors

logger = logging.getLogger(__name__)

# 移除旧的 load_high_dim_data 函数，使用集中化的数据加载函数

@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    主函数，用于训练流形重建模型。
    """
    start_run_time = time.time()
    config = DrprConfig.from_hydra_config(cfg)
    setup_logger(log_level=config.training.logging.get("level", "INFO"))

    logger.info("Starting Manifold Reconstruction Model Training script...")
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    # --- Parameters ---
    manifold_cfg = config.model.path1_reconstruction # 流形重建参数在此
    target_field = config.model.dim_reduction.som.get("target_fields", ["salinity"])[0]
    input_feature_type = manifold_cfg.get("manifold_input_feature", "distance_vector") # 读取使用的低维特征类型
    model_save_dir = os.path.join(config.project.exp_dir, "models", "manifold") # 保存到 exp/models/manifold
    os.makedirs(model_save_dir, exist_ok=True)

    logger.info(f"Target field: {target_field}")
    logger.info(f"Manifold training input feature type: {input_feature_type}")

    # --- 1. Load Training Data ---
    # 加载训练集的低维特征
    logger.info(f"Loading training low-dimensional features ({input_feature_type})...")
    # 根据特征类型确定加载方法
    if input_feature_type.lower() == "distance_vector":
        features_low_dim_train = load_distance_vectors(cfg, target_field, "train")
    else:
        # 默认加载处理后的特征
        features_low_dim_train = load_processed_data(cfg, target_field, "train")
    
    if features_low_dim_train is None:
        logger.error("Failed to load training low-dimensional features. Exiting.")
        return

    # 加载训练集的高维数据
    logger.info("Loading training high-dimensional data...")
    features_high_dim_train = load_processed_data(cfg, target_field, "train")
    if features_high_dim_train is None:
        logger.error("Failed to load training high-dimensional data. Exiting.")
        return

    # --- 2. Validate Data Shapes ---
    if features_low_dim_train.shape[0] != features_high_dim_train.shape[0]:
        min_len = min(features_low_dim_train.shape[0], features_high_dim_train.shape[0])
        logger.warning(f"Low-dim ({features_low_dim_train.shape[0]}) and High-dim ({features_high_dim_train.shape[0]}) training data lengths mismatch! Truncating to {min_len}.")
        features_low_dim_train = features_low_dim_train[:min_len]
        features_high_dim_train = features_high_dim_train[:min_len]
        if min_len == 0:
             logger.error("有效训练样本数量为 0。退出。")
             return

    logger.info(f"Using {features_low_dim_train.shape[0]} samples for training.")
    logger.info(f"Low-dim training data shape: {features_low_dim_train.shape}")
    logger.info(f"High-dim training data shape: {features_high_dim_train.shape}")

    # --- 3. Instantiate Manifold Reconstruction Model ---
    logger.info("Instantiating Manifold Reconstruction model...")
    try:
        # 使用工厂函数创建模型实例
        manifold_model = create_manifold_reconstruction_model(
            method=manifold_cfg.get("method", "local_pca_knn"),
            neighborhood_size=manifold_cfg.neighborhood_size,
            pca_components=manifold_cfg.pca_components
        )
        logger.info(f"Manifold model created with method: {manifold_cfg.get('method', 'local_pca_knn')}, k={manifold_cfg.neighborhood_size}, pca_comp={manifold_cfg.pca_components}")
    except Exception as e:
        logger.error(f"Failed to instantiate ManifoldReconstruction model: {e}")
        return

    # --- 4. Train Model ---
    logger.info("Starting Manifold Reconstruction model training...")
    training_start_time = time.time()
    try:
        manifold_model.fit(features_low_dim_train, features_high_dim_train)
        training_time = time.time() - training_start_time
        logger.info(f"Manifold model training completed in {training_time:.2f} seconds.")
    except Exception as e:
        logger.error(f"Manifold model training failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return

    # --- 5. Save Model ---
    model_filename = f"manifold_model_{target_field}_{input_feature_type}"
    model_filename += f"_k{manifold_cfg.neighborhood_size}_pca{manifold_cfg.pca_components}.pkl"
    model_save_path = os.path.join(model_save_dir, model_filename)

    try:
        joblib.dump(manifold_model, model_save_path)
        logger.info(f"Trained Manifold Reconstruction model saved successfully to: {model_save_path}")
    except Exception as e:
        logger.error(f"Failed to save Manifold model: {e}")

    total_run_time = time.time() - start_run_time
    logger.info(f"Manifold Model Training script finished in {total_run_time:.2f} seconds.")

if __name__ == "__main__":
    # Ensure src directory is in path for relative imports if needed
    script_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.join(script_dir, '..')
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    main()