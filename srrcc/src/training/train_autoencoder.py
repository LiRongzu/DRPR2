# src/training/train_autoencoder.py
import numpy as np
import joblib
import os
import logging
from omegaconf import DictConfig
from typing import Dict, Any
import torch
from sklearn.preprocessing import StandardScaler
# 导入你的 PyTorch Autoencoder 实现
from src.dimensionality_reduction.autoencoder_pytorch import AutoencoderDimensionalityReduction
from src.utils.data_loader import load_processed_data
from src.utils.model_utils import get_device_from_config

logger = logging.getLogger(__name__)

def train_and_transform_ae(
    cfg: DictConfig,
    high_dim_data_paths: Dict[str, str], # 高维 train/val/test 数据路径
    model_save_path: str,
    scaler_save_path: str, # 用于高维数据的 Scaler
    transformed_save_dir: str, # 保存低维潜在变量的目录
    transformed_filename_pattern: str = "ae_latent_{split}.npy"
) -> Dict[str, Any]:
    """训练自动编码器，保存模型/scaler，转换数据，保存潜在变量。"""
    results = {}
    ae_cfg = cfg.model.dimensionality_reduction.autoencoder # 获取 AE 特定配置
    device = get_device_from_config(cfg)

    # 1. 加载高维训练数据
    train_path = high_dim_data_paths.get('train')
    # ... (与 PCA 类似的加载和展平逻辑) ...
    if not train_path or not os.path.exists(train_path): return results # 基本错误检查
    try:
        data_train_high_dim = np.load(train_path)
        if data_train_high_dim.ndim > 2:
            data_train_high_dim = data_train_high_dim.reshape(data_train_high_dim.shape[0], -1)
        logger.info(f"加载高维训练数据 (AE)，形状: {data_train_high_dim.shape}")
    except Exception as e: logger.error(f"AE 加载高维训练数据失败: {e}"); return results
    input_dim = data_train_high_dim.shape[1] # 获取 AE 的输入维度

    # 2. 缩放高维训练数据
    logger.info("缩放高维训练数据 (AE)...")
    scaler_ae_input = StandardScaler()
    data_train_scaled = scaler_ae_input.fit_transform(data_train_high_dim)
    logger.info("  AE 高维数据缩放完成。")

    # 3. 保存高维数据的 Scaler
    # ... (与 PCA 类似的保存逻辑) ...
    os.makedirs(os.path.dirname(scaler_save_path), exist_ok=True)
    try: joblib.dump(scaler_ae_input, scaler_save_path); results['scaler_path'] = scaler_save_path; logger.info(f"AE 输入 Scaler 保存至: {scaler_save_path}")
    except Exception as e: logger.error(f"保存 AE 输入 Scaler 失败: {e}"); return results

    # 4. 初始化并训练自动编码器
    logger.info("初始化并训练自动编码器模型...")
    ae_model = AutoencoderDimensionalityReduction(
        input_dim=input_dim,
        encoding_dim=ae_cfg.get("encoding_dim", 10),
        hidden_layers=list(ae_cfg.get("hidden_layers", [128, 64, 32])), # 确保是列表
        activation=ae_cfg.get("activation", 'relu'),
        dropout_rate=ae_cfg.get("dropout_rate", 0.1),
        learning_rate=cfg.training.optimizer.get("learning_rate", 0.001),
        epochs=cfg.training.get("epochs", 50), # AE 可能需要不同的 epochs?
        batch_size=ae_cfg.get("batch_size", 32),
        random_seed=cfg.training.random_seed,
        device=device,
        verbose=1 # 或从 cfg 读取
    )
    # Fit 需要缩放后的数据，验证集划分在内部处理
    ae_model.fit(data_train_scaled)
    logger.info("自动编码器训练完成。")

    # 5. 保存自动编码器模型 (使用它自己的保存方法)
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    try:
        ae_model.save(model_save_path) # 使用 AE 的保存方法
        logger.info(f"自动编码器模型已保存到: {model_save_path}")
        results['model_path'] = model_save_path
    except Exception as e:
        logger.error(f"保存自动编码器模型失败: {e}")
        return results

    # 6. 转换所有分割并保存潜在变量
    logger.info("转换所有数据分割到 AE 潜在空间...")
    results['low_dim_data_paths'] = {}
    os.makedirs(transformed_save_dir, exist_ok=True)
    for split in ["train", "val", "test"]:
        path = high_dim_data_paths.get(split)
        if path and os.path.exists(path):
            try:
                data_high_dim = np.load(path)
                if data_high_dim.ndim > 2:
                     data_high_dim = data_high_dim.reshape(data_high_dim.shape[0], -1)

                # 使用已拟合的 scaler 缩放
                data_scaled = scaler_ae_input.transform(data_high_dim)
                # 使用已拟合的 AE 模型的 transform 方法转换
                data_low_dim = ae_model.transform(data_scaled) # 使用 AE 的 transform
                logger.info(f"  转换 {split} 数据，低维形状: {data_low_dim.shape}")

                # 保存转换后的数据 (潜在变量)
                save_path = os.path.join(transformed_save_dir, transformed_filename_pattern.format(split=split))
                np.save(save_path, data_low_dim)
                logger.info(f"  AE 潜在变量 ({split}) 已保存到: {save_path}")
                results['low_dim_data_paths'][split] = save_path
            except Exception as e:
                logger.error(f"  转换或保存 {split} AE 潜在变量失败: {e}")
        else:
             logger.warning(f"  未找到 {split} 高维数据，跳过转换。")

    return results