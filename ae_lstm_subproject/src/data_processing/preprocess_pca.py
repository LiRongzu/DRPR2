import os
import sys

# Find the project root directory (assuming it's the parent of 'src')
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir)) # Go up two levels

# Add project root to the Python path
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# coding: utf-8
# filename: src/data_processing/preprocess_pca.py
"""
执行主成分分析 (PCA) 的预处理脚本。

此脚本加载先前已缩放 (scaled) 的高维数据，
在训练数据上拟合 PCA 模型，然后转换所有数据分割（训练、验证、测试）
并将 PCA 模型和转换后的低维成分保存到磁盘。
"""

import logging
from typing import Dict, Optional

import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import joblib
from sklearn.decomposition import PCA


# 使用你的数据加载器
from src.utils.data_loader import load_processed_data


# 配置日志记录器
logger = logging.getLogger(__name__)
# 可以根据需要添加更详细的日志配置，例如设置级别和处理器
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def run_pca_preprocessing_for_feature(cfg: DictConfig, feature: str) -> bool:
    """
    为单个特征执行 PCA 预处理。

    Args:
        cfg: Hydra 配置对象。
        feature: 要处理的特征名称 (例如, "wind", "salinity")。

    Returns:
        如果处理成功则返回 True，否则返回 False。
    """
    logger.info(f"--- 开始处理特征: {feature} 的 PCA ---")

    # --- 1. 获取配置和路径 ---
    try:
        # 获取 PCA 输出路径
        pca_models_dir = cfg.paths.pca_models_dir       # PCA 模型保存目录
        pca_components_dir = cfg.paths.pca_components_dir  # PCA 成分保存目录

        # 获取 PCA 参数
        # 尝试获取特定特征的 n_components，如果不存在则使用通用值
        n_components = 15
        if n_components is None:
            logger.error(f"未在配置中找到特征 '{feature}' 或通用的 n_components。")
            return False
        logger.info(f"  将使用 n_components = {n_components}")

        # 确保输出目录存在
        os.makedirs(pca_models_dir, exist_ok=True)
        os.makedirs(pca_components_dir, exist_ok=True)

    except Exception as e:
        logger.error(f"从配置中获取路径或参数时出错: {e}", exc_info=True)
        return False

    # --- 2. 加载已缩放的高维数据 ---
    scaled_data: Dict[str, Optional[np.ndarray]] = {}
    splits_to_process = ["train", "val", "test"]
    data_loaded = False
    for split in splits_to_process:
        try:
            # 使用你的加载函数加载 _processed.npy 文件（假设它们是缩放后的）
            data = load_processed_data(cfg, feature, split)
            if data is not None:
                # 确保数据是 2D (N_samples, N_features)
                if data.ndim > 2:
                    original_shape = data.shape
                    # 将除了第一个维度（样本数）之外的所有维度展平
                    data = data.reshape(data.shape[0], -1)
                    logger.info(f"  将 {split} {feature} 数据从 {original_shape} 重塑为 {data.shape}")
                scaled_data[split] = data
                data_loaded = True
                logger.info(f"  已加载 {split} {feature} 的已缩放数据，形状: {data.shape}")
            else:
                logger.warning(f"  加载函数为 {split} {feature} 返回 None。")
                scaled_data[split] = None # 标记为 None

        except FileNotFoundError:
            logger.warning(f"  未找到 {split} {feature} 的已处理（已缩放）数据文件。")
            scaled_data[split] = None # 标记为 None
        except Exception as e:
            logger.error(f"  加载 {split} {feature} 已处理（已缩放）数据时出错: {e}", exc_info=True)
            scaled_data[split] = None # 标记为 None

    # 检查是否至少加载了训练数据
    if not data_loaded or scaled_data.get("train") is None:
        logger.error(f"未能加载特征 '{feature}' 的训练数据。无法继续进行 PCA。")
        return False

    # --- 3. PCA 训练 (仅在训练数据上) ---
    pca = PCA(n_components=n_components)
    try:
        logger.info(f"  在 {feature} 的训练数据上拟合 PCA 模型...")
        train_data = scaled_data['train']
        pca.fit(train_data)
        logger.info(f"  PCA ({feature}) 拟合完成。")
        logger.info(f"  解释的总方差比例: {np.sum(pca.explained_variance_ratio_):.4f}")
        logger.info(f"  各主成分解释的方差比例: {pca.explained_variance_ratio_}")
    except Exception as e:
        logger.error(f"  拟合 PCA ({feature}) 时出错: {e}", exc_info=True)
        return False

    # --- 4. 保存 PCA 模型 ---
    pca_model_path = os.path.join(pca_models_dir, f"pca_model_{feature}.pkl")
    try:
        joblib.dump(pca, pca_model_path)
        logger.info(f"  PCA 模型 ({feature}) 已保存到: {pca_model_path}")
    except Exception as e:
        logger.error(f"  保存 PCA 模型 ({feature}) 时出错: {e}", exc_info=True)
        # 即使模型保存失败，我们可能仍想尝试转换（如果 PCA 对象存在）
        # 但通常这意味着后续步骤也可能失败，所以返回 False
        return False

    # --- 5. PCA 转换 (所有有效的数据分割) ---
    all_transformed = True
    for split in splits_to_process:
        data_to_transform = scaled_data.get(split)
        if data_to_transform is not None:
            logger.info(f"  使用训练好的 PCA 模型转换 {split} {feature} 数据...")
            try:
                pca_components = pca.transform(data_to_transform)
                logger.info(f"  转换后的 {split} {feature} 数据形状: {pca_components.shape}")

                # --- 6. 保存 PCA 成分 ---
                components_save_path = os.path.join(pca_components_dir, f"pca_components_{feature}_{split}.npy")
                np.save(components_save_path, pca_components)
                logger.info(f"  PCA 成分 ({feature} {split}) 已保存到: {components_save_path}")

            except Exception as e:
                logger.error(f"  转换或保存 {split} {feature} 的 PCA 成分时出错: {e}", exc_info=True)
                all_transformed = False # 标记至少有一个分割失败
        else:
            logger.warning(f"  跳过转换 {split} {feature}，因为未加载数据。")
            # 如果某个分割的数据缺失是可以接受的，则保持 all_transformed 不变
            # 如果所有分割都必须存在，则应在此处将 all_transformed 设为 False

    logger.info(f"--- 特征 {feature} 的 PCA 处理完成 {'并成功保存所有转换结果' if all_transformed else '但部分转换或保存失败'} ---")
    return all_transformed # 如果所有转换和保存都成功，则返回 True


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig):
    """
    主函数，使用 Hydra 配置运行 PCA 预处理。
    """
    logger.info("====== 开始执行 PCA 预处理脚本 ======")
    logger.info(f"使用的配置文件:\n{OmegaConf.to_yaml(cfg)}") # 打印配置用于调试

    # 获取要处理的特征列表
    # 假设在 model.dimensionality_reduction.pca 下有一个 target_features 列表
    # 或者，如果 components 字典存在，则使用其键作为特征列表
    features_to_process = cfg.model.dimensionality_reduction.pca.get("target_features", None)
    if features_to_process is None:
        if cfg.model.dimensionality_reduction.pca.n_components == 15:
             features_to_process = list(cfg.model.dimensionality_reduction.pca.n_components.keys())
        else:
             # 如果都没有，可以尝试使用默认值，或者报错
             logger.warning("未在配置中找到 model.dimensionality_reduction.pca.target_features，将尝试处理 'wind' 和 'salinity'")
             features_to_process = ["wind", "salinity"] # 默认值

    logger.info(f"将要处理的特征: {features_to_process}")

    overall_success = True
    for feature in features_to_process:
        success = run_pca_preprocessing_for_feature(cfg, feature)
        if not success:
            overall_success = False
            logger.error(f"处理特征 '{feature}' 时遇到错误。")
            # 你可以选择在这里停止，或者继续处理其他特征
            # break # 如果一个失败就停止

    if overall_success:
        logger.info("====== PCA 预处理脚本成功完成 ======")
    else:
        logger.error("====== PCA 预处理脚本完成，但至少有一个特征的处理失败 ======")


if __name__ == "__main__":
    # 配置基本的日志记录器，以便在 Hydra 初始化前也能看到日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    main()