# 在 src/training/train_pca.py 文件中

import os
import joblib
import numpy as np
import logging
from sklearn.decomposition import PCA
from omegaconf import DictConfig

logger = logging.getLogger(__name__)

def train_and_transform_pca(cfg: DictConfig, scaled_input_paths: dict,
                           model_save_dir: str, transformed_save_dir: str,
                           model_filename: str, transformed_filename_pattern: str,
                           n_components: int, target_feature: str) -> dict:
    """
    训练 PCA 模型并转换数据。

    Args:
        cfg (DictConfig): 配置对象。
        scaled_input_paths (dict): 包含 'train', 'val', 'test' 缩放数据路径的字典。
                                   Example: {'train': 'path/scaled_salinity_train.npy', ...}
        model_save_dir (str): 保存 PCA 模型的目录。
        transformed_save_dir (str): 保存转换后的低维数据的目录。
        model_filename (str): PCA 模型文件名 (e.g., pca_model_salinity.pkl).
        transformed_filename_pattern (str): 低维数据文件名模式 (e.g., pca_components_salinity_{split}.npy).
        n_components (int): PCA 主成分数量。
        target_feature (str): 正在处理的目标特征名称 (用于日志记录和内部逻辑)。

    Returns:
        dict: 包含 PCA 结果的字典。keys: 'success', 'model_path', 'n_components',
              'low_dim_data_paths', 'explained_variance_ratio', 'cumulative_explained_variance'.
    """
    results = {'success': False, 'low_dim_data_paths': {}}
    try:
        # --- 1. 加载训练数据 ---
        train_data_path = scaled_input_paths.get("train")
        if not train_data_path or not os.path.exists(train_data_path):
            logger.error(f"PCA 训练 ({target_feature}) 所需的缩放训练数据路径无效或文件不存在: {train_data_path}")
            return results

        scaled_high_dim_data_train = np.load(train_data_path)
        logger.info(f"为 PCA ({target_feature}) 加载已缩放的高维训练数据，形状: {scaled_high_dim_data_train.shape}")

        # --- 2. 检查数据维度与 n_components ---
        n_samples, n_features = scaled_high_dim_data_train.shape
        if n_components <= 0:
            logger.error(f"无效的 n_components ({n_components}) 用于 PCA ({target_feature})。必须大于 0。")
            return results
        # scikit-learn PCA 处理 n_components > min(n_samples, n_features) 的情况
        # 但我们可以在这里添加一个警告或检查
        max_possible_components = min(n_samples, n_features)
        if n_components > max_possible_components:
            logger.warning(f"请求的 n_components ({n_components}) 大于数据的最大可能成分数 ({max_possible_components}) "
                           f"对于特征 '{target_feature}'。PCA 将使用 {max_possible_components} 个成分。")
            # 注意：PCA(n_components=X) 如果 X > max_possible 会自动调整或报错，取决于版本。
            # 这里仅记录警告，让 scikit-learn 处理。或者可以强制 n_components = max_possible_components

        # --- 3. 训练 PCA 模型 ---
        logger.info(f"训练 PCA 模型 ({target_feature})，请求的 n_components={n_components}...")
        pca = PCA(n_components=n_components)
        pca.fit(scaled_high_dim_data_train)
        actual_n_components = pca.n_components_ # 获取实际使用的成分数
        logger.info(f"PCA 模型训练完成 ({target_feature})，实际使用 n_components={actual_n_components}。")


        # --- 4. 记录方差信息 ---
        explained_variance_ratio = pca.explained_variance_ratio_.tolist()
        cumulative_explained_variance = np.cumsum(pca.explained_variance_ratio_).tolist()
        logger.info(f"  ({target_feature}) 解释方差比: {explained_variance_ratio}")
        logger.info(f"  ({target_feature}) 累计解释方差: {cumulative_explained_variance}")

        # --- 5. 保存模型 ---
        model_path = os.path.join(model_save_dir, model_filename)
        joblib.dump(pca, model_path)
        logger.info(f"PCA 模型 ({target_feature}) 已保存到: {model_path}")

        results.update({
            'success': True,
            'model_path': model_path,
            'n_components': actual_n_components, # 返回实际使用的成分数
            'explained_variance_ratio': explained_variance_ratio,
            'cumulative_explained_variance': cumulative_explained_variance
        })

        # --- 6. 转换所有数据分割 ---
        logger.info(f"转换所有数据分割到 PCA 空间 ({target_feature})...")
        processed_splits = 0
        for split in ["train", "val", "test"]:
            data_path = scaled_input_paths.get(split)
            if not data_path or not os.path.exists(data_path):
                logger.warning(f"未找到用于 PCA 转换的 '{split}' 数据路径 ({target_feature}): {data_path}，跳过此分割。")
                continue # 或者应该报错并使整个过程失败？看需求

            scaled_high_dim_data = np.load(data_path)
            # 确保数据至少是二维的
            if scaled_high_dim_data.ndim < 2:
                 logger.error(f"PCA 转换的 '{split}' 数据 ({target_feature}) 维度不正确 (需要至少2D)，形状: {scaled_high_dim_data.shape}。")
                 results['success'] = False # 标记失败
                 return results # 提前退出

            try:
                data_low_dim = pca.transform(scaled_high_dim_data)
                logger.info(f"  转换 {split} 数据 ({target_feature})，低维形状: {data_low_dim.shape}")

                # 保存转换后的数据
                save_path = os.path.join(transformed_save_dir, transformed_filename_pattern.format(split=split))
                np.save(save_path, data_low_dim)
                logger.info(f"  PCA 组件 ({split}, {target_feature}) 已保存到: {save_path}")
                results['low_dim_data_paths'][split] = save_path
                processed_splits += 1
            except ValueError as ve:
                 logger.error(f"PCA 转换 '{split}' 数据 ({target_feature}) 时出错: {ve}. "
                              f"输入数据形状: {scaled_high_dim_data.shape}, PCA 期望特征数: {pca.n_features_in_}")
                 results['success'] = False
                 return results # 转换失败则整体失败


        # 检查是否所有分割都被处理了
        if processed_splits != len(scaled_input_paths):
             logger.warning(f"并非所有数据分割都已成功转换并保存 ({target_feature})。成功处理了 {processed_splits}/{len(scaled_input_paths)} 个。")
             # 可以根据需求决定这是否算作失败
             # results['success'] = False

        # 确保 low_dim_data_paths 包含所有预期的键，即使文件未处理也用 None 填充？
        # for split in ["train", "val", "test"]:
        #      results['low_dim_data_paths'].setdefault(split, None)


    except Exception as e:
        logger.error(f"训练或转换 PCA ({target_feature}) 时发生意外错误: {e}", exc_info=True)
        results['success'] = False

    # 在最终返回前确认 n_components 存在于结果中
    if 'n_components' not in results and results['success']:
        # 如果成功但没有 n_components（理论上不应发生），尝试用配置值
        results['n_components'] = n_components
        logger.warning(f"PCA 结果中缺少 n_components ({target_feature})，已使用配置值 {n_components} 填充。")

    return results