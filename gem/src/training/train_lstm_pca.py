# src/training/train_lstm_pca.py
import os
import sys
import numpy as np
import torch
import logging
import time
import hydra
from omegaconf import DictConfig, OmegaConf, ListConfig
import pickle
import shutil
from typing import Optional, Dict, Any, Tuple, List

# --- 集中导入 ---
from src.utils.hydra_config import DrprConfig
from src.prediction_models.lstm_pca import LSTMPredictionModel # 使用 PCA 版本的 LSTM
from src.data_processing.sequence_utils import create_sequences # 已更新
from src.utils.model_utils import get_device_from_config

logger = logging.getLogger(__name__)

def train_and_predict_lstm_pca( # Renamed function for clarity
    cfg: DictConfig,
    low_dim_data_paths: Dict[str, Dict[str, str]], # 结构简化: {feature: {split: path_to_pca_data.npy}}
    pca_components_info: Dict[str, int], # 新增: {feature: n_components}
    # --- 目录和模式 ---
    model_save_dir: str,
    prediction_save_dir: str,
    prediction_filename_pattern: str = "predicted_target_pca_lstm_{split}.npy" # Updated pattern
) -> Dict[str, Any]:
    """
    Trains an LSTM model on continuous low-dimensional data (e.g., PCA components)
    and predicts the target feature's low-dimensional representation.
    """
    config = DrprConfig.from_hydra_config(cfg)
    results = {}
    start_time = time.time()
    device = get_device_from_config(cfg)

    # --- 1. 确定输入特征、顺序和 PCA 组件数 ---
    target_field_name = cfg.model.dimensionality_reduction.get("target_feature", "salinity")
    actual_features = list(low_dim_data_paths.keys())
    if target_field_name not in actual_features:
        logger.error(f"目标特征 '{target_field_name}' 在 low_dim_data_paths 的键中未找到: {actual_features}")
        return {}
    if target_field_name not in pca_components_info:
        logger.error(f"目标特征 '{target_field_name}' 在 pca_components_info 中未找到组件数。")
        return {}

    # 确保目标特征在第一位
    feature_names_ordered = [target_field_name] + [f for f in actual_features if f != target_field_name]
    logger.info(f"初步 LSTM (PCA) 输入特征顺序: {feature_names_ordered}")

    # --- 验证特征在 pca_components_info 中都有定义 ---
    valid_feature_names = []
    total_input_components = 0
    target_output_components = 0
    feature_component_slices: Dict[str, slice] = {} # Store slice for each feature in the combined array
    current_start_index = 0

    for feature in feature_names_ordered:
        if feature in pca_components_info and feature in low_dim_data_paths:
            n_comp = pca_components_info[feature]
            if n_comp <= 0:
                 logger.warning(f"特征 '{feature}' 的 PCA 组件数 ({n_comp}) 无效，跳过此特征。")
                 continue

            valid_feature_names.append(feature)
            feature_slice = slice(current_start_index, current_start_index + n_comp)
            feature_component_slices[feature] = feature_slice
            current_start_index += n_comp
            total_input_components += n_comp

            if feature == target_field_name:
                target_output_components = n_comp
                logger.info(f"目标特征 '{feature}' 使用 {n_comp} PCA 组件 (将作为 LSTM 输出)。")
            else:
                 logger.info(f"输入特征 '{feature}' 使用 {n_comp} PCA 组件。")

        else:
            logger.warning(f"特征 '{feature}' 在 low_dim_data_paths 或 pca_components_info 中缺失，将从 LSTM 输入中排除。")

    # 更新为最终有效的、有序的特征列表
    feature_names_ordered = valid_feature_names
    num_input_features = len(feature_names_ordered) # Number of feature *groups*
    logger.info(f"最终有效的 LSTM (PCA) 输入特征顺序: {feature_names_ordered} (共 {num_input_features} 组)")
    logger.info(f"总输入 PCA 组件数 (LSTM input_size): {total_input_components}")
    logger.info(f"目标输出 PCA 组件数 (LSTM output_size): {target_output_components}")
    logger.info(f"特征组件切片: {feature_component_slices}")


    if num_input_features == 0 or total_input_components == 0 or target_output_components == 0:
        logger.error("没有有效的输入/输出特征或 PCA 组件可供 LSTM 使用。")
        return {}
    if feature_names_ordered[0] != target_field_name:
         logger.error(f"逻辑错误：目标特征 '{target_field_name}' 未成为第一个有效特征。顺序: {feature_names_ordered}")
         return {}


    # --- 2. 加载并合并低维 PCA 数据 ---
    logger.info("加载并合并 PCA 数据...")
    low_dim_data: Dict[str, np.ndarray] = {} # 存储合并后的数据: {split: (T, total_input_components)}

    try:
        # --- 循环处理 train, val, test ---
        for split in ["train", "val", "test"]:
            split_data_list = [] # 存储当前 split 的所有特征的 (T, n_comp) 数组
            has_split_data = False
            current_split_length = -1 # 当前 split 的预期时间长度 T

            # --- 循环处理每个需要的特征 ---
            for feature in feature_names_ordered:
                path_to_load = low_dim_data_paths.get(feature, {}).get(split)

                if path_to_load and os.path.exists(path_to_load):
                    try:
                        # 1. 加载数据 (T, n_components_feature)
                        loaded_data = np.load(path_to_load)
                        expected_components = pca_components_info[feature]
                        logger.debug(f"  加载 {split} - 特征 '{feature}' from: {path_to_load}, 形状: {loaded_data.shape}")

                        # 2. 验证形状
                        if loaded_data.ndim != 2 or loaded_data.shape[1] != expected_components:
                            raise ValueError(f"PCA 文件 {path_to_load} 的形状 {loaded_data.shape} 不符合预期 ({expected_components} 组件)。")

                        has_split_data = True # 标记此 split 有数据

                        # 3. 长度检查
                        if current_split_length == -1: # 这是此 split 的第一个特征
                            current_split_length = len(loaded_data)
                            logger.debug(f"  设置 {split} 分割的预期长度为: {current_split_length}")
                        elif len(loaded_data) != current_split_length:
                            raise ValueError(f"序列长度不一致! {split} - 特征 '{feature}' ({len(loaded_data)}) != 预期 ({current_split_length})")

                        # 4. 添加到列表
                        split_data_list.append(loaded_data) # 添加 (T, n_comp) 数组

                    # --- Error Handling ---
                    except FileNotFoundError:
                        logger.error(f"  文件未找到: {path_to_load}")
                        has_split_data = False; break # 停止处理此 split
                    except ValueError as ve:
                        logger.error(f"  处理 PCA 文件 {path_to_load} 时出错: {ve}")
                        has_split_data = False; break # 停止处理此 split
                    except Exception as load_err:
                        logger.error(f"  加载 PCA 文件 {path_to_load} 失败: {load_err}", exc_info=True)
                        has_split_data = False; break # 停止处理此 split
                else:
                    # 如果某个必需特征的路径找不到
                    logger.error(f"  未找到 {split} 数据 for 必需特征 '{feature}' (路径: {path_to_load})。无法继续处理此分割。")
                    has_split_data = False
                    break # 停止处理此 split

            # --- 合并当前 split 的数据 ---
            if has_split_data and len(split_data_list) == num_input_features:
                # *** 核心合并步骤 ***
                # split_data_list 是 [(T, nc1), (T, nc2), ...] 列表
                # 使用 np.concatenate(..., axis=1)
                try:
                    # 确保按 feature_names_ordered 的顺序合并
                    combined_split_data = np.concatenate(split_data_list, axis=1)

                    # --- >>> 验证形状 <<< ---
                    if combined_split_data.shape != (current_split_length, total_input_components):
                         logger.error(f"合并 {split} 数据后的形状不正确: {combined_split_data.shape}，预期: ({current_split_length}, {total_input_components})")
                         continue # 跳过这个错误的 split
                    logger.info(f"  合并后 {split} 数据形状: {combined_split_data.shape}")
                    low_dim_data[split] = combined_split_data
                except Exception as concat_err:
                    logger.error(f"合并 {split} 分割的 PCA 特征数据时出错: {concat_err}", exc_info=True)
                    # 跳过这个 split

            elif has_split_data: # 加载了部分但非全部特征
                logger.error(f"未能加载 {split} 的所有必需特征数据 ({len(split_data_list)}/{num_input_features})，跳过此分割。")

            # 重置长度以便下一个 split 重新确定
            current_split_length = -1

    except Exception as e:
        logger.error(f"加载或合并 LSTM (PCA) 输入数据时发生意外错误: {e}", exc_info=True)
        return {}

    # --- 检查是否有训练数据 ---
    if 'train' not in low_dim_data or low_dim_data['train'].size == 0:
        logger.error("缺少合并后的训练数据或训练数据为空，无法训练 LSTM (PCA)。")
        return results

    # --- 3. 准备 LSTM 序列 ---
    lstm_cfg = config.model.prediction.lstm # Assuming LSTM config is still under this path
    seq_len = lstm_cfg.get("sequence_length", 10)
    logger.info(f"为 LSTM (PCA) 创建序列，输入序列长度: {seq_len}...")

    # 确定目标特征在合并数据中的切片
    target_feature_slice = feature_component_slices[target_field_name]
    logger.info(f"目标特征 '{target_field_name}' 在合并数据中的切片: {target_feature_slice}")

    X_seq: Dict[str, np.ndarray] = {}
    y_seq: Dict[str, np.ndarray] = {} # 目标是 Target PCA components (float)

    for split, combined_data in low_dim_data.items():
        # combined_data shape: (T, total_input_components)
        logger.debug(f"为 split '{split}' 创建序列，输入数据形状: {combined_data.shape}")

        if len(combined_data) >= seq_len + 1: # 确保至少有 seq_len+1 个点来创建一个序列
            # target_data shape: (T, target_output_components)
            target_data = combined_data[:, target_feature_slice]

            # 使用更新后的 create_sequences
            X_seq_split, y_seq_split = create_sequences(
                X=combined_data,         # (T, total_input_components)
                y=target_data,           # (T, target_output_components)
                sequence_length=seq_len
            )

            # 验证 create_sequences 的输出形状
            expected_x_shape = (len(combined_data) - seq_len, seq_len, total_input_components)
            expected_y_shape = (len(combined_data) - seq_len, target_output_components)
            # Adjust expected_y_shape if target_output_components is 1 (create_sequences squeezes)
            if target_output_components == 1:
                 expected_y_shape = (len(combined_data) - seq_len,)


            if X_seq_split.shape != expected_x_shape:
                logger.error(f"'{split}' 的 X_seq 形状不正确: {X_seq_split.shape} (预期: {expected_x_shape})")
                continue # 跳过这个 split
            if y_seq_split.shape != expected_y_shape:
                 # If target_output_components is 1, y_seq_split might be (N,) while expected is (N, 1) before squeeze.
                 # The check inside create_sequences handles the squeeze, so this check should align.
                 logger.error(f"'{split}' 的 y_seq 形状不正确: {y_seq_split.shape} (预期: {expected_y_shape})")
                 continue


            X_seq[split] = X_seq_split
            y_seq[split] = y_seq_split
            logger.info(f"  {split} 序列创建完成: X={X_seq[split].shape}, y={y_seq[split].shape}")
        else:
            logger.warning(f"  {split} 数据长度不足 ({len(combined_data)})，无法创建长度为 {seq_len} 的序列。")

    if 'train' not in X_seq or X_seq['train'].size == 0:
         logger.error("未能成功创建训练序列。检查数据长度和序列长度设置。")
         return {}

    # --- 4. 标准化 (可选，但对 PCA 输出通常推荐) ---
    # TODO: Implement scaling if needed (e.g., StandardScaler)
    # scaler_X = StandardScaler()
    # scaler_y = StandardScaler()
    # X_train_scaled = scaler_X.fit_transform(X_seq['train'].reshape(-1, total_input_components)).reshape(X_seq['train'].shape)
    # y_train_scaled = scaler_y.fit_transform(y_seq['train']) # Adjust reshape based on y shape
    # ... apply transform to val/test ...
    # Remember to save scalers and inverse transform predictions later
    logger.info("PCA 数据标准化步骤已跳过 (TODO: implement if needed)。")
    # Using unscaled data for now
    X_train_final = X_seq['train']
    y_train_final = y_seq['train']
    X_val_final = X_seq.get('val')
    y_val_final = y_seq.get('val')


    # --- 5. 初始化和训练 LSTM 模型 ---
    logger.info("初始化 LSTM (PCA) 模型...")

    lstm_model = LSTMPredictionModel(
        # --- 架构 ---
        input_size=total_input_components,   # Total PCA components from all input features
        output_size=target_output_components, # PCA components for the target feature
        hidden_size=lstm_cfg.get("hidden_size", 64),
        num_layers=lstm_cfg.get("num_layers", 2),
        dropout=lstm_cfg.get("dropout", 0.1),
        # --- 训练 ---
        epochs=lstm_cfg.get("epochs", 100),
        batch_size=lstm_cfg.get("batch_size", 32),
        learning_rate=lstm_cfg.get("lr", 0.001),
        patience=lstm_cfg.get("patience", 10),
        # --- 其他 ---
        sequence_length=seq_len, # 参考用
        random_seed=cfg.training.random_seed,
        device=device
    )

    # --- 训练 ---
    logger.info("开始训练 LSTM (PCA) 模型...")
    lstm_model.fit(
        X_train=X_train_final, y_train=y_train_final,
        X_val=X_val_final, y_val=y_val_final # 传递验证集 (如果存在)
    )

    # --- 6. 保存 LSTM 模型 ---
    os.makedirs(model_save_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    lstm_model_path = os.path.join(model_save_dir, f"lstm_pca_model_{timestamp}.pt")
    latest_path = os.path.join(model_save_dir, "latest_pca_lstm.pt") # 使用特定名称
    try:
        lstm_model.save(lstm_model_path)
        logger.info(f"LSTM (PCA) 模型已保存到: {lstm_model_path}")
        shutil.copy2(lstm_model_path, latest_path)
        logger.info(f"LSTM (PCA) 模型副本已保存为 {latest_path}")
        results['model_path'] = latest_path # 返回最新模型的路径
    except Exception as e:
        logger.error(f"保存 LSTM (PCA) 模型失败: {e}")
        # return results # 或者在这里返回

    # --- 7. 使用 LSTM 模型预测 ---
    logger.info("使用训练好的 LSTM (PCA) 预测 *目标* PCA 组件序列...")
    results['predicted_target_low_dim_paths'] = {}
    os.makedirs(prediction_save_dir, exist_ok=True)

    for split in ["train", "val", "test"]:
        if split not in X_seq: # Use original X_seq for prediction input
            logger.warning(f"跳过 {split} 分割的 LSTM (PCA) 预测，因为没有输入序列。")
            continue

        # Use the final (potentially scaled) X for prediction if scaling was applied
        X_pred_input = X_seq[split] # Or X_test_scaled if scaling was done

        logger.info(f"  预测 {split} 分割...")
        try:
            # predict method returns shape (N, output_size)
            predicted_target_components = lstm_model.predict(X_pred_input)
            logger.info(f"  预测的 {split} 目标 PCA 组件形状: {predicted_target_components.shape}")

            # --- TODO: Inverse transform predictions if scaling was applied ---
            # predicted_target_components = scaler_y.inverse_transform(predicted_target_components)
            # logger.info(f"  逆标准化后的 {split} 目标 PCA 组件形状: {predicted_target_components.shape}")


            # --- 保存预测结果 ---
            pred_save_path = os.path.join(prediction_save_dir, prediction_filename_pattern.format(split=split))
            np.save(pred_save_path, predicted_target_components)
            logger.info(f"  预测的 {split} *目标* PCA 组件序列已保存到: {pred_save_path}")
            results['predicted_target_low_dim_paths'][split] = pred_save_path

        except Exception as e:
            logger.error(f"  预测 {split} 目标 PCA 组件序列失败: {e}", exc_info=True)


    total_run_time = time.time() - start_time
    logger.info(f"LSTM (PCA) 训练和预测完成，总用时：{total_run_time:.2f}秒")

    results['success'] = True # Indicate overall success if reached here
    return results

# --- Placeholder for potential main execution or testing ---
# if __name__ == '__main__':
#     # Example usage (requires setting up mock cfg, paths, info)
#     pass