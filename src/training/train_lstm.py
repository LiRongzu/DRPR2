# src/training/train_lstm.py
import os
import sys
import numpy as np
import torch
import logging
import time
import hydra
from omegaconf import DictConfig, OmegaConf
import pickle
import shutil
from typing import Optional, Dict, Any, Tuple, List

# --- 项目设置 ---
# ... (如果需要) ...

# --- 集中导入 ---
from src.utils.hydra_config import DrprConfig
# from src.utils.logger import setup_logger # 假设 Pipeline 设置 logger
from src.prediction_models.lstm_pytorch import LSTMPredictionModel # 使用修改后的版本
from src.data_processing.sequence_utils import create_sequences # 假设这个函数OK
# from src.utils.data_loader import load_processed_data # 可能不再需要
from src.utils.model_utils import get_device_from_config

logger = logging.getLogger(__name__)

def train_and_predict_lstm(
    cfg: DictConfig,
    low_dim_data_paths: Dict[str, Dict[str, str]],
    # target_field_name: str,              # 目标特征名 (e.g., 'salinity')
    # input_feature_info: Dict[str, Dict], # 包含节点数: {'salinity': {'num_nodes': 100}, 'wind_flow': {'num_nodes': 64}}
    # --- 目录和模式 ---
    model_save_dir: str,                 # LSTM 模型保存目录
    prediction_save_dir: str,            # 预测结果保存目录
    prediction_filename_pattern: str = "predicted_target_lstm_{split}.npy" # 保存预测的目标 BMU 文件名模式
) -> Dict[str, Any]:
    # \"\"\"
    # 在低维 BMU 索引数据上训练 LSTM 模型并预测目标 BMU 序列。
    # 支持多特征输入，但只预测目标特征。

    # Args:
    #     cfg: Hydra 配置对象。
    #     low_dim_data_paths: 包含各特征 BMU 索引文件路径的字典 {feature: {split: path}}。
    #     target_field_name: 目标特征的名称。
    #     input_feature_info: 包含每个输入特征信息的字典，至少需要 {'num_nodes': N}。
    #                        特征顺序应与加载/堆叠顺序一致 (target 通常放第一个)。
    #     model_save_dir: 保存训练好的 LSTM 模型的目录。
    #     prediction_save_dir: 保存预测出的 *目标* BMU 序列的目录。
    #     prediction_filename_pattern: 预测的文件名模式 (应包含 {split})。

    # Returns:
    #     包含结果（如模型路径、预测路径）的字典。
    # \"\"\"
    config = DrprConfig.from_hydra_config(cfg)
    results = {}
    start_time = time.time()
    device = get_device_from_config(cfg) # 获取设备

    # --- 1. 确定输入特征和顺序 ---
    # 确保 target 在第一个位置，以便后续处理
    target_field_name = "salinity"
    # --- MODIFICATION START: Correctly get feature names from the dictionary keys ---
    # feature_names_ordered = [target_field_name] + \
    #                         [f for f in low_dim_data_paths.keys() if f != target_field_name]
    # Get actual feature names present in the loaded data dictionary
    actual_features = list(low_dim_data_paths.keys())
    if target_field_name not in actual_features:
        logger.error(f"Target field '{target_field_name}' not found in the keys of low_dim_data_paths: {actual_features}")
        return {}
    feature_names_ordered = [target_field_name] + [f for f in actual_features if f != target_field_name]
    # --- MODIFICATION END ---
    num_input_features = len(feature_names_ordered)
    target_feature_index = 0 # 目标特征在堆叠后的索引
    logger.info(f"LSTM 输入特征顺序: {feature_names_ordered} (共 {num_input_features} 个)")
    logger.info(f"目标特征 '{target_field_name}' 在索引 {target_feature_index}")

    # --- 动态构建 input_feature_info ---
    # Initialize with the target feature
    try:
        input_feature_info = {
            target_field_name: {
                'num_nodes': cfg.training.som.map_size_sta[0] * cfg.training.som.map_size_sta[1]
            }
        }
        # --- MODIFICATION START: Use cfg.model instead of cfg.models ---
        # Add observation features dynamically if enabled
        if cfg.model.prediction.lstm.get('use_observation_features', False): # Use .get for safety
            obs_features = cfg.model.prediction.lstm.observation_features
        # --- MODIFICATION END ---
            if isinstance(obs_features, list): # Ensure it's a list
                num_obs_nodes = cfg.training.som.map_size_obs[0] * cfg.training.som.map_size_obs[1]
                for feature_name in obs_features:
                    if feature_name != target_field_name: # Avoid overwriting target if listed again
                        input_feature_info[feature_name] = {'num_nodes': num_obs_nodes}
            elif obs_features: # Handle if it's unexpectedly a single string? Log warning.
                 logger.warning(f"Expected 'observation_features' in config to be a list, but got {type(obs_features)}. Processing as single feature: {obs_features}")
                 num_obs_nodes = cfg.training.som.map_size_obs[0] * cfg.training.som.map_size_obs[1]
                 if obs_features != target_field_name:
                     input_feature_info[obs_features] = {'num_nodes': num_obs_nodes}

        logger.info(f"Constructed input_feature_info: {input_feature_info}")

    except AttributeError as e:
        logger.error(f"Error accessing configuration for map sizes (e.g., cfg.training.som...). Check config structure: {e}", exc_info=True)
        return {}
    except Exception as e:
        logger.error(f"Error constructing input_feature_info: {e}", exc_info=True)
        return {}


    # --- 检查 input_feature_info 是否完整 ---
    # (Ensure feature_names_ordered matches the keys generated above)
    # Re-check feature_names_ordered based on the actual keys in input_feature_info if necessary,
    # maintaining target_field_name at index 0.
    current_features = list(input_feature_info.keys())
    if target_field_name not in current_features:
        logger.error(f"Target field '{target_field_name}' not found in constructed input_feature_info keys!")
        return {}
    # --- MODIFICATION START: Ensure feature_names_ordered only contains features present in input_feature_info ---
    # feature_names_ordered = [target_field_name] + [f for f in current_features if f != target_field_name]
    # Filter feature_names_ordered to only include those actually in input_feature_info
    feature_names_ordered = [f for f in feature_names_ordered if f in current_features]
    if not feature_names_ordered or feature_names_ordered[0] != target_field_name: # Double check target is still first
        logger.warning(f"Reordering features to place target '{target_field_name}' first.")
        if target_field_name in feature_names_ordered:
             feature_names_ordered.remove(target_field_name)
        feature_names_ordered.insert(0, target_field_name)
    # --- MODIFICATION END ---
    num_input_features = len(feature_names_ordered) # Update num_input_features
    logger.info(f"Final LSTM input feature order: {feature_names_ordered}")


    if not all(f in input_feature_info and 'num_nodes' in input_feature_info[f] for f in feature_names_ordered):
        logger.error(f"input_feature_info 缺失某些特征的 'num_nodes' 信息。需要: {feature_names_ordered}")
        return results

    # --- 2. 加载并合并低维 BMU 索引数据 ---
    # This part seems to have been corrected previously to handle dictionary paths
    logger.info("加载并合并 BMU 索引数据...")
    low_dim_data: Dict[str, np.ndarray] = {} # 存储合并后的数据: {split: (T, num_features)}
    expected_length = -1

    try:
        for split in ["train", "val", "test"]:
            split_data_list = []
            has_split_data = False
            # --- MODIFICATION START: Iterate over the FINAL feature_names_ordered ---
            for i, feature in enumerate(feature_names_ordered): # Use the final ordered list
            # --- MODIFICATION END ---
                # path = low_dim_data_paths.get(feature, {}).get(split) # Original logic assumed paths were strings
                # --- Corrected logic from previous step (assuming it's correct) ---
                path_info = low_dim_data_paths.get(feature, {}).get(split)
                path_to_load = None
                if isinstance(path_info, dict):
                    bmu_positions_path = path_info.get('positions')
                    if bmu_positions_path and os.path.exists(bmu_positions_path):
                        path_to_load = bmu_positions_path
                    else:
                        logger.warning(f"  未找到或无效的 'positions' 路径 for 特征 '{feature}', 分割 '{split}': {path_info}")
                elif isinstance(path_info, str) and os.path.exists(path_info):
                     logger.warning(f"    特征 '{feature}' 的 BMU 路径直接是字符串: {path_info}. 假设这是 BMU 索引文件。")
                     path_to_load = path_info
                # --- End Corrected logic ---

                if path_to_load:
                    try:
                        data = np.load(path_to_load).flatten() # 确保是一维
                        has_split_data = True # 只要有一个特征有数据，就处理这个 split
                        logger.debug(f"  加载 {split} - 特征 '{feature}' from: {path_to_load}, 形状: {data.shape}")

                        # 长度检查
                        if not split_data_list: # 第一个成功加载的特征决定长度
                             expected_length = len(data)
                        elif len(data) != expected_length:
                             raise ValueError(f"序列长度不一致! {split} - 特征 '{feature}' ({len(data)}) != 预期 ({expected_length})")

                        split_data_list.append(data)
                    except FileNotFoundError:
                         logger.error(f"  文件未找到: {path_to_load}")
                         # Decide how to handle - break split?
                         has_split_data = False
                         break
                    except Exception as load_err:
                         logger.error(f"  加载 BMU 文件 {path_to_load} 失败: {load_err}")
                         has_split_data = False
                         break
                else:
                    # If a required feature's path is missing for this split
                    logger.error(f"  未找到 {split} 数据 for 必需特征 '{feature}'。无法继续处理此分割。")
                    has_split_data = False
                    break # Stop processing this split

            if has_split_data and len(split_data_list) == num_input_features:
                 # 堆叠特征 -> (T, num_features)
                 combined_split_data = np.stack(split_data_list, axis=-1)
                 logger.info(f"  合并后 {split} 数据形状: {combined_split_data.shape}")
                 low_dim_data[split] = combined_split_data
                 expected_length = -1 # 重置
            elif has_split_data: # 意味着有些特征缺失，无法堆叠
                 logger.error(f"未能加载 {split} 的所有必需特征数据 ({len(split_data_list)}/{num_input_features})，跳过此分割。")


    except ValueError as e: # Catch length mismatch error
        logger.error(f"加载或合并 LSTM 输入数据时出错: {e}", exc_info=True)
        return results
    except Exception as e:
        logger.error(f"加载或合并 LSTM 输入数据时发生意外错误: {e}", exc_info=True)
        return results

    if 'train' not in low_dim_data:
        logger.error("缺少合并后的训练数据，无法训练 LSTM。")
        return results


    # --- 3. 准备 LSTM 序列 ---
    # --- MODIFICATION START: Use cfg.model ---
    lstm_cfg = config.model.prediction.lstm
    # --- MODIFICATION END ---
    seq_len = lstm_cfg.get("sequence_length", 10)
    logger.info(f"为 LSTM 创建序列，输入序列长度: {seq_len}...")

    # 确定目标特征在堆叠数据中的索引 (我们约定是 0)
    target_feature_index = 0
    # --- MODIFICATION START: Check against the FINAL feature_names_ordered ---
    if feature_names_ordered[target_feature_index] != target_field_name:
        logger.error(f"逻辑错误：目标特征 '{target_field_name}' 未按预期放在第 {target_feature_index} 列！顺序: {feature_names_ordered}")
        # Attempt to find it
        try:
            target_feature_index = feature_names_ordered.index(target_field_name)
            logger.warning(f"目标特征在索引 {target_feature_index} 找到。")
        except ValueError:
             logger.error("逻辑错误：目标特征未在最终输入特征列表中找到！")
             return {}
    # --- MODIFICATION END ---


    X_seq: Dict[str, np.ndarray] = {}
    y_seq: Dict[str, np.ndarray] = {} # 目标是 Target BMU index (int)

    for split, stacked_data in low_dim_data.items():
        # stacked_data shape: (T, num_input_features)
        logger.debug(f"为 split '{split}' 创建序列，输入数据形状: {stacked_data.shape}")

        if len(stacked_data) > seq_len:
            # target_data shape: (T,)
            target_data = stacked_data[:, target_feature_index]

            X_seq_split, y_seq_split = create_sequences(
                X=stacked_data,         # (T, num_features)
                y=target_data,          # (T,)
                sequence_length=seq_len
            )

            # 检查返回的形状是否符合预期
            if X_seq_split.ndim != 3 or X_seq_split.shape[1] != seq_len or X_seq_split.shape[2] != num_input_features:
                logger.error(f"'{split}' 的 X_seq 形状不正确: {X_seq_split.shape} (预期: (N, {seq_len}, {num_input_features}))")
                # 可以选择跳过这个 split 或中止
                continue
            if y_seq_split.ndim != 1:
                logger.error(f"'{split}' 的 y_seq 形状不正确: {y_seq_split.shape} (预期: (N,))")
                continue

            X_seq[split] = X_seq_split
            y_seq[split] = y_seq_split

            logger.info(f"  {split} 序列创建完成: X={X_seq[split].shape}, y={y_seq[split].shape}")
        else:
            logger.warning(f"  {split} 数据长度不足 ({len(stacked_data)})，无法创建长度为 {seq_len} 的序列。")

    # --- 后续检查 'train' 是否成功创建序列 ---
    if 'train' not in X_seq or X_seq['train'].size == 0:
        logger.error("未能成功创建训练序列。检查数据长度和序列长度设置。")
        return {} # 返回空字典表示失败

    # --- 4. 移除 Scaling ---
    logger.info("BMU 索引数据，跳过标准化缩放。")

    # --- 5. 初始化和训练 LSTM 模型 ---
    logger.info("初始化并训练 LSTM 模型 (用于 BMU 索引预测)...")

    # --- 准备模型参数 ---
    num_embeddings_list = [input_feature_info[f]['num_nodes'] for f in feature_names_ordered]
    # 假设所有 embedding 维度相同，从配置读取
    # --- MODIFICATION START: Use cfg.model ---
    default_emb_dim = lstm_cfg.get("embedding_dim", 32) # Example default
    # --- MODIFICATION END ---
    embedding_dims_list = [default_emb_dim] * num_input_features
    target_som_num_nodes = input_feature_info[target_field_name]['num_nodes']

    lstm_model = LSTMPredictionModel(
        # --- 架构 ---
        num_embeddings_list=num_embeddings_list,
        embedding_dims_list=embedding_dims_list,
        # --- MODIFICATION START: Use cfg.model ---
        hidden_size=lstm_cfg.get("hidden_size", 64),
        target_som_num_nodes=target_som_num_nodes,
        num_layers=lstm_cfg.get("num_layers", 2),
        dropout=lstm_cfg.get("dropout", 0.1),
        # --- 训练 ---
        epochs=cfg.training.get("epochs", 100),
        batch_size=lstm_cfg.get("batch_size", 32),
        # --- MODIFICATION END ---
        learning_rate=cfg.training.optimizer.get("learning_rate", 0.001),
        patience=cfg.training.early_stopping.get("patience", 10),
        # --- 其他 ---
        sequence_length=seq_len, # 参考用
        random_seed=cfg.training.random_seed,
        device=device
    )

    # --- 训练 ---
    lstm_model.fit(
        X_train=X_seq['train'], y_train=y_seq['train'],
        X_val=X_seq.get('val'), y_val=y_seq.get('val') # 传递验证集 (如果存在)
    )

    # --- 6. 保存 LSTM 模型 ---
    os.makedirs(model_save_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    lstm_model_path = os.path.join(model_save_dir, f"lstm_model_{timestamp}.pt")
    latest_path = os.path.join(model_save_dir, "latest_bmu_lstm.pt") # 使用特定名称
    try:
        lstm_model.save(lstm_model_path)
        logger.info(f"LSTM 模型已保存到: {lstm_model_path}")
        shutil.copy2(lstm_model_path, latest_path)
        logger.info(f"LSTM 模型副本已保存为 {latest_path}")
        results['model_path'] = latest_path # 返回最新模型的路径
    except Exception as e:
        logger.error(f"保存 LSTM 模型失败: {e}")
        # 即使保存失败，也可能继续预测（如果模型仍在内存中）
        # return results # 或者在这里返回

    # --- 7. 使用 LSTM 模型预测 ---
    logger.info("使用训练好的 LSTM 预测 *目标* BMU 索引序列...")
    # 注意：key 改为 'predicted_target_low_dim_paths'
    results['predicted_target_low_dim_paths'] = {}
    os.makedirs(prediction_save_dir, exist_ok=True)

    for split in ["train", "val", "test"]:
        if split not in X_seq:
             logger.warning(f"跳过 {split} 分割的 LSTM 预测，因为没有输入序列。")
             continue

        logger.info(f"  预测 {split} 分割...")
        try:
            # 使用模型的 predict 方法
            predicted_target_indices = lstm_model.predict(X_seq[split]) # (num_samples,)
            logger.info(f"  预测的 {split} 目标 BMU 索引形状: {predicted_target_indices.shape}")

            # 保存预测的目标索引结果
            pred_save_path = os.path.join(prediction_save_dir, prediction_filename_pattern.format(split=split))
            np.save(pred_save_path, predicted_target_indices)
            logger.info(f"  预测的 {split} *目标* BMU 索引序列已保存到: {pred_save_path}")
            results['predicted_target_low_dim_paths'][split] = pred_save_path

        except Exception as e:
            logger.error(f"  预测 {split} 目标 BMU 索引序列失败: {e}", exc_info=True)


    total_run_time = time.time() - start_time
    logger.info(f"LSTM 训练和预测完成，总用时：{total_run_time:.2f}秒")

    return results