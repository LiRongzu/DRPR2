# src/training/train_lstm.py
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
from src.prediction_models.lstm_pytorch import LSTMPredictionModel # 使用修改后的版本
from src.data_processing.sequence_utils import create_sequences # 假设这个函数OK
from src.utils.model_utils import get_device_from_config
from src.utils.bmu_utils import convert_grid_to_linear

logger = logging.getLogger(__name__)

def train_and_predict_lstm(
    cfg: DictConfig,
    low_dim_data_paths: Dict[str, Dict[str, Dict[str, str]]], # 结构 {feature: {split: {'positions': path}}}
    # --- 目录和模式 ---
    model_save_dir: str,
    prediction_save_dir: str,
    prediction_filename_pattern: str = "predicted_target_lstm_{split}.npy"
) -> Dict[str, Any]:
    # """Docstring"""
    config = DrprConfig.from_hydra_config(cfg)
    results = {}
    start_time = time.time()
    device = get_device_from_config(cfg)

    # --- 1. 确定输入特征和顺序 ---
    target_field_name = cfg.model.dimensionality_reduction.get("target_feature", "salinity") # 从DR配置获取目标
    actual_features = list(low_dim_data_paths.keys())
    if target_field_name not in actual_features:
        logger.error(f"目标特征 '{target_field_name}' 在 low_dim_data_paths 的键中未找到: {actual_features}")
        return {}
    # 确保目标特征在第一位
    feature_names_ordered = [target_field_name] + [f for f in actual_features if f != target_field_name]
    logger.info(f"初步 LSTM 输入特征顺序: {feature_names_ordered}")

    # --- 动态构建 input_feature_info (包含节点数) ---
    # (这部分代码似乎在上次修改中已经比较健壮了，但要注意观测特征名的处理)
    try:
        input_feature_info = {
            target_field_name: {
                'num_nodes': cfg.training.som.map_size_sta[0] * cfg.training.som.map_size_sta[1]
            }
        }
        # --- 处理观测特征 ---
        if cfg.model.prediction.lstm.get('use_observation_features', False):
            obs_features_config = cfg.model.prediction.lstm.observation_features
            if isinstance(obs_features_config, (list, ListConfig)): # 接受 list 或 ListConfig
                # 假设观测特征共享一个 SOM (组合特征)
                num_obs_nodes = cfg.training.som.map_size_obs[0] * cfg.training.som.map_size_obs[1]
                # 使用组合特征名作为键 (需要与 SOM 训练/保存时一致)
                obs_feature_combined_name = "_".join(sorted(list(obs_features_config)))
                if obs_feature_combined_name != target_field_name:
                     if obs_feature_combined_name in low_dim_data_paths: # 检查路径字典中是否存在
                         input_feature_info[obs_feature_combined_name] = {'num_nodes': num_obs_nodes}
                     else:
                          logger.warning(f"配置了使用观测特征，但未在 low_dim_data_paths 中找到组合特征 '{obs_feature_combined_name}' 的路径。")
                          # 可能需要移除 obs_feature_combined_name 从 feature_names_ordered
                # 如果需要处理每个观测特征单独的 SOM，逻辑会更复杂
            elif obs_features_config:
                logger.warning(f"预期 observation_features 是列表，但得到 {type(obs_features_config)}。按单一特征处理: {obs_features_config}")


        logger.info(f"构建的 input_feature_info: {input_feature_info}")

        # --- 重新验证和排序 feature_names_ordered ---
        current_features_in_info = list(input_feature_info.keys())
        # 只保留那些信息和路径都存在的特征
        valid_feature_names = [f for f in feature_names_ordered if f in current_features_in_info and f in low_dim_data_paths]
        if target_field_name not in valid_feature_names:
             logger.error(f"目标特征 '{target_field_name}' 的信息或路径缺失，无法继续。")
             return {}
        # 确保目标在第一个
        if valid_feature_names[0] != target_field_name:
             valid_feature_names.remove(target_field_name)
             valid_feature_names.insert(0, target_field_name)
        feature_names_ordered = valid_feature_names # 更新为最终有效的、有序的特征列表
        num_input_features = len(feature_names_ordered)
        logger.info(f"最终有效的 LSTM 输入特征顺序: {feature_names_ordered} (共 {num_input_features} 个)")

        if num_input_features == 0:
            logger.error("没有有效的输入特征可供 LSTM 使用。")
            return {}

    except AttributeError as e:
        logger.error(f"访问配置错误 (例如 cfg.training.som...). 检查配置结构: {e}", exc_info=True)
        return {}
    except Exception as e:
        logger.error(f"构建 input_feature_info 时出错: {e}", exc_info=True)
        return {}

    # --- 2. 加载、转换并合并低维 BMU 数据 ---
    logger.info("加载、转换并合并 BMU 数据...")
    low_dim_data: Dict[str, np.ndarray] = {} # 存储合并后的数据: {split: (T, num_features)}

    try:
        # --- 获取必要的地图宽度 ---
        map_width_target = cfg.training.som.map_size_sta[1]
        map_size_obs_cfg = cfg.training.som.get("map_size_obs", cfg.training.som.map_size_sta) # 回退到 sta 大小
        map_width_obs = map_size_obs_cfg[1]
        feature_to_map_width = {target_field_name: map_width_target}
        for f in feature_names_ordered:
            if f != target_field_name:
                # 假设所有非目标特征都是观测特征，使用 map_width_obs
                feature_to_map_width[f] = map_width_obs
        logger.info(f"将使用的特征到地图宽度映射: {feature_to_map_width}")

        # --- 循环处理 train, val, test ---
        for split in ["train", "val", "test"]:
            split_data_list = [] # 存储当前 split 的所有特征的 1D (T,) 数组
            has_split_data = False
            current_split_length = -1 # 当前 split 的预期时间长度 T

            # --- 循环处理每个需要的特征 ---
            for feature in feature_names_ordered:
                path_info = low_dim_data_paths.get(feature, {}).get(split)
                path_to_load = None
                if isinstance(path_info, dict):
                    bmu_positions_path = path_info.get('positions')
                    if bmu_positions_path and os.path.exists(bmu_positions_path):
                        path_to_load = bmu_positions_path
                    else:
                        logger.warning(f"  特征 '{feature}', 分割 '{split}': 未找到或无效的 'positions' 路径: {path_info}")
                # 可以选择性地添加对 path_info 是字符串的处理

                if path_to_load:
                    try:
                        # 1. 加载数据
                        loaded_data = np.load(path_to_load)
                        logger.debug(f"  加载 {split} - 特征 '{feature}' from: {path_to_load}, 原始形状: {loaded_data.shape}")

                        # 2. 转换数据为 1D 线性索引 (如果需要)
                        data_1d = None
                        if loaded_data.ndim == 2 and loaded_data.shape[1] == 2:
                            map_width = feature_to_map_width.get(feature)
                            if map_width is None:
                                raise ValueError(f"无法找到特征 '{feature}' 的地图宽度进行转换。")
                            data_1d = convert_grid_to_linear(loaded_data, map_width) # Shape (T,)
                            logger.debug(f"  转换后 1D 索引形状: {data_1d.shape}")
                        elif loaded_data.ndim == 1:
                            logger.debug(f"  特征 '{feature}' 文件已是 1D，假设为有效索引。")
                            data_1d = loaded_data # Shape (T,)
                        else:
                            raise ValueError(f"BMU 文件 {path_to_load} 的形状 {loaded_data.shape} 不可识别。需要 (T, 2) 或 (T,).")

                        has_split_data = True # 标记此 split 有数据

                        # 3. 长度检查
                        if current_split_length == -1: # 这是此 split 的第一个特征
                            current_split_length = len(data_1d)
                            logger.debug(f"  设置 {split} 分割的预期长度为: {current_split_length}")
                        elif len(data_1d) != current_split_length:
                            raise ValueError(f"序列长度不一致! {split} - 特征 '{feature}' ({len(data_1d)}) != 预期 ({current_split_length})")

                        # 4. 添加到列表
                        split_data_list.append(data_1d) # 添加 (T,) 数组

                    # --- Error Handling ---
                    except FileNotFoundError:
                        logger.error(f"  文件未找到: {path_to_load}")
                        has_split_data = False; break # 停止处理此 split
                    except ValueError as ve:
                        logger.error(f"  处理 BMU 文件 {path_to_load} 时出错: {ve}")
                        has_split_data = False; break # 停止处理此 split
                    except Exception as load_err:
                        logger.error(f"  加载或转换 BMU 文件 {path_to_load} 失败: {load_err}", exc_info=True)
                        has_split_data = False; break # 停止处理此 split
                else:
                    # 如果某个必需特征的路径找不到
                    logger.error(f"  未找到 {split} 数据 for 必需特征 '{feature}'。无法继续处理此分割。")
                    has_split_data = False
                    break # 停止处理此 split

            # --- 合并当前 split 的数据 ---
            if has_split_data and len(split_data_list) == num_input_features:
                # *** 核心合并步骤 ***
                # split_data_list 是 [(T,), (T,), ...] 列表
                # 使用 np.stack(..., axis=-1) 或 np.column_stack(...)
                try:
                    combined_split_data = np.stack(split_data_list, axis=-1)
                    # 或者: combined_split_data = np.column_stack(split_data_list)

                    # --- >>> 验证形状 <<< ---
                    if combined_split_data.shape != (current_split_length, num_input_features):
                         logger.error(f"合并 {split} 数据后的形状不正确: {combined_split_data.shape}，预期: ({current_split_length}, {num_input_features})")
                         #可以选择继续处理下一个 split 或直接返回错误
                         continue # 跳过这个错误的 split
                    logger.info(f"  合并后 {split} 数据形状: {combined_split_data.shape}") #<--- 检查这个日志输出!
                    low_dim_data[split] = combined_split_data
                except Exception as stack_err:
                    logger.error(f"堆叠 {split} 分割的特征数据时出错: {stack_err}", exc_info=True)
                    # 跳过这个 split

            elif has_split_data: # 加载了部分但非全部特征
                logger.error(f"未能加载 {split} 的所有必需特征数据 ({len(split_data_list)}/{num_input_features})，跳过此分割。")

            # 重置长度以便下一个 split 重新确定
            current_split_length = -1

    except AttributeError as e:
         logger.error(f"访问配置错误 (可能在获取地图宽度时): {e}", exc_info=True)
         return {}
    except Exception as e:
        logger.error(f"加载、转换或合并 LSTM 输入数据时发生意外错误: {e}", exc_info=True)
        return {}

    # --- 检查是否有训练数据 ---
    if 'train' not in low_dim_data or low_dim_data['train'].size == 0:
        logger.error("缺少合并后的训练数据或训练数据为空，无法训练 LSTM。")
        return results # results['success'] 默认为 False

    # --- 3. 准备 LSTM 序列 ---
    lstm_cfg = config.model.prediction.lstm
    seq_len = lstm_cfg.get("sequence_length", 10)
    logger.info(f"为 LSTM 创建序列，输入序列长度: {seq_len}...")

    # 确定目标特征在堆叠数据中的索引 (按约定是 0)
    target_feature_index = 0
    if feature_names_ordered[target_feature_index] != target_field_name:
        # 这个错误理论上不应发生，因为前面排序了
        logger.error(f"逻辑错误：目标特征 '{target_field_name}' 未按预期放在第 {target_feature_index} 列！顺序: {feature_names_ordered}")
        return {}

    X_seq: Dict[str, np.ndarray] = {}
    y_seq: Dict[str, np.ndarray] = {} # 目标是 Target BMU index (int)

    for split, stacked_data in low_dim_data.items():
        # stacked_data shape: (T, num_input_features)
        logger.debug(f"为 split '{split}' 创建序列，输入数据形状: {stacked_data.shape}")

        if len(stacked_data) >= seq_len + 1: # 确保至少有 seq_len+1 个点来创建一个序列
            # target_data shape: (T,)
            target_data = stacked_data[:, target_feature_index].astype(int) # 确保目标是整数索引

            X_seq_split, y_seq_split = create_sequences(
                X=stacked_data,          # (T, num_features)
                y=target_data,           # (T,)
                sequence_length=seq_len
            )

            # 验证 create_sequences 的输出形状
            expected_x_shape_suffix = (seq_len, num_input_features)
            if X_seq_split.ndim != 3 or X_seq_split.shape[1:] != expected_x_shape_suffix:
                logger.error(f"'{split}' 的 X_seq 形状不正确: {X_seq_split.shape} (预期后缀: {expected_x_shape_suffix})")
                continue # 跳过这个 split
            if y_seq_split.ndim != 1:
                logger.error(f"'{split}' 的 y_seq 形状不正确: {y_seq_split.shape} (预期: (N,))")
                continue

            X_seq[split] = X_seq_split
            y_seq[split] = y_seq_split
            logger.info(f"  {split} 序列创建完成: X={X_seq[split].shape}, y={y_seq[split].shape}") #<--- 检查 X 的最后一个维度是否为 num_input_features
        else:
            logger.warning(f"  {split} 数据长度不足 ({len(stacked_data)})，无法创建长度为 {seq_len} 的序列。")

    # ... (后续检查 'train' 是否有序列) ...
    if 'train' not in X_seq or X_seq['train'].size == 0:
         logger.error("未能成功创建训练序列。检查数据长度和序列长度设置。")
         return {} # 返回空字典表示失败

    # --- 4. 跳过标准化 ---
    logger.info("BMU 索引数据，跳过标准化缩放。")

    # --- 5. 初始化和训练 LSTM 模型 ---
    # ... (准备模型参数，调用 lstm_model.fit) ...
    # 确保 LSTMPredictionModel 初始化时 num_embeddings_list 和 embedding_dims_list 的长度
    # 与 num_input_features 匹配。
    num_embeddings_list = [input_feature_info[f]['num_nodes'] for f in feature_names_ordered]
    default_emb_dim = lstm_cfg.get("embedding_dim", 32)
    embedding_dims_list = [default_emb_dim] * num_input_features
    target_som_num_nodes = input_feature_info[target_field_name]['num_nodes']

    # 检查列表长度是否匹配
    if len(num_embeddings_list) != num_input_features or len(embedding_dims_list) != num_input_features:
        logger.error(f"嵌入参数列表长度 ({len(num_embeddings_list)}, {len(embedding_dims_list)}) 与输入特征数 ({num_input_features}) 不匹配！")
        return {}

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
        epochs=lstm_cfg.get("epochs", 100),
        batch_size=lstm_cfg.get("batch_size", 32),
        # --- MODIFICATION END ---
        learning_rate=lstm_cfg.get("lr", 0.001),
        patience=lstm_cfg.get("patience", 10),
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
    # 确保 predict 方法能处理形状为 (N, seq_len, num_input_features) 的输入
    logger.info("使用训练好的 LSTM 预测 *目标* BMU 索引序列...")
    results['predicted_target_low_dim_paths'] = {}
    os.makedirs(prediction_save_dir, exist_ok=True)

    for split in ["train", "val", "test"]:
        if split not in X_seq:
            logger.warning(f"跳过 {split} 分割的 LSTM 预测，因为没有输入序列。")
            continue

        logger.info(f"  预测 {split} 分割...")
        try:
            predicted_target_indices = lstm_model.predict(X_seq[split]) # 输入 (N, seq_len, num_features)
            logger.info(f"  预测的 {split} 目标 BMU 索引形状: {predicted_target_indices.shape}") # 输出 (N,)

            # ... (保存预测结果) ...
            pred_save_path = os.path.join(prediction_save_dir, prediction_filename_pattern.format(split=split))
            np.save(pred_save_path, predicted_target_indices)
            logger.info(f"  预测的 {split} *目标* BMU 索引序列已保存到: {pred_save_path}")
            results['predicted_target_low_dim_paths'][split] = pred_save_path

        except Exception as e:
            logger.error(f"  预测 {split} 目标 BMU 索引序列失败: {e}", exc_info=True)


    total_run_time = time.time() - start_time
    logger.info(f"LSTM 训练和预测完成，总用时：{total_run_time:.2f}秒")

    return results