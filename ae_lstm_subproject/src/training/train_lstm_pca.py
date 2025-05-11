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
from sklearn.preprocessing import StandardScaler
import joblib

# --- 集中导入 ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils.hydra_config import DrprConfig
from src.prediction_models.lstm_pca import LSTMPredictionModel # 使用你定义的模型类
from src.data_processing.sequence_utils import create_sequences
from src.utils.model_utils import get_device_from_config
# --- >>> 新增：可能需要数据加载器来加载外部特征 <<< ---
from src.utils.data_loader import load_processed_data # 复用或创建新的加载器

logger = logging.getLogger(__name__)



def train_and_predict_lstm_pca(
    cfg: DictConfig,
    input_features: List[str], # PCA input features
    target_feature: str,
    low_dim_data_paths: Dict[str, Dict[str, str]], # PCA data paths
    pca_n_components: Dict[str, int],
    model_save_subdir: str = "lstm_pca_model",
    prediction_save_subdir: str = "lstm_pca_predictions",
    prediction_filename_pattern: str = "predicted_pca_{target}_lstm_{split}.npy"
) -> Dict[str, Any]:
    """
    使用 PCA 成分和可选的外部特征训练 LSTM 模型，并预测目标特征的 PCA 成分。
    包含标准化步骤。
    # ... (其余文档字符串) ...
    """
    config = DrprConfig.from_hydra_config(cfg) # Optional
    results = {'success': False, 'predicted_target_low_dim_paths': {}}
    start_time = time.time()
    device = get_device_from_config(cfg)
    lstm_cfg = cfg.model.prediction.lstm_pca # 获取 LSTM(PCA) 的配置

    try:
        # --- 1. 获取 Hydra 输出目录并构建保存路径 (同前) ---
        hydra_output_dir = os.getcwd(); logger.info(f"Hydra 输出目录: {hydra_output_dir}")
        model_save_fullpath = os.path.join(hydra_output_dir, model_save_subdir)
        prediction_save_fullpath = os.path.join(hydra_output_dir, prediction_save_subdir)
        scaler_save_fullpath = os.path.join(hydra_output_dir, "pca_lstm_scalers")
        os.makedirs(model_save_fullpath, exist_ok=True); os.makedirs(prediction_save_fullpath, exist_ok=True); os.makedirs(scaler_save_fullpath, exist_ok=True)
        logger.info(f"模型将保存在: {model_save_fullpath}")
        logger.info(f"预测将保存在: {prediction_save_fullpath}")
        logger.info(f"Scaler 将保存在: {scaler_save_fullpath}")

        # --- 2. 验证 PCA 输入和确定维度 (同前) ---
        logger.info(f"LSTM (PCA) 输入特征: {input_features}")
        logger.info(f"LSTM (PCA) 目标特征: {target_feature}")
        total_pca_components = 0
        target_output_components = pca_n_components.get(target_feature)
        if target_output_components is None or target_output_components <= 0: logger.error(f"目标特征 '{target_feature}' PCA 组件数无效。"); return results
        for feature in input_features:
             n_comp = pca_n_components.get(feature);
             if n_comp is None or n_comp <= 0: logger.error(f"输入特征 '{feature}' PCA 组件数无效。"); return results
             if feature not in low_dim_data_paths: logger.error(f"未找到输入特征 '{feature}' PCA 数据路径。"); return results
             total_pca_components += n_comp
        if target_feature not in low_dim_data_paths: logger.error(f"未找到目标特征 '{target_feature}' PCA 数据路径。"); return results
        logger.info(f"总输入 PCA 组件数: {total_pca_components}")
        logger.info(f"目标输出 PCA 组件数: {target_output_components}")

        # --- >>> 2.1 获取并验证外部特征 <<< ---
        external_features = list(lstm_cfg.get("external_features", []))
        total_external_dims = 0
        logger.info(f"LSTM 外部输入特征: {external_features}")
        loaded_external_data: Dict[str, Dict[str, np.ndarray]] = {split: {} for split in ['train', 'val', 'test']} # {split: {feature: data}}

        if external_features :
            # --- >>> 实现加载外部特征数据的逻辑 <<< ---
            logger.info("加载外部特征数据 (例如 flow)...")
            processed_dir = config.paths.processed_data_dir # 获取处理数据目录
            for ext_feature in external_features: # 循环处理列表中的每个外部特征
                feature_dims_loaded = None # 用于记录该特征的维度
                for split in ['train', 'val', 'test']:
                    # 构建文件路径
                    ext_feature_path = os.path.join(processed_dir, f"{split}_{ext_feature}_processed.npy")
                    if os.path.exists(ext_feature_path):
                        try:
                            data = np.load(ext_feature_path)
                            # 确保数据是 2D: (T, num_dims)
                            if data.ndim == 1:
                                data = data.reshape(-1, 1) # 转换为 T x 1
                            elif data.ndim != 2:
                                raise ValueError(f"外部特征 {ext_feature} ({split}) 的维度不是 1 或 2: {data.ndim}")

                            # 记录该特征的维度（基于训练集）
                            if split == 'train':
                                feature_dims_loaded = data.shape[1]

                            # 检查时间长度是否匹配 (需要先加载 PCA 数据确定 split_lengths)
                            # --- 移动到 PCA 加载之后检查 ---

                            loaded_external_data[split][ext_feature] = data
                            logger.info(f"  加载 {split} {ext_feature} 数据，形状: {data.shape}")
                        except Exception as e:
                            logger.error(f"加载或处理外部特征文件失败: {ext_feature_path} ({e})", exc_info=True)
                            return results # 加载失败则退出
                    else:
                        logger.error(f"外部特征文件未找到: {ext_feature_path} (特征: {ext_feature})")
                        return results # 文件不存在则退出

                # 累加该特征的维度到总维度
                if feature_dims_loaded is not None:
                     total_external_dims += feature_dims_loaded
                else:
                     # 如果训练集文件加载失败（虽然前面会退出），则无法确定维度
                     logger.error(f"未能确定外部特征 '{ext_feature}' 的维度（训练集未加载？）。")
                     return results

            if total_external_dims > 0:
                 logger.info(f"总外部特征维度: {total_external_dims}")
            elif external_features: # 配置了但未加载成功或维度为0
                 logger.warning("配置了外部特征，但未能成功加载或确定维度。将不使用外部特征。")
                 external_features = [] # 清空列表，后续不再处理
            # --- >>> 加载逻辑结束 <<< ---


        # --- 3. 加载 PCA 数据 ---
        loaded_pca_input_data: Dict[str, List[np.ndarray]] = {'train': [], 'val': [], 'test': []}
        loaded_pca_target_data: Dict[str, Optional[np.ndarray]] = {'train': None, 'val': None, 'test': None}
        split_lengths = {} # 记录每个 split 的时间长度 T (从 PCA 数据确定)
        logger.info("加载 PCA 特征数据...")
        for feature in input_features: # PCA features
            for split in ['train', 'val', 'test']:
                path = low_dim_data_paths.get(feature, {}).get(split)
                if path and os.path.exists(path):
                    try:
                        data = np.load(path)
                        expected_shape = (data.shape[0], pca_n_components[feature])
                        if data.shape != expected_shape: raise ValueError(f"PCA 形状不匹配: {data.shape} vs {expected_shape}")
                        # 记录或验证长度
                        if split not in split_lengths: split_lengths[split] = data.shape[0]
                        elif split_lengths[split] != data.shape[0]: raise ValueError(f"PCA 特征 '{feature}' ({split}) 长度 {data.shape[0]} 与之前长度 {split_lengths[split]} 不匹配")
                        loaded_pca_input_data[split].append(data)
                        logger.debug(f"  加载 {split} {feature} PCA: {data.shape}")
                    except Exception as e: logger.error(f"加载或验证 PCA 文件失败: {path} ({e})"); return results
                else: logger.error(f"PCA 文件未找到: {path}"); return results
        logger.info("加载 PCA 目标数据...")
        for split in ['train', 'val', 'test']:
            path = low_dim_data_paths.get(target_feature, {}).get(split)
            if path and os.path.exists(path):
                 try:
                      data = np.load(path)
                      expected_shape = (data.shape[0], target_output_components)
                      if data.shape != expected_shape: raise ValueError(f"PCA 目标形状不匹配: {data.shape} vs {expected_shape}")
                      if split not in split_lengths: split_lengths[split] = len(data) # 理论上 PCA 输入加载时已记录
                      elif split_lengths[split] != len(data): raise ValueError("PCA Target length mismatch")
                      loaded_pca_target_data[split] = data
                      logger.debug(f"  加载 {split} {target_feature} PCA (目标): {data.shape}")
                 except Exception as e: logger.error(f"加载或验证 PCA 目标文件失败: {path} ({e})"); return results
            else: logger.error(f"PCA 目标文件未找到: {path}"); return results

        # --- >>> 3.2 再次验证外部特征数据长度 <<< ---
        if external_features:
             logger.info("再次验证外部特征数据长度...")
             for split in loaded_external_data.keys():
                  if split in split_lengths: # 仅检查有 PCA 数据的 split
                       expected_len = split_lengths[split]
                       for ext_feature, data in loaded_external_data[split].items():
                            if data.shape[0] != expected_len:
                                 logger.error(f"长度不匹配! 外部特征 '{ext_feature}' ({split}) 长度 {data.shape[0]} 与 PCA 数据长度 {expected_len} 不符。")
                                 return results
                  else:
                       # 如果 PCA 数据没有这个 split，外部特征数据也用不上
                       if loaded_external_data[split]:
                           logger.warning(f"加载了 {split} 的外部特征数据，但没有对应的 PCA 数据，将忽略外部数据。")
                           loaded_external_data[split] = {} # 清空无效的外部数据


        # --- 4. 准备合并数据 (PCA) ---
        lstm_pca_input_X: Dict[str, np.ndarray] = {}
        lstm_pca_target_Y: Dict[str, np.ndarray] = {} # PCA Target remains the same
        for split in ['train', 'val', 'test']:
             if loaded_pca_input_data.get(split) and loaded_pca_target_data.get(split) is not None:
                  try:
                      lstm_pca_input_X[split] = np.concatenate(loaded_pca_input_data[split], axis=1)
                      lstm_pca_target_Y[split] = loaded_pca_target_data[split]
                      logger.info(f"合并后 {split} PCA 输入 X 形状: {lstm_pca_input_X[split].shape}")
                      logger.info(f"{split} PCA 目标 Y 形状: {lstm_pca_target_Y[split].shape}")
                  except ValueError as e: logger.error(f"合并 PCA 数据时出错 ({split}): {e}"); continue
             else: logger.warning(f"跳过 {split} 因为缺少 PCA 输入或目标数据。")
        if 'train' not in lstm_pca_input_X or 'train' not in lstm_pca_target_Y: logger.error("缺少合并后的 PCA 训练数据。"); return results

        # --- 4.1 准备合并数据 (外部特征) ---
        lstm_external_input_X: Dict[str, Optional[np.ndarray]] = {split: None for split in ['train', 'val', 'test']}
        if external_features:
             for split in loaded_external_data.keys():
                  if loaded_external_data[split]: # 如果这个 split 加载了外部特征
                       try:
                           data_to_concat = [loaded_external_data[split][feat] for feat in external_features] # 按配置顺序合并
                           lstm_external_input_X[split] = np.concatenate(data_to_concat, axis=1)
                           # 验证维度是否与 total_external_dims 匹配
                           if lstm_external_input_X[split].shape[1] != total_external_dims:
                                raise ValueError(f"合并后维度 {lstm_external_input_X[split].shape[1]} 与预期 {total_external_dims} 不符")
                           logger.info(f"合并后 {split} 外部特征输入 X 形状: {lstm_external_input_X[split].shape}")
                       except ValueError as e:
                           logger.error(f"合并外部特征时出错 ({split}): {e}"); lstm_external_input_X[split] = None


        # --- 5. 标准化 PCA 成分 & 外部特征 & PCA 目标 ---
        logger.info("对 PCA 成分、外部特征和 PCA 目标进行标准化...")
        scaler_pca_X = StandardScaler()
        scaler_target_Y = StandardScaler()
        scaler_external_X = StandardScaler() if external_features and total_external_dims > 0 else None

        # 拟合 Scaler (仅在训练集)
        try:
            logger.info("拟合 PCA 输入 (X) Scaler..."); scaler_pca_X.fit(lstm_pca_input_X['train'])
            logger.info("拟合 PCA 目标 (Y) Scaler..."); scaler_target_Y.fit(lstm_pca_target_Y['train'])
            if scaler_external_X and lstm_external_input_X.get('train') is not None:
                logger.info("拟合外部特征 (X) Scaler..."); scaler_external_X.fit(lstm_external_input_X['train'])
        except Exception as fit_err: logger.error(f"拟合 Scaler 时出错: {fit_err}"); return results

        # 保存 Scaler
        joblib.dump(scaler_pca_X, os.path.join(scaler_save_fullpath, f"scaler_pca_X_{'_'.join(input_features)}.pkl"))
        joblib.dump(scaler_target_Y, os.path.join(scaler_save_fullpath, f"scaler_pca_Y_{target_feature}.pkl"))
        if scaler_external_X: joblib.dump(scaler_external_X, os.path.join(scaler_save_fullpath, f"scaler_external_X_{'_'.join(external_features)}.pkl"))
        logger.info(f"Scalers 已保存到: {scaler_save_fullpath}")
        results['scaler_pca_X_path'] = os.path.join(scaler_save_fullpath, f"scaler_pca_X_{'_'.join(input_features)}.pkl")
        results['scaler_target_Y_path'] = os.path.join(scaler_save_fullpath, f"scaler_pca_Y_{target_feature}.pkl")
        if scaler_external_X: results['scaler_external_X_path'] = os.path.join(scaler_save_fullpath, f"scaler_external_X_{'_'.join(external_features)}.pkl")


        # 应用 Scaler
        lstm_pca_input_X_scaled = {}
        lstm_target_Y_scaled = {}
        lstm_external_input_X_scaled = {}
        logger.info("对所有分割应用 Scalers...")
        for split in lstm_pca_input_X.keys():
            if split in lstm_pca_target_Y:
                try:
                    lstm_pca_input_X_scaled[split] = scaler_pca_X.transform(lstm_pca_input_X[split])
                    lstm_target_Y_scaled[split] = scaler_target_Y.transform(lstm_pca_target_Y[split])
                    if scaler_external_X and lstm_external_input_X.get(split) is not None:
                         lstm_external_input_X_scaled[split] = scaler_external_X.transform(lstm_external_input_X[split])
                    logger.debug(f"完成 {split} 的标准化")
                except Exception as transform_err: logger.error(f"应用 Scaler 到 {split} 时出错: {transform_err}"); continue


        # --- 5.1 拼接所有标准化的输入 ---
        lstm_final_input_X_scaled : Dict[str, np.ndarray] = {}
        final_input_size = total_pca_components + total_external_dims # 使用更新后的外部维度
        logger.info(f"最终 LSTM 输入维度 (PCA + 外部): {final_input_size}")

        for split in lstm_pca_input_X_scaled.keys(): # 只处理 PCA X 标准化成功的 split
            if external_features:
                 if split in lstm_external_input_X_scaled and lstm_external_input_X_scaled[split] is not None:
                      try:
                           lstm_final_input_X_scaled[split] = np.concatenate(
                               [lstm_pca_input_X_scaled[split], lstm_external_input_X_scaled[split]], axis=1
                           )
                           if lstm_final_input_X_scaled[split].shape[1] != final_input_size: raise ValueError(f"维度不匹配")
                           logger.info(f"最终 {split} 输入 X 形状: {lstm_final_input_X_scaled[split].shape}")
                      except ValueError as e: logger.error(f"拼接最终输入时出错 ({split}): {e}"); continue
                 else: logger.warning(f"跳过 {split} 因为缺少标准化的外部特征。"); continue
            else: # 没有外部特征
                 lstm_final_input_X_scaled[split] = lstm_pca_input_X_scaled[split]
                 logger.info(f"最终 {split} 输入 X 形状 (仅PCA): {lstm_final_input_X_scaled[split].shape}")


        # --- 6. 创建序列 (使用最终拼接和标准化的输入数据) ---
        seq_len = lstm_cfg.get("sequence_length", 10)
        logger.info(f"使用最终标准化数据创建序列，序列长度: {seq_len}...")
        X_seq_final_scaled: Dict[str, np.ndarray] = {}
        y_seq_final_scaled: Dict[str, np.ndarray] = {} # 目标 Y 不变

        for split in lstm_final_input_X_scaled.keys():
            if split in lstm_target_Y_scaled: # 目标 Y 也要存在 (且已标准化)
                input_data_split_final = lstm_final_input_X_scaled[split]
                target_data_split_scaled = lstm_target_Y_scaled[split] # 使用标准化的 PCA 目标
                if len(input_data_split_final) >= seq_len + 1:
                    X_seq_split_final, y_seq_split_final = create_sequences(
                        X=input_data_split_final, y=target_data_split_scaled, sequence_length=seq_len
                    )
                    if X_seq_split_final is not None and y_seq_split_final is not None:
                        X_seq_final_scaled[split] = X_seq_split_final; y_seq_final_scaled[split] = y_seq_split_final
                        logger.info(f"  {split} 最终序列创建完成: X={X_seq_final_scaled[split].shape}, y={y_seq_final_scaled[split].shape}")
                    else: logger.error(f"  为 {split} 创建最终序列失败。")
                else: logger.warning(f"  {split} 最终数据长度不足。")
        if 'train' not in X_seq_final_scaled or X_seq_final_scaled['train'].size == 0: logger.error("未能成功创建最终训练序列。"); return results

        # 更新用于训练/预测的最终变量
        X_train_final = X_seq_final_scaled['train']
        y_train_final = y_seq_final_scaled['train']
        X_val_final = X_seq_final_scaled.get('val')
        y_val_final = y_seq_final_scaled.get('val')

        # --- 7. 初始化 LSTM 模型 (使用最终的 input_size) ---
        logger.info("初始化 LSTMPredictionModel (PCA + 外部)...")
        try:
            model = LSTMPredictionModel(
                input_size=final_input_size, output_size=target_output_components,
                hidden_size=lstm_cfg.get("hidden_size", 64), num_layers=lstm_cfg.get("num_layers", 2),
                dropout=lstm_cfg.get("dropout", 0.1),
                epochs=lstm_cfg.get("num_epochs", 50), batch_size=lstm_cfg.get("batch_size", 32),
                learning_rate=lstm_cfg.get("learning_rate", 0.001), patience=lstm_cfg.get("patience", 10),
                sequence_length=seq_len, random_seed=cfg.training.get('random_seed'), device=device
            )
        except Exception as model_init_err: logger.error(f"初始化 LSTMPredictionModel 出错: {model_init_err}"); return results

        # --- 8. 训练模型 (使用最终序列数据) ---
        logger.info("开始训练 LSTM (PCA + 外部) 模型...")
        model.fit(
            X_train=X_train_final, y_train=y_train_final,
            X_val=X_val_final, y_val=y_val_final
        )

        # --- 9. 保存 LSTM 模型 (同前) ---
        # ... (保存模型和 latest 链接的逻辑) ...
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        input_desc = f"pca{'_'.join(input_features)}_ext{'_'.join(external_features)}" if external_features else f"pca{'_'.join(input_features)}"
        lstm_model_filename = f"lstm_model_{input_desc}_to_{target_feature}_{timestamp}.pt"
        lstm_model_path = os.path.join(model_save_fullpath, lstm_model_filename)
        latest_path = os.path.join(model_save_fullpath, "latest_pca_lstm.pt")
        best_model_path_to_return = None
        try:
            model.save(lstm_model_path); logger.info(f"模型已保存到: {lstm_model_path}")
            shutil.copy2(lstm_model_path, latest_path); logger.info(f"模型副本已保存为 {latest_path}")
            best_model_path_to_return = latest_path
        except Exception as e: logger.error(f"保存模型失败: {e}"); best_model_path_to_return = lstm_model_path
        if best_model_path_to_return: results['model_path'] = best_model_path_to_return
        else: logger.warning("没有最终模型路径可记录。")


        # --- 10. 使用 LSTM 模型预测 ---
        logger.info("使用训练好的 LSTM (PCA + 外部) 预测 *目标* PCA 组件序列...")
        results['predicted_target_low_dim_paths'] = {}

        model_to_predict_with = model # 直接使用训练后的模型

        for split in ["train", "val", "test"]:
            if split not in X_seq_final_scaled: # 使用最终的序列数据进行预测
                logger.warning(f"跳过 {split} 预测，因为没有最终输入序列。")
                continue

            X_pred_input_final_scaled = X_seq_final_scaled[split] # 最终标准化序列输入
            logger.info(f"  预测 {split} 分割 (输入形状: {X_pred_input_final_scaled.shape})...")
            try:
                # 预测得到的是标准化尺度的 PCA 成分
                predicted_target_components_scaled = model_to_predict_with.predict(X_pred_input_final_scaled)
                logger.info(f"  预测的 {split} 标准化目标 PCA 组件形状: {predicted_target_components_scaled.shape}")

                # --- 对预测结果进行逆标准化 (使用 scaler_target_Y) ---
                try:
                    predicted_target_components = scaler_target_Y.inverse_transform(predicted_target_components_scaled)
                    logger.info(f"  逆标准化后的 {split} 目标 PCA 组件形状: {predicted_target_components.shape}")
                except Exception as inv_err:
                    logger.error(f"对 {split} 预测结果执行 inverse_transform 失败: {inv_err}");
                    predicted_target_components = predicted_target_components_scaled
                    logger.warning(f"将保存标准化的预测结果 ({split})。")

                # --- 保存逆标准化后的 (原始尺度的) 预测结果 ---
                filename = prediction_filename_pattern.format(target=target_feature, split=split)
                save_target = os.path.join(prediction_save_fullpath, filename)
                np.save(save_target, predicted_target_components) # 保存原始尺度的成分
                results['predicted_target_low_dim_paths'][split] = save_target
                logger.info(f"  预测的 {split} *目标* PCA 组件序列 (原始尺度) 已保存到: {save_target}")

            except Exception as e: logger.error(f"  预测 {split} 目标 PCA 组件序列失败: {e}")


        # --- 设置最终成功状态 ---
        if results['predicted_target_low_dim_paths']: results['success'] = True

    except FileNotFoundError as e: logger.error(f"文件未找到错误: {e}"); results['success'] = False
    except ValueError as e: logger.error(f"值错误: {e}"); results['success'] = False
    except Exception as e: logger.error(f"训练或预测时发生意外错误: {e}"); results['success'] = False
    finally:
        end_time = time.time()
        logger.info(f"LSTM (PCA + 外部) 训练和预测完成，总用时：{end_time - start_time:.2f}秒，成功: {results.get('success')}")

    return results