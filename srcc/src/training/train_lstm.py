# src/training/train_lstm.py (重构后的结构)

import os
import sys
import numpy as np
import torch
import logging
import time
import hydra
from omegaconf import DictConfig, OmegaConf
import pickle
from typing import Optional, Dict, Any, Tuple

# --- 项目设置 ---
# ... (如果需要，包含项目根路径) ...

# --- 集中导入 ---
from src.utils.hydra_config import DrprConfig
from src.utils.logger import setup_logger # 假设存在 setup_logger
from src.prediction_models.lstm_pytorch import LSTMPredictionModel # 你的 LSTM 实现
from src.data_processing.sequence_utils import create_sequences
from src.utils.data_loader import load_processed_data # 用于加载低维数据
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

def train_and_predict_lstm(
    cfg: DictConfig,
    low_dim_data_paths: Dict[str, str], # 类似 {'train': 'path/to/train.npy', ...} 的字典
    model_save_dir: str,
    scaler_save_path: str,
    prediction_save_dir: str,
    prediction_filename_pattern: str = "predicted_low_dim_{split}.npy" # 保存预测的文件名模式
) -> Dict[str, Any]:
    """
    在低维数据上训练 LSTM 模型并预测序列。

    Args:
        cfg: Hydra 配置对象。
        low_dim_data_paths: 包含低维训练、验证、测试数据 (.npy) 路径的字典。
        model_save_dir: 保存训练好的 LSTM 模型的目录。
        scaler_save_path: 保存用于 LSTM 输入的 StandardScaler 的路径。
        prediction_save_dir: 保存预测出的低维序列的目录。
        prediction_filename_pattern: 预测的文件名模式。

    Returns:
        包含结果（如模型路径、scaler 路径、预测路径）的字典。
    """
    config = DrprConfig.from_hydra_config(cfg)
    results = {}
    start_time = time.time()

    # --- 1. 加载低维数据 ---
    logger.info("加载用于 LSTM 训练/预测的低维数据...")
    low_dim_data = {}
    for split in ["train", "val", "test"]:
        path = low_dim_data_paths.get(split)
        if path and os.path.exists(path):
            try:
                low_dim_data[split] = np.load(path)
                # 确保数据是 2D (n_samples, n_features_low)
                if low_dim_data[split].ndim == 1:
                   low_dim_data[split] = low_dim_data[split].reshape(-1, 1)
                elif low_dim_data[split].ndim > 2:
                    # 尝试展平除时间外的维度
                    logger.warning(f"{split} 低维数据维度 > 2 ({low_dim_data[split].shape})，将展平非时间维度。")
                    T = low_dim_data[split].shape[0]
                    low_dim_data[split] = low_dim_data[split].reshape(T, -1)

                logger.info(f"  加载 {split} 数据，形状: {low_dim_data[split].shape}")
            except Exception as e:
                logger.error(f"  加载 {split} 低维数据失败: {path} - {e}")
                return results # 如果必需数据缺失则尽早失败
        else:
            logger.warning(f"  未找到 {split} 的低维数据路径: {path}")
            # 判断 val/test 缺失是否可接受
            if split == 'train': return results

    if 'train' not in low_dim_data:
        logger.error("缺少训练数据，无法训练 LSTM。")
        return results

    # --- 2. 准备 LSTM 序列 ---
    lstm_cfg = config.model.prediction.lstm
    seq_len = lstm_cfg.get("sequence_length", 10)
    logger.info(f"为 LSTM 创建序列，长度: {seq_len}...")

    X_seq, y_seq = {}, {}
    for split, data in low_dim_data.items():
        if data is not None and len(data) > seq_len:
             # 预测下一个时间步的低维向量
            X_seq[split], y_seq[split] = create_sequences(data, data, seq_len)
            logger.info(f"  {split} 序列形状: X={X_seq[split].shape}, y={y_seq[split].shape}")
        else:
             logger.warning(f"  {split} 数据长度不足 ({len(data) if data is not None else 0})，无法创建序列。")

    if 'train' not in X_seq or X_seq['train'].size == 0:
         logger.error("未能成功创建训练序列。")
         return results

    # --- 3. 缩放 LSTM 输入数据 ---
    # LSTM 输入是序列 X_seq，目标是 y_seq
    logger.info("缩放 LSTM 的输入序列数据...")
    scaler_lstm = StandardScaler()
    n_features_low = X_seq['train'].shape[2] # 低维特征数量

    # 在训练序列上拟合 Scaler
    X_train_seq_flat = X_seq['train'].reshape(-1, n_features_low)
    scaler_lstm.fit(X_train_seq_flat)
    logger.info("  LSTM 输入 Scaler 拟合完成。")

    # 应用 Scaler 到所有 X_seq
    X_seq_scaled = {}
    for split, data in X_seq.items():
        num_seq, seq_len_split, _ = data.shape
        data_flat = data.reshape(-1, n_features_low)
        scaled_flat = scaler_lstm.transform(data_flat)
        X_seq_scaled[split] = scaled_flat.reshape(num_seq, seq_len_split, n_features_low)
        logger.info(f"  已缩放 {split} X_seq，形状: {X_seq_scaled[split].shape}")

    # 应用 Scaler 到所有 y_seq (目标也是低维向量)
    y_seq_scaled = {}
    for split, data in y_seq.items():
        # y_seq 已经是 (num_samples, n_features_low)
        y_seq_scaled[split] = scaler_lstm.transform(data)
        logger.info(f"  已缩放 {split} y_seq，形状: {y_seq_scaled[split].shape}")

    # --- 4. 保存 LSTM Scaler ---
    os.makedirs(os.path.dirname(scaler_save_path), exist_ok=True)
    try:
        with open(scaler_save_path, 'wb') as f:
            pickle.dump(scaler_lstm, f)
        logger.info(f"LSTM 输入 Scaler 已保存到: {scaler_save_path}")
        results['scaler_path'] = scaler_save_path
    except Exception as e:
        logger.error(f"保存 LSTM Scaler 失败: {e}")
        # 判断这是否是关键错误 - 很可能是
        return results

    # --- 5. 训练 LSTM 模型 ---
    logger.info("初始化并训练 LSTM 模型...")
    lstm_model = LSTMPredictionModel(
        # --- 从 cfg 读取 LSTM 特定参数 ---
        input_size=n_features_low,
        output_size=n_features_low, # 预测下一个低维向量
        hidden_size=lstm_cfg.get("hidden_size", 64), # 使用 cfg 中的名字
        num_layers=lstm_cfg.get("num_layers", 2),
        dropout=lstm_cfg.get("dropout", 0.1),
        learning_rate=cfg.training.optimizer.get("learning_rate", 0.001), # 使用训练优化器的学习率
        epochs=cfg.training.get("epochs", 100), # 使用全局 epochs
        batch_size=lstm_cfg.get("batch_size", 32),
        sequence_length=seq_len, # LSTM 内部可能需要
        patience=cfg.training.early_stopping.get("patience", 10), # 使用全局 patience
        random_seed=cfg.training.random_seed,
        device=get_device_from_config(cfg) # 使用工具函数获取设备
    )

    # 训练 (使用缩放后的数据)
    # LSTMPredictionModel 中的 Fit 方法应在内部处理训练/验证集划分（如果需要）
    if 'val' in X_seq_scaled and 'val' in y_seq_scaled:
         # 如果有验证数据且 LSTMPredictionModel 支持，则传递验证数据
         lstm_model.fit(X_seq_scaled['train'], y_seq_scaled['train'],
                        X_val=X_seq_scaled['val'], y_val=y_seq_scaled['val'])
    else:
         # 否则，仅在训练数据上拟合
         lstm_model.fit(X_seq_scaled['train'], y_seq_scaled['train'])

    # --- 6. 保存 LSTM 模型 ---
    os.makedirs(model_save_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    lstm_model_path = os.path.join(model_save_dir, f"lstm_model_{timestamp}.pt")
    latest_path = os.path.join(model_save_dir, "latest.pt")
    try:
        lstm_model.save(lstm_model_path) # 假设存在 .save 方法
        logger.info(f"LSTM 模型已保存到: {lstm_model_path}")
        import shutil
        shutil.copy2(lstm_model_path, latest_path)
        logger.info(f"LSTM 模型副本已保存为 {latest_path} (用于预测)")
        results['model_path'] = latest_path # 返回最新模型的路径
    except Exception as e:
        logger.error(f"保存 LSTM 模型失败: {e}")
        return results

    # --- 7. 使用 LSTM 模型预测 ---
    logger.info("使用训练好的 LSTM 预测低维序列...")
    results['predicted_low_dim_paths'] = {}
    os.makedirs(prediction_save_dir, exist_ok=True)

    for split in ["train", "val", "test"]:
        if split not in X_seq_scaled:
            logger.warning(f"跳过 {split} 分割的 LSTM 预测，因为没有输入序列。")
            continue

        logger.info(f"  预测 {split} 分割...")
        try:
            # 使用模型的 predict 方法（假设它处理好了 device 和 no_grad）
            predicted_scaled = lstm_model.predict(X_seq_scaled[split])

            # 反向缩放预测结果
            predicted_low_dim = scaler_lstm.inverse_transform(predicted_scaled)
            logger.info(f"  预测的 {split} 低维数据形状 (反向缩放后): {predicted_low_dim.shape}")

            # 保存预测结果
            pred_save_path = os.path.join(prediction_save_dir, prediction_filename_pattern.format(split=split))
            np.save(pred_save_path, predicted_low_dim)
            logger.info(f"  预测的 {split} 低维序列已保存到: {pred_save_path}")
            results['predicted_low_dim_paths'][split] = pred_save_path

        except Exception as e:
            logger.error(f"  预测 {split} 低维序列失败: {e}", exc_info=True)

    total_run_time = time.time() - start_time
    logger.info(f"LSTM 训练和预测完成，总用时：{total_run_time:.2f}秒")

    return results

# --- 如果你想独立运行此脚本，可以添加一个 Hydra 入口点 (可选) ---
# @hydra.main(version_base=None, config_path="../../conf", config_name="config")
# def hydra_main(cfg: DictConfig) -> None:
#     # 这部分需要逻辑来确定使用哪个 low_dim_data_paths，
#     # 可能基于命令行覆盖或其他配置设置。
#     # 对于 pipeline 集成，`train_and_predict_lstm` 函数就足够了。
#     pass

# if __name__ == "__main__":
#     hydra_main()