import logging
import numpy as np
from typing import Tuple


logger = logging.getLogger(__name__)

# def create_sequences(X: np.ndarray, y: np.ndarray, sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
#     """创建LSTM训练/预测序列。"""
#     logger = logging.getLogger(__name__)
#     X_seq, y_seq = [], []
#     if len(X) <= sequence_length:
#          logger.error(f"数据长度 ({len(X)}) 对于序列长度 ({sequence_length}) 不足。")
#          return np.array([]), np.array([]) # 返回空数组

#     for i in range(len(X) - sequence_length):
#         X_seq.append(X[i:i + sequence_length])
#         y_seq.append(y[i + sequence_length]) # 预测序列后的步骤

#     logger.info(f"创建的序列: X_seq 形状 {np.array(X_seq).shape}, y_seq 形状 {np.array(y_seq).shape}")
#     return np.array(X_seq), np.array(y_seq)

def create_sequences(X: np.ndarray, y: np.ndarray, sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
    logger = logging.getLogger(__name__) # 建议在函数内获取 logger
    X_seq, y_seq = [], []
    num_samples_possible = len(X) - sequence_length

    if num_samples_possible < 1: # 至少需要一个样本才能创建序列
        logger.warning(f"数据长度 ({len(X)}) 对于序列长度 ({sequence_length}) 不足。无法创建序列。")
        # 返回正确维度的空数组
        num_input_features = X.shape[1] if X.ndim > 1 else 1
        num_output_features = y.shape[1] if y.ndim > 1 else 1
        return np.empty((0, sequence_length, num_input_features)), np.empty((0, num_output_features))

    for i in range(num_samples_possible):
        X_seq.append(X[i:i + sequence_length])
        # --- >>> 修改这里处理 y <<< ---
        if y.ndim == 1:
            y_seq.append(y[i + sequence_length]) # 原始逻辑 (y是1D)
        elif y.ndim == 2:
            y_seq.append(y[i + sequence_length, :]) # y是2D，取整行 (T, n_comp) -> (n_comp,)
        else:
            raise ValueError(f"输入 y 的维度不正确: {y.ndim}，需要 1 或 2。")
        # --- >>> 修改结束 <<< ---

    X_seq_np = np.array(X_seq)
    y_seq_np = np.array(y_seq)

    # 验证 y_seq_np 的最终形状
    expected_y_dim = 1 if y.ndim == 1 else y.shape[1]
    if y_seq_np.size > 0: # 只有在非空时才检查
         if y.ndim == 1 and y_seq_np.ndim != 1:
              logger.error(f"y_seq 形状错误 (y为1D)! 得到 {y_seq_np.shape}, 预期 (N,)")
         elif y.ndim == 2 and (y_seq_np.ndim != 2 or y_seq_np.shape[1] != expected_y_dim):
              logger.error(f"y_seq 形状错误 (y为2D)! 得到 {y_seq_np.shape}, 预期 (N, {expected_y_dim})")

    logger.info(f"创建的序列: X_seq 形状 {X_seq_np.shape}, y_seq 形状 {y_seq_np.shape}")
    return X_seq_np, y_seq_np