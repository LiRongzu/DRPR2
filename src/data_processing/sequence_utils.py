import logging
import numpy as np
from typing import Tuple

def create_sequences(X: np.ndarray, y: np.ndarray, sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
    """创建LSTM训练/预测序列。"""
    logger = logging.getLogger(__name__)
    X_seq, y_seq = [], []
    if len(X) <= sequence_length:
         logger.error(f"数据长度 ({len(X)}) 对于序列长度 ({sequence_length}) 不足。")
         return np.array([]), np.array([]) # 返回空数组

    for i in range(len(X) - sequence_length):
        X_seq.append(X[i:i + sequence_length])
        y_seq.append(y[i + sequence_length]) # 预测序列后的步骤

    logger.info(f"创建的序列: X_seq 形状 {np.array(X_seq).shape}, y_seq 形状 {np.array(y_seq).shape}")
    return np.array(X_seq), np.array(y_seq)