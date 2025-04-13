"""
评估指标模块

该模块提供了一系列用于评估模型性能的指标计算函数
包括均方误差(MSE)、平均绝对误差(MAE)、相对误差等
"""

import numpy as np
import logging
from typing import Dict, Union, List, Tuple
from sklearn.metrics import mean_squared_error, mean_absolute_error

def calculate_mse(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray = None) -> float:
    """
    计算均方误差 (MSE)
    
    参数:
        y_true: 真实值数组
        y_pred: 预测值数组
        mask: 掩码数组，用于排除特定位置的值
        
    返回:
        float: 均方误差值
    """
    try:
        if mask is not None:
            y_true = y_true[mask]
            y_pred = y_pred[mask]
        return mean_squared_error(y_true, y_pred)
    except Exception as e:
        logging.error(f"计算MSE时发生错误: {str(e)}")
        return float('nan')

def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray = None) -> float:
    """
    计算平均绝对误差 (MAE)
    
    参数:
        y_true: 真实值数组
        y_pred: 预测值数组
        mask: 掩码数组，用于排除特定位置的值
        
    返回:
        float: 平均绝对误差值
    """
    try:
        if mask is not None:
            y_true = y_true[mask]
            y_pred = y_pred[mask]
        return mean_absolute_error(y_true, y_pred)
    except Exception as e:
        logging.error(f"计算MAE时发生错误: {str(e)}")
        return float('nan')

def calculate_relative_error(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray = None) -> float:
    """
    计算相对误差
    
    参数:
        y_true: 真实值数组
        y_pred: 预测值数组
        mask: 掩码数组，用于排除特定位置的值
        
    返回:
        float: 相对误差值
    """
    try:
        if mask is not None:
            y_true = y_true[mask]
            y_pred = y_pred[mask]
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    except Exception as e:
        logging.error(f"计算相对误差时发生错误: {str(e)}")
        return float('nan')

def evaluate_prediction(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray = None) -> Dict[str, float]:
    """
    综合评估预测结果
    
    参数:
        y_true: 真实值数组
        y_pred: 预测值数组
        mask: 掩码数组，用于排除特定位置的值
        
    返回:
        Dict[str, float]: 包含多个评估指标的字典
    """
    try:
        metrics = {
            'mse': calculate_mse(y_true, y_pred, mask),
            'mae': calculate_mae(y_true, y_pred, mask),
            'relative_error': calculate_relative_error(y_true, y_pred, mask)
        }
        logging.info("预测评估完成:")
        for metric_name, value in metrics.items():
            logging.info(f"  {metric_name}: {value:.6f}")
        return metrics
    except Exception as e:
        logging.error(f"评估预测结果时发生错误: {str(e)}")
        return {}