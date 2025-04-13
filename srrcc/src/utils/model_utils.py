# src/utils/model_utils.py

import os
import torch
import joblib
import logging
from omegaconf import DictConfig
from typing import Optional, Any, Union

logger = logging.getLogger(__name__)

def save_model(
    model: Any,
    save_path: str,
    save_format: str = "auto"
) -> bool:
    """
    保存模型到指定路径
    
    参数:
        model: 要保存的模型
        save_path: 保存路径
        save_format: 保存格式，可选值: "auto", "torch", "joblib"
        
    返回:
        bool: 保存是否成功
    """
    try:
        # 创建保存目录
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # 确定保存格式
        if save_format == "auto":
            if isinstance(model, torch.nn.Module):
                save_format = "torch"
            else:
                save_format = "joblib"
                
        # 根据格式保存
        if save_format == "torch":
            torch.save(model.state_dict(), save_path)
        else:
            joblib.dump(model, save_path)
            
        logger.info(f"模型已保存到: {save_path}")
        return True
        
    except Exception as e:
        logger.error(f"保存模型失败: {e}")
        return False

def load_model(
    load_path: str,
    device: Optional[Union[str, torch.device]] = None,
    load_format: str = "auto"
) -> Optional[Any]:
    """
    从指定路径加载模型
    
    参数:
        load_path: 加载路径
        device: PyTorch设备(仅用于torch模型)
        load_format: 加载格式，可选值: "auto", "torch", "joblib"
        
    返回:
        加载的模型，如果失败则返回None
    """
    try:
        if not os.path.exists(load_path):
            logger.error(f"模型文件不存在: {load_path}")
            return None
            
        # 确定加载格式
        if load_format == "auto":
            if load_path.endswith('.pt') or load_path.endswith('.pth'):
                load_format = "torch"
            else:
                load_format = "joblib"
                
        # 根据格式加载
        if load_format == "torch":
            model = torch.load(load_path, map_location=device)
        else:
            model = joblib.load(load_path)
            
        logger.info(f"模型已加载: {load_path}")
        return model
        
    except Exception as e:
        logger.error(f"加载模型失败: {e}")
        return None

def get_device_from_config(cfg: DictConfig) -> str:
    """从配置中获取设备设置"""
    if cfg.model.device.use_gpu and torch.cuda.is_available():
        return "cuda"
    return "cpu"