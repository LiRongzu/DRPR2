import os
import logging
from typing import Optional
from omegaconf import DictConfig

def setup_logger(
    cfg: DictConfig,
    name: Optional[str] = None,
    log_dir: Optional[str] = None
) -> logging.Logger:
    """
    设置日志记录器，支持同时输出到控制台和文件
    
    参数:
        cfg: Hydra配置对象
        name: 日志记录器名称，默认为root logger
        log_dir: 日志文件保存目录，默认为当前目录
        
    返回:
        配置好的logger实例
    """
    logger = logging.getLogger(name)
    
    # 如果logger已经有处理器，说明已经配置过，直接返回
    if logger.handlers:
        return logger
        
    # 设置日志级别
    log_level = getattr(logging, cfg.training.logging.level.upper())
    logger.setLevel(log_level)
    
    # 创建格式化器
    formatter = logging.Formatter(
        cfg.training.logging.format,
        datefmt=cfg.training.logging.date_format
    )
    
    # 添加控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 如果配置要求，添加文件处理器
    if cfg.training.logging.to_file:
        if log_dir is None:
            log_dir = os.getcwd()
        
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, cfg.training.logging.filename)
        
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        logger.info(f"日志将同时保存到文件: {log_path}")
    
    return logger