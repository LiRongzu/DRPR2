# src/reconstruction/reconstruct_manifold.py

import os
import sys
import numpy as np
import logging
import time
import hydra
from omegaconf import DictConfig, OmegaConf
import joblib
from typing import Optional, Any

# --- 项目设置 ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils.hydra_config import DrprConfig
from src.utils.model_utils import load_model

logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    使用预测的低维特征向量和加载的流形模型进行重建。
    """
    start_run_time = time.time()
    config = DrprConfig.from_hydra_config(cfg)

    logger.info("开始流形重建脚本...")
    logger.info(f"配置信息：\n{OmegaConf.to_yaml(cfg)}")

    # --- 参数设置 ---
    # 从配置中读取路径
    manifold_model_path = config.reconstruction.manifold_model_path
    predicted_dv_path = config.reconstruction.predicted_dv_path
    if not manifold_model_path or not predicted_dv_path:
        logger.error("未在配置中指定流形模型或预测的特征向量路径")
        return
        
    # 重建结果保存到 Hydra 运行目录下的相对路径
    recon_save_dir = "reconstructions/manifold"
    os.makedirs(recon_save_dir, exist_ok=True)
    
    target_field = config.reconstruction.target_field

    # --- 1. 加载流形模型 ---
    logger.info(f"从以下路径加载流形模型: {manifold_model_path}")
    model = load_model(manifold_model_path)
    if model is None:
        logger.error("加载流形模型失败。退出。")
        return

    # --- 2. 加载预测的特征向量 ---
    logger.info(f"从以下路径加载预测的特征向量: {predicted_dv_path}")
    try:
        predicted_features = np.load(predicted_dv_path)
        logger.info(f"加载的特征向量形状: {predicted_features.shape}")
    except Exception as e:
        logger.error(f"加载预测的特征向量失败: {e}")
        return

    # --- 3. 执行重建 ---
    logger.info("开始重建...")
    reconstruction_start_time = time.time()
    try:
        # 检查是否需要重塑输入
        if hasattr(model, 'inverse_transform'):
            # 标准的 sklearn 风格接口
            reconstructed = model.inverse_transform(predicted_features)
        elif hasattr(model, 'decode'):
            # 自动编码器风格接口
            reconstructed = model.decode(predicted_features)
        else:
            logger.error("流形模型缺少必要的重建方法。")
            return
            
        reconstruction_time = time.time() - reconstruction_start_time
        logger.info(f"重建完成，用时: {reconstruction_time:.2f}秒")
        logger.info(f"重建结果形状: {reconstructed.shape}")
        
    except Exception as e:
        logger.error(f"重建过程失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return

    # --- 4. 保存重建结果 ---
    logger.info("保存重建结果...")
    recon_filename = f"reconstructed_{target_field}_from_manifold.npy"
    recon_save_path = os.path.join(recon_save_dir, recon_filename)
    try:
        np.save(recon_save_path, reconstructed)
        logger.info(f"重建结果已保存到: {recon_save_path}")
    except Exception as e:
        logger.error(f"保存重建结果失败: {e}")
        return

    total_run_time = time.time() - start_run_time
    logger.info(f"流形重建完成，总用时：{total_run_time:.2f}秒")

if __name__ == "__main__":
    main()