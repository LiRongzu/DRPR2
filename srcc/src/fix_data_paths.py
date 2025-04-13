# src/fix_data_paths.py

import os
import sys
import logging
import hydra
from omegaconf import DictConfig

# 配置基本日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# --- 项目设置 ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def create_symlinks(data_dir: str, field_name: str):
    """为给定字段创建符号链接，以支持多种文件命名约定
    
    参数:
        data_dir: 数据目录路径
        field_name: 字段名称（如 'salinity', 'wind', 'flow'）
    """
    # 确保正确处理路径
    processed_dir = os.path.join(data_dir, 'processed', 'mini')
    
    # 定义不同的命名约定
    naming_conventions = [
        (f"train_{field_name}_processed.npy", f"{field_name}_train.npy"),
        (f"val_{field_name}_processed.npy", f"{field_name}_val.npy"),
        (f"test_{field_name}_processed.npy", f"{field_name}_test.npy"),
    ]
    
    # 创建符号链接
    for source_name, target_name in naming_conventions:
        source_path = os.path.join(processed_dir, source_name)
        target_path = os.path.join(processed_dir, target_name)
        
        # 检查源文件是否存在
        if os.path.exists(source_path):
            logger.info(f"源文件存在: {source_path}")
            
            # 如果目标已存在但不是符号链接，备份它
            if os.path.exists(target_path) and not os.path.islink(target_path):
                backup_path = target_path + ".bak"
                logger.info(f"备份现有文件: {target_path} -> {backup_path}")
                os.rename(target_path, backup_path)
            
            # 如果目标是符号链接但指向错误的位置，删除它
            elif os.path.islink(target_path) and os.readlink(target_path) != source_path:
                logger.info(f"删除现有符号链接: {target_path}")
                os.remove(target_path)
            
            # 创建符号链接
            if not os.path.exists(target_path):
                logger.info(f"创建符号链接: {target_path} -> {source_path}")
                os.symlink(source_path, target_path)
            else:
                logger.info(f"目标文件已存在: {target_path}")
        else:
            logger.warning(f"源文件不存在: {source_path}")

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """创建数据文件符号链接以适应不同的命名约定"""
    logger.info("开始创建数据文件符号链接...")
    
    # 获取数据目录
    data_dir = os.path.join(cfg.project.root_dir, 'data')
    logger.info(f"数据目录: {data_dir}")
    
    # 为不同字段创建符号链接
    fields = ['salinity', 'wind', 'flow']
    for field in fields:
        logger.info(f"处理字段: {field}")
        create_symlinks(data_dir, field)
    
    logger.info("符号链接创建完成。")

if __name__ == "__main__":
    main()