"""
Generate BMU (Best Matching Unit) sequences

This module provides functionality to generate BMU sequences from trained SOM models.
BMU sequences represent the best matching unit positions for each input sample on the SOM grid.
"""

import os
import numpy as np
import torch
import logging
import matplotlib
matplotlib.use('Agg')  # <-- 设置后端为 Agg (必须在 import pyplot 之前)
import matplotlib.pyplot as plt

from typing import Optional, Dict, Any, List, Tuple
from omegaconf import DictConfig

def plot_bmu_distribution(frequency_map: np.ndarray, title: str = "BMU Distribution", 
                         save_path: Optional[str] = None, figsize: tuple = (10, 8),
                         cmap: str = 'viridis', dpi: int = 100) -> None:
    """
    Plot BMU frequency distribution heatmap
    
    Args:
        frequency_map: BMU frequency distribution array, shape (map_size_x, map_size_y)
        title: Plot title
        save_path: Save path, if None the plot will be displayed instead of saved
        figsize: Figure size
        cmap: Color map
        dpi: Image DPI
    """
    plt.figure(figsize=figsize)
    plt.imshow(frequency_map.T, cmap=cmap, interpolation='none')
    plt.colorbar(label='Sample Count')
    plt.title(title)
    plt.xlabel('BMU X Coordinate')
    plt.ylabel('BMU Y Coordinate')
    
    # Add text annotations for high-frequency regions
    max_freq = np.max(frequency_map)
    for i in range(frequency_map.shape[0]):
        for j in range(frequency_map.shape[1]):
            count = frequency_map[i, j]
            if count > max_freq * 0.1:  # Only show numbers for cells with >10% of max frequency
                plt.text(i, j, str(int(count)), 
                        ha='center', va='center', 
                        color='white' if count > max_freq * 0.5 else 'black',
                        fontsize=8)
    
    # Set ticks
    plt.xticks(np.arange(0, frequency_map.shape[0], 2))
    plt.yticks(np.arange(0, frequency_map.shape[1], 2))
    plt.grid(True, alpha=0.2, linestyle='--')
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def generate_BMU(
    som_model, 
    data: np.ndarray,
    bmu_positions_path: str,
    bmu_map_path: str,
    bmu_freq_path: str,
    bmu_dist_plot_path: Optional[str] = None,
    cfg: Optional[DictConfig] = None
):
    """
    生成 BMU 数据并保存到指定的唯一路径。

    Args:
        som_model: 训练好的 SOM 模型
        data: 输入数据，已标准化
        bmu_positions_path: BMU 位置序列的完整保存路径
        bmu_map_path: BMU 映射字典的完整保存路径
        bmu_freq_path: BMU 频率图数据的完整保存路径
        bmu_dist_plot_path: BMU 分布图的完整保存路径（可选）
        cfg: Hydra 配置对象（可选）
    """
    logger = logging.getLogger(__name__)
    logger.info("开始生成 BMU 数据...")

    # 确保所有路径都是绝对路径
    bmu_positions_path = os.path.abspath(bmu_positions_path)
    bmu_map_path = os.path.abspath(bmu_map_path)
    bmu_freq_path = os.path.abspath(bmu_freq_path)
    if bmu_dist_plot_path:
        bmu_dist_plot_path = os.path.abspath(bmu_dist_plot_path)

    # 确保所有输出目录都存在
    for path in [bmu_positions_path, bmu_map_path, bmu_freq_path]:
        if path:
            os.makedirs(os.path.dirname(path), exist_ok=True)

    # 转换数据到张量
    if isinstance(data, np.ndarray):
        data_tensor = torch.tensor(data, dtype=torch.float32).to(som_model.device)
    else:
        data_tensor = data.to(som_model.device)

    # 计算 BMU
    try:
        _, bmu_grid_indices = som_model(data_tensor)
        
        # 处理BMU索引
        if isinstance(bmu_grid_indices, torch.Tensor):
            bmu_positions = bmu_grid_indices.cpu().numpy()
        elif isinstance(bmu_grid_indices, list):
            bmu_positions = np.array(bmu_grid_indices)
        else:
            raise TypeError(f"意外的BMU索引类型: {type(bmu_grid_indices)}")

        # 1. 保存 BMU 位置序列
        try:
            np.save(bmu_positions_path, bmu_positions)
            logger.info(f"BMU 位置序列已保存至: {bmu_positions_path}")
        except Exception as e:
            logger.error(f"保存 BMU 位置失败: {e}", exc_info=True)
            return None

        # 2. 生成和保存 BMU 映射
        try:
            bmu_map = {}
            for i, pos in enumerate(bmu_positions):
                key_tuple = tuple(pos)
                if key_tuple not in bmu_map:
                    bmu_map[key_tuple] = []
                bmu_map[key_tuple].append(i)
            
            np.save(bmu_map_path, bmu_map, allow_pickle=True)
            logger.info(f"BMU 映射已保存至: {bmu_map_path}")
        except Exception as e:
            logger.error(f"生成或保存 BMU 映射失败: {e}", exc_info=True)

        # 3. 生成和保存 BMU 频率图
        try:
            map_size = som_model.map_size
            frequency_map = np.zeros(map_size, dtype=np.int32)
            
            for pos in bmu_positions:
                x_idx, y_idx = int(pos[0]), int(pos[1])
                if 0 <= x_idx < map_size[0] and 0 <= y_idx < map_size[1]:
                    frequency_map[x_idx, y_idx] += 1
                else:
                    logger.warning(f"BMU位置 {pos} 超出地图大小 {map_size}，跳过频率更新")
            
            np.save(bmu_freq_path, frequency_map)
            logger.info(f"BMU 频率数据已保存至: {bmu_freq_path}")

            # 4. 可选：生成和保存 BMU 分布图
            if bmu_dist_plot_path:
                try:
                    vis_cfg = cfg.get("visualization", {}) if cfg else {}
                    figure_cfg = vis_cfg.get("figure", {})
                    save_cfg = vis_cfg.get("save", {})

                    os.makedirs(os.path.dirname(bmu_dist_plot_path), exist_ok=True)
                    plot_bmu_distribution(
                        frequency_map,
                        title="BMU Distribution",
                        save_path=bmu_dist_plot_path,
                        figsize=tuple(figure_cfg.get("size", [10, 8])),
                        dpi=save_cfg.get("dpi", 100)
                    )
                    logger.info(f"BMU 分布图已保存至: {bmu_dist_plot_path}")
                except Exception as e:
                    logger.error(f"生成 BMU 分布图失败: {e}", exc_info=True)

        except Exception as e:
            logger.error(f"生成或保存 BMU 频率失败: {e}", exc_info=True)

        return bmu_positions

    except Exception as e:
        logger.error(f"生成 BMU 数据时发生错误: {e}", exc_info=True)
        return None