# src/evaluation/visualization.py

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors # 需要导入
import seaborn as sns
from typing import Optional, Union, Tuple, Dict, List
import logging
import hydra
from omegaconf import DictConfig, OmegaConf
from datetime import datetime
import pandas as pd
from scipy import stats
from scipy.stats import pearsonr
# --- 项目设置 ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils.hydra_config import DrprConfig
# 导入集中化的数据加载函数
from src.utils.data_loader import load_raw_data, load_mask

# 导入新的 Cartopy 工具函数 和 保留的 add_colorbar
from src.utils.visualization import (
    get_cartopy_projection,
    get_data_crs,
    add_cartopy_features,
    add_gridlines,
    add_colorbar
)

logger = logging.getLogger(__name__)

def plot_spatial_comparison_at_timestep(
    orig_data_t: np.ndarray, # 原始场 (H, W)
    recon_data_t: np.ndarray, # 重建场 (H, W)
    diff_data_t: np.ndarray, # 差异场 (H, W)
    cfg: DictConfig,
    mask: np.ndarray,
    time_index: int,
    save_path: str,
    title_prefix: str = "Spatial Comparison"
):
    """使用 Cartopy 绘制特定时间点的原始、重建和差异场对比图。"""
    map_cfg = cfg.visualization.map
    vis_cfg = cfg.visualization
    eval_cfg = cfg.evaluation.spatial # 获取评估配置

    # 获取地图投影和数据 CRS
    map_proj = get_cartopy_projection(cfg)
    data_crs = get_data_crs(cfg)

    # 创建 Figure 和 三个 GeoAxes
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), # 调整尺寸适应三个图
                             subplot_kw={'projection': map_proj}, # 直接创建 GeoAxes
                             dpi=vis_cfg.get("dpi", 150))
    fig.suptitle(f"{title_prefix} at Time Index {time_index}", fontsize=16, y=0.98) # 调整 y 避免重叠

    # 加载经纬度数据
    try:
        lonlat_path = os.path.join(cfg.paths.raw_data_dir, cfg.data.coords_filename)
        lonlat_data = np.load(lonlat_path)
        lon, lat = lonlat_data[0], lonlat_data[1]
    except Exception as e: logger.error(f"无法加载绘图所需的经纬度数据: {e}"); plt.close(fig); return


    plot_titles = ["Original", "Reconstructed", "Difference (Recon - Orig)"]
    plot_data = [orig_data_t, recon_data_t, diff_data_t]
    # 色谱: 前两个用配置的，最后一个用发散色谱
    cmaps = [eval_cfg.get("colormap", "viridis"),
             eval_cfg.get("colormap", "viridis"),
             'coolwarm']

    # 确定 Original 和 Reconstructed 的统一颜色范围
    valid_orig = orig_data_t[~mask]
    valid_recon = recon_data_t[~mask]
    if valid_orig.size > 0 or valid_recon.size > 0:
        vmin_orig_recon = np.nanmin([np.nanmin(valid_orig) if valid_orig.size else np.inf,
                                     np.nanmin(valid_recon) if valid_recon.size else np.inf])
        vmax_orig_recon = np.nanmax([np.nanmax(valid_orig) if valid_orig.size else -np.inf,
                                     np.nanmax(valid_recon) if valid_recon.size else -np.inf])
    else:
        vmin_orig_recon, vmax_orig_recon = 0, 1 # Fallback
        logger.warning("无法计算 Orig/Recon 颜色范围，无有效数据。")


    # 确定 Difference 的颜色范围 (中心为 0)
    valid_diff = diff_data_t[~mask]
    if valid_diff.size > 0:
        max_abs_diff = np.nanmax(np.abs(valid_diff))
        vmin_diff, vmax_diff = -max_abs_diff, max_abs_diff
    else:
        vmin_diff, vmax_diff = -1, 1 # Fallback
        logger.warning("无法计算 Difference 颜色范围，无有效数据。")

    vmins = [vmin_orig_recon, vmin_orig_recon, vmin_diff]
    vmaxs = [vmax_orig_recon, vmax_orig_recon, vmax_diff]

    # 循环绘制每个子图
    for i, ax in enumerate(axes.flat):
        # 设置地图范围 (必须在绘制前)
        try:
             ax.set_extent(map_cfg.extent, crs=data_crs)
        except Exception as e: logger.error(f"子图 {i} 设置范围失败: {e}"); continue # 跳过这个子图

        # 添加地图特征
        add_cartopy_features(ax, cfg)

        # 准备数据和颜色映射
        data_to_plot = np.ma.masked_where(mask, plot_data[i])
        cmap = plt.get_cmap(cmaps[i])
        cmap.set_bad(color=map_cfg.get("mask_color", 'darkgray'))

        # 绘制数据
        try:
            pcm = ax.pcolormesh(lon, lat, data_to_plot,
                                transform=data_crs, # 关键
                                cmap=cmap,
                                vmin=vmins[i],
                                vmax=vmaxs[i],
                                shading=vis_cfg.get("shading", 'auto'))
        except Exception as e: logger.error(f"子图 {i} 绘制 pcolormesh 失败: {e}"); continue

        # 添加颜色条 (为每个子图添加)
        add_colorbar(fig, pcm, ax, label=plot_titles[i], shrink=0.6) # 缩小颜色条以适应布局

        # 添加网格线
        add_gridlines(ax, cfg)

        # 设置子图标题
        ax.set_title(plot_titles[i])

    # 调整整体布局
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # 留出顶部空间给 suptitle

    # 保存和显示
    if save_path:
        try:
            plt.savefig(save_path, dpi=vis_cfg.get("dpi", 300), bbox_inches='tight')
            logger.info(f"空间对比图 (t={time_index}) 已保存到: {save_path}")
        except Exception as e: logger.error(f"保存图像失败 {save_path}: {e}")
    if vis_cfg.get("show_figures", False):
        plt.show()
    plt.close(fig)

def set_plotting_style(cfg: DictConfig) -> None:
    """设置matplotlib的绘图样式"""
    plt.style.use(cfg.evaluation.visualization.style)
    plt.rcParams['font.family'] = cfg.evaluation.visualization.font.family
    plt.rcParams['font.size'] = cfg.evaluation.visualization.font.size

def calculate_temporal_mean(data: np.ndarray, window: int = 24) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算时间平均场和置信区间
    
    参数:
        data: 形状为(time_steps, ...) 的数据
        window: 移动平均窗口大小
        
    返回:
        移动平均和置信区间
    """
    # 计算移动平均
    rolling_mean = pd.Series(data).rolling(window=window, center=True).mean().to_numpy()
    
    # 计算置信区间
    rolling_std = pd.Series(data).rolling(window=window, center=True).std().to_numpy()
    confidence_interval = stats.norm.interval(0.95, loc=rolling_mean, scale=rolling_std/np.sqrt(window))
    
    return rolling_mean, confidence_interval

def plot_time_series_comparison(
    reconstructed: np.ndarray,
    original: np.ndarray,
    cfg: DictConfig,
    times: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
    title: str = "Time Series Comparison"
) -> None:
    """绘制时序对比图，包括移动平均和置信区间"""
    plt.figure(figsize=tuple(cfg.evaluation.visualization.figsize.time_series))
    
    if times is None:
        times = np.arange(len(original))
    
    # 绘制原始数据
    plt.plot(times, original, 
             color=cfg.evaluation.visualization.colors.original,
             alpha=0.3, label='Original')
             
    # 绘制重建数据
    plt.plot(times, reconstructed,
             color=cfg.evaluation.visualization.colors.reconstructed,
             alpha=0.3, label='Reconstructed')
             
    # 如果配置要求，计算和绘制移动平均
    if cfg.evaluation.temporal.plot_moving_average:
        window = cfg.evaluation.temporal.window_size
        
        # 原始数据的移动平均和置信区间
        orig_ma, orig_ci = calculate_temporal_mean(original, window)
        plt.plot(times, orig_ma, 
                color=cfg.evaluation.visualization.colors.original,
                linewidth=2, label='Original (MA)')
                
        if cfg.evaluation.temporal.include_confidence_interval:
            plt.fill_between(times, orig_ci[0], orig_ci[1],
                           color=cfg.evaluation.visualization.colors.original,
                           alpha=cfg.evaluation.temporal.error_band_alpha)
        
        # 重建数据的移动平均和置信区间
        recon_ma, recon_ci = calculate_temporal_mean(reconstructed, window)
        plt.plot(times, recon_ma,
                color=cfg.evaluation.visualization.colors.reconstructed,
                linewidth=2, label='Reconstructed (MA)')
                
        if cfg.evaluation.temporal.include_confidence_interval:
            plt.fill_between(times, recon_ci[0], recon_ci[1],
                           color=cfg.evaluation.visualization.colors.reconstructed,
                           alpha=cfg.evaluation.temporal.error_band_alpha)
    
    plt.title(title)
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, dpi=cfg.evaluation.visualization.dpi,
                   bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_spatial_distribution(
    field: np.ndarray,
    cfg: DictConfig,
    mask: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
    title: str = "Spatial Distribution",
    colorbar_label: str = "Value"
) -> None:
    """绘制空间分布图"""
    plt.figure(figsize=tuple(cfg.evaluation.visualization.figsize.spatial))
    
    if mask is not None and cfg.evaluation.spatial.mask_invalid:
        field = np.ma.masked_array(field, mask=mask)
    
    sns.heatmap(field,
                cmap=cfg.evaluation.spatial.colormap,
                mask=mask if cfg.evaluation.spatial.mask_invalid else None,
                cbar_kws={'label': colorbar_label})
    
    plt.title(title)
    if cfg.evaluation.spatial.plot_grid:
        plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, dpi=cfg.evaluation.visualization.dpi,
                   bbox_inches='tight')
        plt.close()
    else:
        plt.show()

# --- 修改 plot_spatial_rmse ---
def plot_spatial_rmse(
    spatial_rmse: np.ndarray,
    cfg: DictConfig,
    mask: np.ndarray,
    save_path: str,
    title: str = "Spatial RMSE"
):
    """使用 Cartopy 绘制空间 RMSE 分布图。"""
    map_cfg = cfg.visualization.map
    eval_cfg = cfg.evaluation.spatial

    # 1. 获取地图投影和数据 CRS
    map_proj = get_cartopy_projection(cfg)
    data_crs = get_data_crs(cfg)

    # 2. 创建 Figure 和 GeoAxes
    fig = plt.figure(figsize=tuple(map_cfg.figsize))
    ax = fig.add_subplot(1, 1, 1, projection=map_proj)

    # 3. 设置地图范围
    try:
        ax.set_extent(map_cfg.extent, crs=data_crs)
    except Exception as e:
        logger.error(f"设置地图范围失败: {e}. extent={map_cfg.extent}, data_crs={data_crs}")
        plt.close(fig)
        return

    # 4. 加载经纬度数据 (假设你的项目中有路径)
    try:
        lonlat_path = os.path.join(cfg.paths.raw_data_dir, cfg.data.coords_filename) # 从配置读取文件名
        lonlat_data = np.load(lonlat_path)
        lon, lat = lonlat_data[0], lonlat_data[1]
        logger.debug(f"Loaded coordinates from {lonlat_path}")
    except Exception as e:
        logger.error(f"无法加载绘图所需的经纬度数据: {e}")
        plt.close(fig)
        return

    # 5. 添加地图特征
    add_cartopy_features(ax, cfg)

    # 6. 准备要绘制的数据和颜色映射
    data_to_plot = np.ma.masked_where(~mask, spatial_rmse)
    cmap_name = eval_cfg.get("colormap", "viridis")
    cmap = plt.get_cmap(cmap_name)
    # 设置无效区域颜色
    cmap.set_bad(color=map_cfg.get("mask_color", 'darkgray'))

    vmin = eval_cfg.get("vmin")
    vmax = eval_cfg.get("vmax")
    if vmin is None or vmax is None:
         valid_data = data_to_plot[~data_to_plot.mask]
         if valid_data.size > 0:
             if vmin is None: vmin = np.min(valid_data)
             if vmax is None: vmax = np.max(valid_data)
         else:
             vmin, vmax = 0, 1 # Fallback if no valid data
             logger.warning("无法自动计算 vmin/vmax，因为没有有效的 RMSE 数据。")

    # 7. 绘制数据 (核心变化: 使用 ax.pcolormesh 并指定 transform)
    try:
        pcm = ax.pcolormesh(lon, lat, data_to_plot,
                            transform=data_crs, # 关键：告知 Cartopy 数据的坐标系
                            cmap=cmap,
                            vmin=vmin,
                            vmax=vmax,
                            shading=eval_cfg.get("shading", 'auto')) # Gouraud or flat shading
        logger.debug(f"绘制 pcolormesh: vmin={vmin}, vmax={vmax}, cmap={cmap_name}")
    except Exception as e:
        logger.error(f"绘制 pcolormesh 失败: {e}")
        plt.close(fig)
        return

    # 8. 添加颜色条 (使用通用函数)
    add_colorbar(fig, pcm, ax, label="RMSE")

    # 9. 添加网格线
    add_gridlines(ax, cfg)

    # 10. 设置标题和保存
    ax.set_title(title) # Cartopy 推荐用 ax.set_title 而不是 fig.suptitle 如果只有一个子图
    # plt.suptitle(title) # 或者用这个，如果喜欢总标题

    plt.tight_layout()
    if save_path:
        try:
            plt.savefig(save_path, dpi=cfg.visualization.get("dpi", 300), bbox_inches='tight')
            logger.info(f"空间 RMSE 图已保存到: {save_path}")
        except Exception as e:
            logger.error(f"保存图像失败 {save_path}: {e}")
    if cfg.visualization.get("show_figures", False):
        plt.show()
    plt.close(fig)

def generate_evaluation_report(
    metrics: Dict[str, float],
    figure_paths: List[str],
    cfg: DictConfig,
    save_dir: str
) -> None:
    """生成评估报告"""
    if not cfg.evaluation.report.generate:
        return
        
    report_content = []
    report_content.append("# 评估报告\n")
    report_content.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # 添加指标
    report_content.append("## 评估指标\n")
    for metric_name, value in metrics.items():
        report_content.append(f"- {metric_name}: {value:.4f}\n")
    
    # 添加图形
    if cfg.evaluation.report.include_figures:
        report_content.append("\n## 可视化结果\n")
        for fig_path in figure_paths:
            fig_name = os.path.basename(fig_path)
            report_content.append(f"### {fig_name}\n")
            report_content.append(f"![{fig_name}]({fig_path})\n")
    
    # 保存报告
    report_path = os.path.join(save_dir, cfg.evaluation.report.filename)
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_content))
    
    logger.info(f"评估报告已保存到: {report_path}")

def plot_spatial_difference(
    diff_data: np.ndarray,  # 差异场 (H, W)
    cfg: DictConfig,
    mask: np.ndarray,
    time_index: int,        # 用于标题
    save_path: str,
    title: str = "Spatial Difference (Recon - Orig)"
):
    """使用 Cartopy 绘制特定时间点的空间差异场。"""
    map_cfg = cfg.visualization.map
    vis_cfg = cfg.visualization

    # 获取地图投影和数据 CRS
    map_proj = get_cartopy_projection(cfg)
    data_crs = get_data_crs(cfg)

    # 创建 Figure 和 GeoAxes
    fig = plt.figure(figsize=tuple(map_cfg.figsize))
    ax = fig.add_subplot(1, 1, 1, projection=map_proj)

    # 设置地图范围
    try:
        ax.set_extent(map_cfg.extent, crs=data_crs)
    except Exception as e:
        logger.error(f"设置地图范围失败: {e}. extent={map_cfg.extent}, data_crs={data_crs}")
        plt.close(fig)
        return

    # 加载经纬度数据
    try:
        lonlat_path = os.path.join(cfg.paths.raw_data_dir, cfg.data.coords_filename)
        lonlat_data = np.load(lonlat_path)
        lon, lat = lonlat_data[0], lonlat_data[1]
    except Exception as e:
        logger.error(f"无法加载绘图所需的经纬度数据: {e}")
        plt.close(fig)
        return

    # 添加地图特征
    add_cartopy_features(ax, cfg)

    # 准备要绘制的数据和颜色映射
    data_to_plot = np.ma.masked_where(mask, diff_data)
    cmap_name = 'coolwarm' # 差异场通常用发散色谱
    cmap = plt.get_cmap(cmap_name)
    cmap.set_bad(color=map_cfg.get("mask_color", 'darkgray'))

    # 确定颜色范围 (中心为 0)
    valid_data = data_to_plot[~data_to_plot.mask]
    if valid_data.size > 0:
        max_abs_diff = np.max(np.abs(valid_data))
        vmin, vmax = -max_abs_diff, max_abs_diff
    else:
        vmin, vmax = -1, 1 # Fallback
        logger.warning("无法计算差异场颜色范围，无有效数据。")


    # 绘制数据 (使用 Cartopy 的 pcolormesh 并指定 transform)
    try:
        pcm = ax.pcolormesh(lon, lat, data_to_plot,
                            transform=data_crs, # 关键
                            cmap=cmap,
                            vmin=vmin,
                            vmax=vmax,
                            shading=vis_cfg.get("shading", 'auto'))
    except Exception as e:
        logger.error(f"绘制 pcolormesh 失败: {e}")
        plt.close(fig)
        return

    # 添加颜色条
    add_colorbar(fig, pcm, ax, label="Difference (Recon - Orig)")

    # 添加网格线
    add_gridlines(ax, cfg)

    # 设置标题和保存
    full_title = f"{title} at Time Index {time_index}"
    ax.set_title(full_title)

    plt.tight_layout()
    if save_path:
        try:
            plt.savefig(save_path, dpi=vis_cfg.get("dpi", 300), bbox_inches='tight')
            logger.info(f"空间差异图已保存到: {save_path}")
        except Exception as e:
            logger.error(f"保存图像失败 {save_path}: {e}")
    if vis_cfg.get("show_figures", False):
        plt.show()
    plt.close(fig)

def plot_spatial_statistic(
    stat_map: np.ndarray, # 计算好的统计量图 (H, W)
    cfg: DictConfig,
    mask: np.ndarray,
    save_path: str,
    title: str,
    cmap: str = 'viridis', # 默认色谱
    vmin: float = None,    # 允许外部传入 vmin/vmax
    vmax: float = None,
    cbar_label: str = "Value"
):
    """使用 Cartopy 绘制空间统计量分布图 (通用)。"""
    map_cfg = cfg.visualization.map
    vis_cfg = cfg.visualization

    # 获取地图投影和数据 CRS
    map_proj = get_cartopy_projection(cfg)
    data_crs = get_data_crs(cfg)

    # 创建 Figure 和 GeoAxes
    fig = plt.figure(figsize=tuple(map_cfg.figsize))
    ax = fig.add_subplot(1, 1, 1, projection=map_proj)

    # 设置地图范围
    try:
        ax.set_extent(map_cfg.extent, crs=data_crs)
    except Exception as e: logger.error(f"设置地图范围失败: {e}"); plt.close(fig); return

    # 加载经纬度数据
    try:
        lonlat_path = os.path.join(cfg.paths.raw_data_dir, cfg.data.coords_filename)
        lonlat_data = np.load(lonlat_path)
        lon, lat = lonlat_data[0], lonlat_data[1]
    except Exception as e: logger.error(f"无法加载绘图所需的经纬度数据: {e}"); plt.close(fig); return

    # 添加地图特征
    add_cartopy_features(ax, cfg)

    # 准备数据和颜色映射
    data_to_plot = np.ma.masked_where(mask, stat_map)
    cmap_obj = plt.get_cmap(cmap)
    cmap_obj.set_bad(color=map_cfg.get("mask_color", 'darkgray'))

    # 确定 vmin/vmax (如果未提供)
    if vmin is None or vmax is None:
         valid_data = data_to_plot[~data_to_plot.mask]
         if valid_data.size > 0:
             if vmin is None: vmin = np.min(valid_data)
             if vmax is None: vmax = np.max(valid_data)
         else:
             vmin, vmax = 0, 1 # Fallback
             logger.warning(f"无法自动计算 '{title}' 的 vmin/vmax，无有效数据。")

    # 绘制数据
    try:
        pcm = ax.pcolormesh(lon, lat, data_to_plot,
                            transform=data_crs, # 关键
                            cmap=cmap_obj,
                            vmin=vmin,
                            vmax=vmax,
                            shading=vis_cfg.get("shading", 'auto'))
    except Exception as e: logger.error(f"绘制 pcolormesh 失败: {e}"); plt.close(fig); return

    # 添加颜色条
    add_colorbar(fig, pcm, ax, label=cbar_label)

    # 添加网格线
    add_gridlines(ax, cfg)

    # 设置标题和保存
    ax.set_title(title)
    plt.tight_layout()
    if save_path:
        try:
            plt.savefig(save_path, dpi=vis_cfg.get("dpi", 300), bbox_inches='tight')
            logger.info(f"空间统计图 '{title}' 已保存到: {save_path}")
        except Exception as e: logger.error(f"保存图像失败 {save_path}: {e}")
    if vis_cfg.get("show_figures", False): plt.show()
    plt.close(fig)

def calculate_correlation_map(data1, data2, mask):
     """计算两个3D场的时间相关性图"""
     T, H, W = data1.shape
     corr_map = np.full((H, W), np.nan)
     valid_points = ~mask
     for r, c in np.argwhere(valid_points):
          ts1 = data1[:, r, c]
          ts2 = data2[:, r, c]
          valid_mask_ts = ~np.isnan(ts1) & ~np.isnan(ts2)
          if np.sum(valid_mask_ts) > 2 and np.nanvar(ts1[valid_mask_ts]) > 1e-9 and np.nanvar(ts2[valid_mask_ts]) > 1e-9:
               # 使用 try-except 捕获 pearsonr 可能的错误
               try:
                   corr, _ = pearsonr(ts1[valid_mask_ts], ts2[valid_mask_ts])
                   corr_map[r, c] = corr
               except ValueError as e:
                   logger.warning(f"无法计算点 ({r},{c}) 的相关性: {e}") # 例如输入包含 NaN/inf
          # else: # 可选: 记录无法计算相关性的原因
          #     logger.debug(f"Skipping correlation at ({r},{c}): insufficient data or zero variance.")
     return corr_map



@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """主函数：评估重建结果并测试 Cartopy 绘图"""
    # 包装配置 (可选, 但使用了 DrprConfig)
    try:
        config = DrprConfig.from_hydra_config(cfg)
        logger.info("Configuration loaded successfully via DrprConfig.")
    except Exception as e:
        logger.error(f"Failed to load config via DrprConfig: {e}. Using raw DictConfig.")
        config = cfg # Fallback to raw DictConfig if needed

    # 设置绘图样式
    set_plotting_style(cfg)

    # --- 1. 加载数据 ---
    logger.info("Loading data...")
    try:
        # 使用集中化数据加载函数
        original_data = load_raw_data(cfg, 'salinity') # 假设评估目标是 salinity
        mask = load_mask(cfg)

        # 重建数据路径 (从配置读取或保持现状)
        # 注意: 这里的路径可能需要根据你的实际输出调整，或者从 cfg 中读取
        recons_dir = config.paths.reconstructions_dir # 使用包装后的 config
        weight_recon_path = os.path.join(recons_dir, "reconstructed_salinity_from_bmu_weights.npy")
        proto_recon_path = os.path.join(recons_dir, "reconstructed_salinity_from_bmu_prototypes.npy") # 注意文件名是否准确

        # 加载重建数据
        weight_recon = np.load(weight_recon_path)
        # 检查 proto 文件是否存在，如果不存在则跳过评估
        if os.path.exists(proto_recon_path):
             proto_recon = np.load(proto_recon_path)
             has_proto_recon = True
        else:
             proto_recon = None # 或者设为 None
             has_proto_recon = False
             logger.warning(f"Prototype reconstruction file not found: {proto_recon_path}")


        # 确保数据形状匹配 (非常重要)
        min_len = min(len(original_data), len(weight_recon))
        if has_proto_recon:
            min_len = min(min_len, len(proto_recon))

        original_data = original_data[:min_len]
        weight_recon = weight_recon[:min_len]
        if has_proto_recon:
            proto_recon = proto_recon[:min_len]

        logger.info("Data loaded and aligned:")
        logger.info(f"  Original data shape: {original_data.shape}")
        logger.info(f"  Weight recon shape: {weight_recon.shape}")
        if has_proto_recon:
             logger.info(f"  Prototype recon shape: {proto_recon.shape}")
        else:
             logger.info("  Prototype recon data not available.")

    except FileNotFoundError as e:
         logger.error(f"Required data file not found: {e}")
         return
    except Exception as e:
        logger.error(f"Data loading failed: {e}")
        return

    # --- 2. 创建评估目录结构 ---
    eval_dir = os.path.join(os.getcwd(), cfg.evaluation.save.root_dir)
    figures_dir = os.path.join(eval_dir, cfg.evaluation.save.subfolders.figures)
    metrics_dir = os.path.join(eval_dir, cfg.evaluation.save.subfolders.metrics)

    for directory in [eval_dir, figures_dir, metrics_dir]:
        os.makedirs(directory, exist_ok=True)

    # --- 计算差异场 ---
    diff_weight = weight_recon - original_data
    if has_proto_recon:
        diff_proto = proto_recon - original_data

    # --- 计算指标 ---
    logger.info("Calculating metrics for weight-based reconstruction...")
    # 确保传入的 mask 是 2D
    weight_metrics = calculate_correlation_map(weight_recon, original_data, mask)
    if has_proto_recon:
        logger.info("Calculating metrics for prototype-based reconstruction...")
        proto_metrics = calculate_correlation_map(proto_recon, original_data, mask)
    else:
        proto_metrics = None

    # 用于报告的图形路径列表
    figure_paths = []
    evaluation_results = {} # 用于存储最终指标

    # --- 3. 评估权重方法 ---
    logger.info("\n--- Evaluating Weight-Based Reconstruction ---")
    evaluation_results['weight_based'] = {}

    # 3.1 时间序列图 (非 Cartopy)
    ts_weight_path = os.path.join(figures_dir, "time_series_weights.png")
    plot_time_series_comparison(
        np.nanmean(weight_recon, axis=(1,2)), # 计算空间平均值
        np.nanmean(original_data, axis=(1,2)), # 计算空间平均值
        cfg,
        save_path=ts_weight_path,
        title="Time Series Comparison (Weight-based)"
    )
    figure_paths.append(ts_weight_path)
    logger.info(f"  Time series comparison saved for weights.")

    # 3.2 空间 RMSE 图 (Cartopy)
    rmse_cartopy_weight_path = os.path.join(figures_dir, "spatial_rmse_cartopy_weights.png")
    plot_spatial_rmse(
        weight_metrics['rmse'], cfg, mask, rmse_cartopy_weight_path,
        title="Spatial RMSE (Weight-based, Cartopy)"
    )
    figure_paths.append(rmse_cartopy_weight_path)
    evaluation_results['weight_based']['rmse_map'] = weight_metrics['rmse'] # 保存指标图本身
    evaluation_results['weight_based']['mean_rmse'] = float(np.nanmean(weight_metrics['rmse'])) # 保存统计值

    # 3.3 (可选) 空间 RMSE 图 (Heatmap)
    if cfg.evaluation.get("plot_heatmap_rmse", False): # 添加配置开关
        rmse_heatmap_weight_path = os.path.join(figures_dir, "rmse_distribution_heatmap_weights.png")
        plot_spatial_distribution( # 注意函数名
            weight_metrics['rmse'], cfg, mask=mask,
            save_path=rmse_heatmap_weight_path,
            title="RMSE Distribution (Weight-based, Heatmap)",
            colorbar_label="RMSE"
        )
        figure_paths.append(rmse_heatmap_weight_path)
        logger.info(f"  RMSE Heatmap saved for weights.")


    # 3.4 空间相关性图 (Cartopy)
    corr_cartopy_weight_path = os.path.join(figures_dir, "spatial_correlation_cartopy_weights.png")
    plot_spatial_statistic(
        weight_metrics['correlation'], cfg, mask, corr_cartopy_weight_path,
        title="Spatial Correlation (Weight-based, Cartopy)",
        cmap='coolwarm', vmin=-1, vmax=1, cbar_label="Correlation Coeff."
    )
    figure_paths.append(corr_cartopy_weight_path)
    evaluation_results['weight_based']['correlation_map'] = weight_metrics['correlation']
    evaluation_results['weight_based']['mean_correlation'] = float(np.nanmean(weight_metrics['correlation']))


    # 3.5 特定时间点的对比和差异图 (Cartopy)
    time_indices_to_plot = [0, len(original_data) // 2, len(original_data) - 1] # 选几个时间点
    for t_idx in time_indices_to_plot:
        if 0 <= t_idx < len(original_data):
            logger.info(f"  Plotting weight-based spatial comparison/difference for t={t_idx}...")
            # 并排对比图
            comp_weight_path = os.path.join(figures_dir, f"spatial_comparison_weights_t{t_idx}.png")
            plot_spatial_comparison_at_timestep(
                original_data[t_idx], weight_recon[t_idx], diff_weight[t_idx],
                cfg, mask, t_idx, comp_weight_path,
                title_prefix="Spatial Comparison (Weight-based)"
            )
            figure_paths.append(comp_weight_path)
            # 差异图
            diff_weight_path = os.path.join(figures_dir, f"spatial_difference_weights_t{t_idx}.png")
            plot_spatial_difference(
                diff_weight[t_idx], cfg, mask, t_idx, diff_weight_path,
                title="Spatial Difference (Weight-based, Recon - Orig)"
            )
            figure_paths.append(diff_weight_path)

    # 添加其他指标到结果
    evaluation_results['weight_based']['mae_map'] = weight_metrics['mae']
    evaluation_results['weight_based']['mean_mae'] = float(np.nanmean(weight_metrics['mae']))


    # --- 4. 评估簇类均值方法 (如果数据存在) ---
    if has_proto_recon and proto_metrics is not None:
        logger.info("\n--- Evaluating Prototype-Based Reconstruction ---")
        evaluation_results['prototype_based'] = {}

        # 4.1 时间序列图
        ts_proto_path = os.path.join(figures_dir, "time_series_prototypes.png")
        plot_time_series_comparison(
            np.nanmean(proto_recon, axis=(1,2)),
            np.nanmean(original_data, axis=(1,2)),
            cfg,
            save_path=ts_proto_path,
            title="Time Series Comparison (Prototype-based)"
        )
        figure_paths.append(ts_proto_path)
        logger.info(f"  Time series comparison saved for prototypes.")

        # 4.2 空间 RMSE 图 (Cartopy)
        rmse_cartopy_proto_path = os.path.join(figures_dir, "spatial_rmse_cartopy_prototypes.png")
        plot_spatial_rmse(
            proto_metrics['rmse'], cfg, mask, rmse_cartopy_proto_path,
            title="Spatial RMSE (Prototype-based, Cartopy)"
        )
        figure_paths.append(rmse_cartopy_proto_path)
        evaluation_results['prototype_based']['rmse_map'] = proto_metrics['rmse']
        evaluation_results['prototype_based']['mean_rmse'] = float(np.nanmean(proto_metrics['rmse']))

        # 4.3 (可选) 空间 RMSE 图 (Heatmap)
        if cfg.evaluation.get("plot_heatmap_rmse", False):
            rmse_heatmap_proto_path = os.path.join(figures_dir, "rmse_distribution_heatmap_prototypes.png")
            plot_spatial_distribution(
                proto_metrics['rmse'], cfg, mask=mask,
                save_path=rmse_heatmap_proto_path,
                title="RMSE Distribution (Prototype-based, Heatmap)",
                colorbar_label="RMSE"
            )
            figure_paths.append(rmse_heatmap_proto_path)
            logger.info(f"  RMSE Heatmap saved for prototypes.")

        # 4.4 空间相关性图 (Cartopy)
        corr_cartopy_proto_path = os.path.join(figures_dir, "spatial_correlation_cartopy_prototypes.png")
        plot_spatial_statistic(
            proto_metrics['correlation'], cfg, mask, corr_cartopy_proto_path,
            title="Spatial Correlation (Prototype-based, Cartopy)",
            cmap='coolwarm', vmin=-1, vmax=1, cbar_label="Correlation Coeff."
        )
        figure_paths.append(corr_cartopy_proto_path)
        evaluation_results['prototype_based']['correlation_map'] = proto_metrics['correlation']
        evaluation_results['prototype_based']['mean_correlation'] = float(np.nanmean(proto_metrics['correlation']))

        # 4.5 特定时间点的对比和差异图 (Cartopy)
        for t_idx in time_indices_to_plot:
             if 0 <= t_idx < len(original_data):
                logger.info(f"  Plotting prototype-based spatial comparison/difference for t={t_idx}...")
                # 并排对比图
                comp_proto_path = os.path.join(figures_dir, f"spatial_comparison_prototypes_t{t_idx}.png")
                plot_spatial_comparison_at_timestep(
                    original_data[t_idx], proto_recon[t_idx], diff_proto[t_idx],
                    cfg, mask, t_idx, comp_proto_path,
                    title_prefix="Spatial Comparison (Prototype-based)"
                )
                figure_paths.append(comp_proto_path)
                # 差异图
                diff_proto_path = os.path.join(figures_dir, f"spatial_difference_prototypes_t{t_idx}.png")
                plot_spatial_difference(
                    diff_proto[t_idx], cfg, mask, t_idx, diff_proto_path,
                    title="Spatial Difference (Prototype-based, Recon - Orig)"
                )
                figure_paths.append(diff_proto_path)

        # 添加其他指标到结果
        evaluation_results['prototype_based']['mae_map'] = proto_metrics['mae']
        evaluation_results['prototype_based']['mean_mae'] = float(np.nanmean(proto_metrics['mae']))

    # --- 5. 保存评估结果 ---
    logger.info("\nSaving evaluation metrics...")
    # 保存指标 (包含统计值和可能的地图)
    # 注意: 保存地图可能会使 .npy 文件很大，如果只关心统计值可以只保存它们
    metrics_save_path = os.path.join(metrics_dir, "evaluation_metrics.npy")
    np.save(metrics_save_path, evaluation_results)
    logger.info(f"Evaluation metrics saved to: {metrics_save_path}")

    # --- 6. 生成报告 ---
    # 准备报告用的简化指标字典 (只包含统计值)
    report_metrics = {}
    for method, metrics_dict in evaluation_results.items():
        report_metrics[method] = {k: v for k, v in metrics_dict.items() if isinstance(v, (float, int))}

    logger.info("Generating evaluation report...")
    generate_evaluation_report(report_metrics, figure_paths, cfg, eval_dir)

    # --- 7. 打印评估结果摘要 ---
    logger.info("\n--- Evaluation Summary ---")
    for method, stats in report_metrics.items():
        logger.info(f"{method.replace('_', ' ').title()}:")
        for metric_name, value in stats.items():
            logger.info(f"  {metric_name}: {value:.4f}")
        logger.info("-" * 30)

if __name__ == "__main__":
    main()