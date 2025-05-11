# src/evaluation/visualization.py

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors # 需要导入
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import seaborn as sns
from typing import Optional, Union, Tuple, Dict, List
import logging
import hydra
from omegaconf import DictConfig, OmegaConf
from datetime import datetime
import pandas as pd
from scipy import stats
from scipy.stats import pearsonr
from typing import Optional, Tuple, Dict, Any
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


def plot_spatial_rmse_optimized(
    spatial_rmse: np.ndarray,
    cfg: dict, # 使用普通字典作为示例，你可以传入 DictConfig
    mask: np.ndarray,
    save_path: str,
    # --- 参数 ---
    title: str = "空间RMSE分布 (Spatial RMSE Distribution)",
    cmap_name: str = "magma", # 默认使用 'magma' (感知均匀)
    colorbar_label: str = "RMSE (PSU)", # 包含单位
    colorbar_extend: str = 'neither', # 'neither', 'min', 'max', 'both'
    colorbar_nticks: int = 5, # 颜色条刻度数量
    colorbar_shrink: float = 0.8, # <--- 新增：控制颜色条长度比例，默认80%
    gridline_fontsize: int = 10, # 网格标签字体大小
    title_fontsize: int = 14, # 标题字体大小
    shading: str = 'auto' # 'auto', 'flat', 'gouraud'
):
    """
    使用 Cartopy 绘制优化后的空间 RMSE 分布图。
    经纬度数据从 cfg 指定的文件路径加载。
    包含颜色条长度控制。

    Args:
        spatial_rmse (np.ndarray): 包含空间RMSE值的二维数组。
        cfg (dict): 配置字典，需要包含经纬度文件路径及其他绘图配置。
        mask (np.ndarray): 布尔数组，True表示有效水域，False表示陆地/无效区域。
        save_path (str): 图像保存路径。
        title (str): 图表标题 (建议简洁)。
        cmap_name (str): Matplotlib 颜色映射方案名称。
        colorbar_label (str): 颜色条标签文字。
        colorbar_extend (str): 颜色条末端箭头样式。
        colorbar_nticks (int): 颜色条期望的刻度数量。
        colorbar_shrink (float): 颜色条长度缩放因子 (小于1使其变短)。
        gridline_fontsize (int): 经纬度网格标签字体大小。
        title_fontsize (int): 图表标题字体大小。
        shading (str): pcolormesh 的 shading 参数。
    """
    map_cfg = cfg.get('visualization', {}).get('map', {})
    eval_cfg = cfg.get('evaluation', {}).get('spatial', {})
    vis_cfg = cfg.get('visualization', {})
    paths_cfg = cfg.get('paths', {}) # 获取路径配置
    data_cfg = cfg.get('data', {})   # 获取数据配置

    # --- 1. 获取地图投影和数据 CRS ---
    # [代码用中文注释] (与上一版本相同)
    proj_name = map_cfg.get('projection', 'PlateCarree')
    if proj_name == 'Mercator': map_proj = ccrs.Mercator()
    else: map_proj = ccrs.PlateCarree()
    data_crs = ccrs.PlateCarree()

    # --- 2. 创建 Figure 和 GeoAxes ---
    # [代码用中文注释] (与上一版本相同)
    fig = plt.figure(figsize=tuple(map_cfg.get('figsize', (10, 8))))
    ax = fig.add_subplot(1, 1, 1, projection=map_proj)

    # --- 3. 设置地图范围 ---
    # [代码用中文注释] (与上一版本相同)
    extent = map_cfg.get('extent')
    if extent:
        try: ax.set_extent(extent, crs=data_crs)
        except Exception as e: logger.error(f"设置地图范围失败: {e}"); plt.close(fig); return
    else: logger.warning("未在配置中找到地图范围 'extent'。")

    # --- 4. 加载经纬度数据 ---
    # [代码用中文注释] (与上一版本相同)
    try:
        raw_data_dir = paths_cfg.get('raw_data_dir', '.')
        coords_filename = data_cfg.get('coords_filename')
        if not coords_filename: raise ValueError("配置中未找到 'data.coords_filename'")
        lonlat_path = os.path.join(raw_data_dir, coords_filename)
        lonlat_data = np.load(lonlat_path)
        lon, lat = lonlat_data[0], lonlat_data[1]
        logger.debug(f"成功从 {lonlat_path} 加载经纬度数据。")
        # 可选的维度检查
        # if lon.shape != lat.shape or lon.ndim not in [1, 2] or lat.ndim not in [1, 2]:
        #      logger.warning(f"加载的经纬度数据维度可能不匹配...")
    except FileNotFoundError: logger.error(f"无法找到经纬度文件: {lonlat_path}"); plt.close(fig); return
    except Exception as e: logger.error(f"加载或处理经纬度数据时出错: {e}"); plt.close(fig); return

    # --- 5. 添加地图特征 ---
    # [代码用中文注释] (与上一版本相同)
    try:
        land_color = map_cfg.get("land_color", "lightgrey")
        ax.add_feature(cfeature.LAND, facecolor=land_color, zorder=1)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8, zorder=2)
        # 添加其他可选要素...
    except Exception as e: logger.error(f"添加地图要素时出错: {e}")

    # --- 6. 准备要绘制的数据和颜色映射 ---
    # [代码用中文注释] (与上一版本相同)
    data_to_plot = np.ma.masked_where(~mask, spatial_rmse)
    cmap_name_resolved = eval_cfg.get("colormap", cmap_name)
    try:
        cmap = plt.get_cmap(cmap_name_resolved)
        cmap.set_bad(color=map_cfg.get("mask_color", 'darkgray'))
    except ValueError:
        logger.warning(f"颜色映射方案 '{cmap_name_resolved}' 不存在，将使用 'viridis'。")
        cmap_name_resolved = 'viridis'
        cmap = plt.get_cmap(cmap_name_resolved); cmap.set_bad(color=map_cfg.get("mask_color", 'darkgray'))

    # --- 7. 确定颜色映射范围 vmin, vmax ---
    # [代码用中文注释] (与上一版本相同)
    vmin = eval_cfg.get("vmin"); vmax = eval_cfg.get("vmax")
    # 自动计算 vmin/vmax 的逻辑... (保持不变)
    if vmin is None or vmax is None:
        valid_data = data_to_plot[~data_to_plot.mask]
        if valid_data.size > 0:
            auto_vmin = np.percentile(valid_data, eval_cfg.get("vmin_percentile", 1))
            auto_vmax = np.percentile(valid_data, eval_cfg.get("vmax_percentile", 99))
            if vmin is None: vmin = auto_vmin
            if vmax is None: vmax = auto_vmax
            logger.info(f"自动计算得到 vmin={vmin:.2f}, vmax={vmax:.2f}")
        else: vmin, vmax = 0, 1; logger.warning("无有效RMSE数据，使用[0, 1]范围。")
    if vmin >= vmax: vmin, vmax = 0, 1; logger.warning(f"vmin>=vmax，使用[0, 1]范围。")


    # --- 8. 绘制数据 (pcolormesh) ---
    # [代码用中文注释] (与上一版本相同)
    try:
        shading_option = eval_cfg.get("shading", shading)
        pcm = ax.pcolormesh(lon, lat, data_to_plot, transform=data_crs, cmap=cmap,
                            vmin=vmin, vmax=vmax, shading=shading_option, zorder=3)
    except Exception as e: logger.error(f"绘制 pcolormesh 失败: {e}"); plt.close(fig); return

    # --- 9. 添加颜色条 (关键修改处) ---
    try:
        # [代码用中文注释] 优先使用函数参数 colorbar_shrink，其次查找配置项，最后使用默认值
        shrink_factor = map_cfg.get("colorbar_shrink", colorbar_shrink)

        cb = fig.colorbar(pcm, ax=ax, orientation='vertical',
                          shrink=shrink_factor, # <--- 应用缩放因子
                          # [代码用中文注释] fraction 和 pad 可能需要根据 shrink 调整以获得最佳外观
                          # fraction=0.046 * (1 / shrink_factor), # 尝试根据shrink调整? (需要实验)
                          fraction=0.04, # 或者保持默认/固定值
                          pad=0.04,
                          extend=colorbar_extend, label=colorbar_label)

        cb.ax.tick_params(labelsize=gridline_fontsize)
        cb.set_label(colorbar_label, size=gridline_fontsize + 1)
        if colorbar_nticks is not None:
            tick_locator = mticker.MaxNLocator(nbins=colorbar_nticks)
            cb.locator = tick_locator
            cb.update_ticks()
    except Exception as e:
        logger.error(f"添加颜色条失败: {e}")

    # --- 10. 添加网格线 ---
    # [代码用中文注释] (与上一版本相同)
    try:
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.5,
                          color='gray', alpha=0.5, linestyle='--')
        gl.top_labels = False; gl.right_labels = False
        gl.xformatter = mticker.FormatStrFormatter('%.2f°E')
        gl.yformatter = mticker.FormatStrFormatter('%.2f°N')
        gl.xlabel_style = {'size': gridline_fontsize, 'color': 'black'}
        gl.ylabel_style = {'size': gridline_fontsize, 'color': 'black'}
    except Exception as e: logger.error(f"添加网格线失败: {e}")

    # --- 11. 设置标题 ---
    ax.set_title(title, fontsize=title_fontsize)

    # --- 12. 调整布局和保存 ---
    # [代码用中文注释] (与上一版本相同)
    try: plt.tight_layout()
    except Exception as e: logger.warning(f"执行 tight_layout 时出错: {e}")
    if save_path:
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=vis_cfg.get("dpi", 300), bbox_inches='tight')
            logger.info(f"空间 RMSE 图已保存到: {save_path}")
        except Exception as e: logger.error(f"保存图像失败 {save_path}: {e}")
    if vis_cfg.get("show_figures", False): plt.show()
    plt.close(fig)

def plot_spatial_comparison_at_timestep(
    orig_data_t: np.ndarray, # 原始场 (H, W)
    recon_data_t: np.ndarray, # 重建场 (H, W)
    diff_data_t: np.ndarray, # 差异场 (H, W)
    cfg: DictConfig,
    mask: np.ndarray,
    time_index: int,
    save_path: str,
    title_prefix: str = "Spatial Salinity Comparison"
):

    map_cfg = cfg.visualization.map
    vis_cfg = cfg.visualization
    eval_cfg = cfg.evaluation.spatial # 获取评估配置

    # 获取地图投影和数据 CRS
    map_proj = get_cartopy_projection(cfg)
    data_crs = get_data_crs(cfg)

    # 创建 Figure 和 三个 GeoAxes
    fig, axes = plt.subplots(1, 3, figsize=(18, 4), # 调整尺寸适应三个图
                             subplot_kw={'projection': map_proj}, # 直接创建 GeoAxes
                             dpi=vis_cfg.get("dpi", 150))
    suptitle_y_pos = vis_cfg.get("suptitle_y", 0.94) # 从配置获取或设默认值
    fig.suptitle(f"{title_prefix} (t={time_index})",
                fontsize=vis_cfg.get("title_fontsize", 16),
                y=suptitle_y_pos)

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

        if i <= 1:
        # 添加颜色条 (为每个子图添加)
            add_colorbar(fig, pcm, ax, label="盐度（PSU）", shrink=0.6) # 缩小颜色条以适应布局
        else:   
            add_colorbar(fig, pcm, ax, label="盐度（PSU）", shrink=0.6) # 缩小颜色条以适应布局


        # 添加网格线
        add_gridlines(ax, cfg)

        # 设置子图标题
        ax.set_title(plot_titles[i])

    # 调整整体布局
    # plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # 留出顶部空间给 suptitle
    # [代码用中文注释] 定义布局参数，可以从配置获取或使用默认值
    default_adjust = {
        'left': 0.04,     # 图形左边缘到子图左边缘的距离 (占图形宽度比例)
        'right': 0.96,    # 图形右边缘到子图右边缘的距离 (需要足够空间给最右侧颜色条)
        'top': 0.93, # 子图顶部边缘的位置 (要低于 suptitle)
        'bottom': 0.1,    # 子图底部边缘的位置 (为 x 轴标签留空间)
        'wspace': 0.25     # 子图之间的水平间距 (占平均子图宽度比例)
        # 'hspace': 0.2    # 子图之间的垂直间距 (对于 1xN 布局不重要)
    }
    adjust_params = vis_cfg.get("subplots_adjust", default_adjust)
    try:
        plt.subplots_adjust(**adjust_params)
        logger.debug(f"应用 subplots_adjust: {adjust_params}")
    except Exception as e:
        logger.warning(f"执行 subplots_adjust 时出错: {e}")
        # 可以选择在这里添加 fallback 到 tight_layout
        # try: plt.tight_layout(rect=[0, 0.05, 1, suptitle_y_pos - 0.03])
        # except Exception as te: logger.error(f"执行 tight_layout 也失败: {te}")

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
    style_name = cfg.evaluation.visualization.style
    try:
        plt.style.use(style_name)
    except OSError:
        logger.warning(f"Matplotlib style '{style_name}' 不可用，自动切换为 'default'")
        plt.style.use('default')
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

def calculate_moving_rmse(original: np.ndarray,
                          reconstructed: np.ndarray,
                          window: int,
                          center: bool = True) -> np.ndarray:
    """
    计算原始数据和重建数据之间的移动窗口RMSE。

    Args:
        original (np.ndarray): 原始时间序列数据。
        reconstructed (np.ndarray): 重建/预测的时间序列数据。
        window (int): 移动窗口的大小。
        center (bool): 窗口是否居中。True表示居中，False表示后视窗口。

    Returns:
        np.ndarray: 移动RMSE时间序列。
    """
    # [代码用中文注释] 确保输入是pandas Series，以便使用rolling方法
    original_s = pd.Series(original)
    reconstructed_s = pd.Series(reconstructed)
    # [代码用中文注释] 计算逐点误差的平方
    squared_error = (original_s - reconstructed_s)**2
    # [代码用中文注释] 计算移动窗口内的均方误差 (MSE)
    # min_periods=1 允许在窗口未满时也进行计算（处理边缘情况）
    moving_mse = squared_error.rolling(window=window, center=center, min_periods=1).mean()
    # [代码用中文注释] 计算移动RMSE
    moving_rmse = np.sqrt(moving_mse)
    return moving_rmse.values

def plot_comparison_and_error(
    reconstructed: np.ndarray,
    original: np.ndarray,
    # --- 配置项 (示例，你可以根据你的 DictConfig 调整) ---
    config: Optional[Dict[str, Any]] = None,
    # --- 其他参数 ---
    times: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
    fig_title: str = "时间序列对比与误差分析 (Time Series Comparison and Error Analysis)",
    ts_title: str = "原始序列 vs 重建序列 (Original vs Reconstructed)",
    error_title: str = "时序误差与移动RMSE (Time Series Error and Moving RMSE)",
    y_label_ts: str = "空间平均盐度 (PSU)", # [代码用中文注释] 时间序列子图Y轴标签
    y_label_error: str = "误差 (PSU)",       # [代码用中文注释] 误差子图左侧Y轴标签
    y_label_rmse: str = "移动RMSE (PSU)",   # [代码用中文注释] 误差子图右侧Y轴标签
    x_label: str = "时间 (天)"             # [代码用中文注释] X轴标签
) -> None:

    default_config = {
        'figsize': (12, 6),          # [代码用中文注释] 图形尺寸
        'dpi': 300,                 # [代码用中文注释] 保存分辨率
        'colors': {                 # [代码用中文注释] 颜色配置
            'original': 'tab:blue',
            'reconstructed': 'tab:orange',
            'error': 'tab:green',
            'rmse': 'tab:red',
            'grid': 'lightgray'
        },
        'fill_alpha': 0.2,           # [代码用中文注释] 置信区间填充透明度
        'ma_linewidth': 2.0,         # [代码用中文注释] 移动平均线宽
        'error_linewidth': 1.5,      # [代码用中文注释] 误差线线宽
        'rmse_linewidth': 1.5,       # [代码用中文注释] RMSE线线宽
        'grid_linestyle': '--',      # [代码用中文注释] 网格线样式
        'grid_linewidth': 0.6,       # [代码用中文注释] 网格线线宽
        'fontsize': {               # [代码用中文注释] 字体大小配置
            'title': 16,
            'subtitle': 14,
            'label': 12,
            'tick': 10,
            'legend': 10
        },
        'plot_raw_ts': False,        # [代码用中文注释] 是否绘制原始的、未平滑的时间序列线 (建议False以减少混乱)
        'plot_ci': True,             # [代码用中文注释] 是否绘制置信区间
        'ma_window': 14,             # [代码用中文注释] 移动平均/RMSE的窗口大小
        'ci_window': 14,             # [代码用中文注释] 置信区间的窗口大小 (可以与MA不同)
        'center_window': True,       # [代码用中文注释] 移动窗口是否居中
        'remove_spines': ['top', 'right'] # [代码用中文注释] 移除哪些边框
    }
    # [代码用中文注释] 合并用户配置和默认配置 (简单合并，用户可覆盖默认值)
    cfg = {**default_config, **(config or {})}
    # [代码用中文注释] 处理嵌套字典的合并 (如果需要更精细控制)
    for key in default_config:
        if isinstance(default_config[key], dict):
            cfg[key] = {**default_config[key], **(cfg.get(key) or {})}

    if times is None:
        times = np.arange(len(original))

    # [代码用中文注释] --- 创建图形和子图 ---
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=cfg['figsize'],
                             gridspec_kw={'height_ratios': [2, 1]}) # 上图高一些
    ax1 = axes[0] # [代码用中文注释] 上面的子图：时间序列对比
    ax2 = axes[1] # [代码用中文注释] 下面的子图：误差分析

    fig.suptitle(fig_title, fontsize=cfg['fontsize']['title'])

    # [代码用中文注释] --- 绘制第一个子图：时间序列对比 ---
    ax1.set_title(ts_title, fontsize=cfg['fontsize']['subtitle'])

    # [代码用中文注释] 计算移动平均 (MA) 和置信区间 (CI)
    original_s = pd.Series(original)
    reconstructed_s = pd.Series(reconstructed)
    orig_ma = original_s.rolling(window=cfg['ma_window'], center=cfg['center_window'], min_periods=1).mean()
    recon_ma = reconstructed_s.rolling(window=cfg['ma_window'], center=cfg['center_window'], min_periods=1).mean()

    # [代码用中文注释] ******** 修改开始 ********
    # [代码用中文注释] 检查是否需要绘制置信区间
    if cfg.get('plot_ci', True): # 从配置读取，默认为True
        # [代码用中文注释] 使用移动标准差作为置信区间的示例 (你需要根据你的统计定义调整)
        orig_std = original_s.rolling(window=cfg['ci_window'], center=cfg['center_window'], min_periods=1).std()
        recon_std = reconstructed_s.rolling(window=cfg['ci_window'], center=cfg['center_window'], min_periods=1).std()
        orig_ci = (orig_ma - orig_std, orig_ma + orig_std) # 示例 CI: MA +/- STD
        recon_ci = (recon_ma - recon_std, recon_ma + recon_std) # 示例 CI: MA +/- STD

        # [代码用中文注释] 获取填充透明度，默认0.3
        fill_alpha_value = cfg.get('fill_alpha', 0.3)

        # [代码用中文注释] 绘制置信区间 (恢复原先样式: 同色半透明填充，无hatch)
        ax1.fill_between(times, orig_ci[0], orig_ci[1],
                         color=cfg['colors']['original'],  # 使用与MA线相同的基色
                         alpha=fill_alpha_value,          # 设置透明度
                         label='_nolegend_')               # 不在图例中显示填充区域本身
                         # 如果需要在图例中显示CI，可以使用 label='原始数据 CI'

        ax1.fill_between(times, recon_ci[0], recon_ci[1],
                         color=cfg['colors']['reconstructed'],# 使用与MA线相同的基色
                         alpha=fill_alpha_value,             # 设置透明度
                         label='_nolegend_')                  # 不在图例中显示填充区域本身
                         # 如果需要在图例中显示CI，可以使用 label='重建数据 CI'
    # [代码用中文注释] ******** 修改结束 ********

    # [代码用中文注释] 可选：绘制原始（未平滑）数据线
    if cfg.get('plot_raw_ts', False): # 从配置读取，默认为False
        ax1.plot(times, original, color=cfg['colors']['original'], alpha=0.2, linewidth=0.8, label='原始数据 (Raw)')
        ax1.plot(times, reconstructed, color=cfg['colors']['reconstructed'], alpha=0.2, linewidth=0.8, label='重建数据 (Raw)')

    # [代码用中文注释] 绘制移动平均线 (使用不同线型)
    (line1,) = ax1.plot(times, orig_ma, color=cfg['colors']['original'], linewidth=cfg['ma_linewidth'],
                        linestyle='-', label=f'原始数据 ({cfg["ma_window"]}-day MA)')
    (line2,) = ax1.plot(times, recon_ma, color=cfg['colors']['reconstructed'], linewidth=cfg['ma_linewidth'],
                        linestyle='--', label=f'重建数据 ({cfg["ma_window"]}-day MA)')


    ax1.set_ylabel(y_label_ts, fontsize=cfg['fontsize']['label'])
    ax1.tick_params(axis='y', labelsize=cfg['fontsize']['tick'])
    ax1.grid(True, color=cfg['colors']['grid'], linestyle=cfg['grid_linestyle'], linewidth=cfg['grid_linewidth'])
    # [代码用中文注释] 合并图例 (如果CI也加了label)
    handles, labels = ax1.get_legend_handles_labels()
    # [代码用中文注释] (或者只显示MA的图例)
    # handles = [line1, line2]
    # labels = [h.get_label() for h in handles]
    ax1.legend(handles, labels, fontsize=cfg['fontsize']['legend'], loc='best')

    # [代码用中文注释] --- 绘制第二个子图：误差分析 ---
    ax2.set_title(error_title, fontsize=cfg['fontsize']['subtitle'])

    # [代码用中文注释] 计算逐点误差
    point_error = original - reconstructed
    # [代码用中文注释] 计算移动RMSE
    moving_rmse = calculate_moving_rmse(original, reconstructed, window=cfg['ma_window'], center=cfg['center_window'])

    # [代码用中文注释] 绘制逐点误差 (左Y轴)
    (line_err,) = ax2.plot(times, point_error, color=cfg['colors']['error'], linewidth=cfg['error_linewidth'],
                           linestyle='-', label=y_label_error)
    ax2.set_ylabel(y_label_error, fontsize=cfg['fontsize']['label'], color=cfg['colors']['error'])
    ax2.tick_params(axis='y', labelcolor=cfg['colors']['error'], labelsize=cfg['fontsize']['tick'])
    ax2.axhline(0, color='gray', linestyle=':', linewidth=0.8) # [代码用中文注释] 添加y=0参考线

    # [代码用中文注释] 创建右侧Y轴
    ax3 = ax2.twinx()
    # [代码用中文注释] 绘制移动RMSE (右Y轴)
    (line_rmse,) = ax3.plot(times, moving_rmse, color=cfg['colors']['rmse'], linewidth=cfg['rmse_linewidth'],
                            linestyle='-.', label=y_label_rmse)
    ax3.set_ylabel(y_label_rmse, fontsize=cfg['fontsize']['label'], color=cfg['colors']['rmse'])
    ax3.tick_params(axis='y', labelcolor=cfg['colors']['rmse'], labelsize=cfg['fontsize']['tick'])
    # [代码用中文注释] 确保RMSE从0开始显示可能更好
    ax3.set_ylim(bottom=0)

    # [代码用中文注释] 设置X轴
    ax2.set_xlabel(x_label, fontsize=cfg['fontsize']['label'])
    ax2.tick_params(axis='x', labelsize=cfg['fontsize']['tick'])

    # [代码用中文注释] 合并误差图的图例
    handles_err = [line_err, line_rmse]
    labels_err = [h.get_label() for h in handles_err]
    ax2.legend(handles_err, labels_err, fontsize=cfg['fontsize']['legend'], loc='best')

    ax2.grid(True, color=cfg['colors']['grid'], linestyle=cfg['grid_linestyle'], linewidth=cfg['grid_linewidth'])

    # [代码用中文注释] --- 移除边框 ---
    for ax in [ax1, ax2, ax3]:
        for spine_pos in cfg['remove_spines']:
            if spine_pos in ax.spines:
                ax.spines[spine_pos].set_visible(False)
        # [代码用中文注释] 对于双轴图，需要单独处理 ax3 的边框使其与 ax2 匹配
        if ax == ax3:
             for spine_pos in cfg['remove_spines']:
                 if spine_pos == 'left': # 如果移除了左边框，也要移除ax3的（虽然它在右边显示）
                      ax.spines['left'].set_visible(False)
                 if spine_pos == 'right': # 如果移除了右边框，也要移除ax3的
                      ax.spines['right'].set_visible(False)

    # [代码用中文注释] 调整布局防止标签重叠
    fig.tight_layout(rect=[0, 0.03, 1, 0.95]) # rect 留出空间给 fig_title

    # [代码用中文注释] --- 保存或显示 ---
    if save_path:
        plt.savefig(save_path, dpi=cfg['dpi'], bbox_inches='tight')
        plt.close(fig) # [代码用中文注释] 保存后关闭图形，防止在notebook中重复显示
        print(f"图像已保存至: {save_path}")
    else:
        plt.show()

def plot_time_series_comparison(
    reconstructed: np.ndarray,
    original: np.ndarray,
    cfg: DictConfig,
    times: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
    title: str = "Comparison of Original and Reconstructed Time Series in "
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
    plt.xlabel('Day')
    plt.ylabel('Spatially Averaged Salinity (PSU)')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, dpi=cfg.evaluation.visualization.dpi,
                   bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_comparison_and_error_adjusted_style(
    reconstructed: np.ndarray,
    original: np.ndarray,
    config: Optional[Dict[str, Any]] = None,
    times: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
    fig_title: str = "Time Series Comparison and Error Analysis",
    ts_title: str = "Original vs Reconstructed",
    error_title: str = "Time Series Error and Moving RMSE",
    y_label_ts: str = "空间平均盐度 (PSU)",
    y_label_error: str = "误差 (PSU)", # 标签现在代表原始误差和MA误差
    y_label_rmse: str = "移动RMSE (PSU)",
    x_label: str = "时间 (天)",
) -> None:
    """
    绘制时序对比与误差分析图。
    上部子图风格模拟旧版。
    下部子图增加误差的移动平均线。
    """
    # --- 默认配置 (增加误差MA和原始误差样式) ---
    default_config = {
        'figsize': (12, 8), # 调整figsize可能有助于图例显示
        'dpi': 300,
        'colors': {
            'original': 'tab:blue', 'reconstructed': 'tab:orange',
            'error': 'tab:green',       # 原始误差颜色
            'error_ma': '#2ca02c',      # 误差MA颜色 (深绿色) 或 'darkgreen'
            'rmse': 'tab:red',
            'grid': 'lightgray'
        },
        'fill_alpha': 0.3,
        'raw_ts_alpha': 0.3,
        'raw_ts_linewidth': 0.8,
        'ma_linewidth': 2.0,
        # 误差相关线条样式
        'error_raw_alpha': 0.4,      # <--- 原始误差透明度 (调低)
        'error_raw_linewidth': 1.0,  # <--- 原始误差线宽 (调细)
        'error_ma_linewidth': 1.8,   # <--- 误差MA线宽
        'error_ma_linestyle': '-',   # <--- 误差MA线型 (实线)
        'rmse_linewidth': 1.5,
        'rmse_linestyle': '-.',      # <--- RMSE线型 (点划线保持不变)
        'grid_linestyle': '--',
        'grid_linewidth': 0.6,
        'fontsize': { 'title': 16, 'subtitle': 14, 'label': 12, 'tick': 10, 'legend': 9 }, # 调整图例字体大小
        'plot_raw_ts': True,
        'plot_ci': True,
        'ma_window': 14, 'ci_window': 14, 'center_window': True,
        'remove_spines': ['top', 'right'],
        'error_legend_loc': 'best', # <--- 误差图图例位置
        'ts_legend_loc': 'best'     # <--- 对比图图例位置
    }
    cfg = {**default_config, **(config or {})}
    for key in default_config:
        if isinstance(default_config[key], dict):
            cfg[key] = {**default_config[key], **(cfg.get(key) or {})}

    if times is None: times = np.arange(len(original))

    # --- 创建图形和子图 (设为等高) ---
    fig, axes = plt.subplots(2, 1, sharex=True,
                             figsize=cfg['figsize']) # 移除 gridspec_kw 使其等高

    ax1, ax2 = axes[0], axes[1]
    if fig_title: fig.suptitle(fig_title, fontsize=cfg['fontsize']['title'])
    if ts_title: ax1.set_title(ts_title, fontsize=cfg['fontsize']['subtitle'])

    # --- 绘制第一个子图：时间序列对比 ---
    # ... [绘制 ax1 的代码不变，但使用 ts_legend_loc] ...
    original_s = pd.Series(original); reconstructed_s = pd.Series(reconstructed)
    orig_ma = original_s.rolling(window=cfg['ma_window'], center=cfg['center_window'], min_periods=1).mean()
    recon_ma = reconstructed_s.rolling(window=cfg['ma_window'], center=cfg['center_window'], min_periods=1).mean()
    if cfg.get('plot_ci', True): # 绘制 CI
        orig_std = original_s.rolling(window=cfg['ci_window'], center=cfg['center_window'], min_periods=1).std()
        recon_std = reconstructed_s.rolling(window=cfg['ci_window'], center=cfg['center_window'], min_periods=1).std()
        orig_ci = (orig_ma - orig_std, orig_ma + orig_std); recon_ci = (recon_ma - recon_std, recon_ma + recon_std)
        fill_alpha_value = cfg.get('fill_alpha', 0.3)
        ax1.fill_between(times, orig_ci[0], orig_ci[1], color=cfg['colors']['original'], alpha=fill_alpha_value, label='_nolegend_')
        ax1.fill_between(times, recon_ci[0], recon_ci[1], color=cfg['colors']['reconstructed'], alpha=fill_alpha_value, label='_nolegend_')
    if cfg.get('plot_raw_ts', True): # 绘制原始细线
        ax1.plot(times, original, color=cfg['colors']['original'], alpha=cfg['raw_ts_alpha'], linewidth=cfg['raw_ts_linewidth'], label='Original')
        ax1.plot(times, reconstructed, color=cfg['colors']['reconstructed'], alpha=cfg['raw_ts_alpha'], linewidth=cfg['raw_ts_linewidth'], label='Reconstructed')
    # 绘制 MA 线
    ma_label_orig = f'Original ({cfg["ma_window"]}-day MA)'; ma_label_recon = f'Reconstructed ({cfg["ma_window"]}-day MA)'
    (line1,) = ax1.plot(times, orig_ma, color=cfg['colors']['original'], linewidth=cfg['ma_linewidth'], linestyle='-', label=ma_label_orig)
    (line2,) = ax1.plot(times, recon_ma, color=cfg['colors']['reconstructed'], linewidth=cfg['ma_linewidth'], linestyle='-', label=ma_label_recon)
    # 设置 ax1 其他属性
    ax1.set_ylabel(y_label_ts, fontsize=cfg['fontsize']['label'])
    ax1.tick_params(axis='y', labelsize=cfg['fontsize']['tick'])
    ax1.grid(True, color=cfg['colors']['grid'], linestyle=cfg['grid_linestyle'], linewidth=cfg['grid_linewidth'])
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles, labels, fontsize=cfg['fontsize']['legend'], loc=cfg['ts_legend_loc']) # 使用配置的位置

    # --- 绘制第二个子图：误差分析 (关键修改处) ---
    if error_title: ax2.set_title(error_title, fontsize=cfg['fontsize']['subtitle'])

    # 计算逐点误差 和 误差的移动平均
    point_error = original - reconstructed
    point_error_s = pd.Series(point_error)
    error_ma = point_error_s.rolling(window=cfg['ma_window'], center=cfg['center_window'], min_periods=1).mean()

    # 计算移动RMSE (右轴数据)
    moving_rmse = calculate_moving_rmse(original, reconstructed, window=cfg['ma_window'], center=cfg['center_window'])

    # 绘制调整后的原始误差线 (左Y轴)
    (line_err,) = ax2.plot(times, point_error,
                           color=cfg['colors']['error'],
                           linewidth=cfg['error_raw_linewidth'], # 使用新配置的线宽
                           alpha=cfg['error_raw_alpha'],       # 使用新配置的透明度
                           linestyle='-', label=y_label_error) # 保持标签不变
    ax2.set_ylabel(y_label_error, fontsize=cfg['fontsize']['label'], color=cfg['colors']['error']) # Y轴标签颜色仍用原始误差颜色
    ax2.tick_params(axis='y', labelcolor=cfg['colors']['error'], labelsize=cfg['fontsize']['tick'])
    ax2.axhline(0, color='gray', linestyle=':', linewidth=0.8)

    # 绘制新增的误差移动平均线 (左Y轴)
    error_ma_label = f'{y_label_error} MA ({cfg["ma_window"]}-day)'
    (line_err_ma,) = ax2.plot(times, error_ma,
                              color=cfg['colors']['error_ma'],        # 使用误差MA的颜色
                              linewidth=cfg['error_ma_linewidth'],  # 使用误差MA的线宽
                              linestyle=cfg['error_ma_linestyle'], # 使用误差MA的线型
                              label=error_ma_label)                # 新的标签

    # 创建右侧Y轴并绘制移动RMSE
    ax3 = ax2.twinx()
    (line_rmse,) = ax3.plot(times, moving_rmse,
                            color=cfg['colors']['rmse'],
                            linewidth=cfg['rmse_linewidth'],
                            linestyle=cfg['rmse_linestyle'], # 使用配置的RMSE线型
                            label=y_label_rmse)
    ax3.set_ylabel(y_label_rmse, fontsize=cfg['fontsize']['label'], color=cfg['colors']['rmse'])
    ax3.tick_params(axis='y', labelcolor=cfg['colors']['rmse'], labelsize=cfg['fontsize']['tick'])
    ax3.set_ylim(bottom=0)

    # 设置X轴
    ax2.set_xlabel(x_label, fontsize=cfg['fontsize']['label'])
    ax2.tick_params(axis='x', labelsize=cfg['fontsize']['tick'])

    # --- 修改：合并误差图的图例 (包含三条线) ---
    # 选择要在图例中显示哪些线 (可以根据需要调整)
    handles_to_show = [line_err, line_err_ma, line_rmse] # 显示全部三条
    # handles_to_show = [line_err_ma, line_rmse] # 只显示两条MA/RMSE曲线可能更清晰
    labels_to_show = [h.get_label() for h in handles_to_show]

    # 将图例放在 ax2 上 (因为 ax3 是 twinx)
    ax2.legend(handles_to_show, labels_to_show,
               fontsize=cfg['fontsize']['legend'],
               loc=cfg['error_legend_loc']) # 使用配置的位置

    ax2.grid(True, color=cfg['colors']['grid'], linestyle=cfg['grid_linestyle'], linewidth=cfg['grid_linewidth'])


    # --- 移除边框 ---
    # ... [移除边框的代码不变] ...
    remove_spines_list = cfg.get('remove_spines', ['top', 'right'])
    for ax in [ax1, ax2, ax3]:
        all_spines = ['top', 'right', 'bottom', 'left'];
        for spine_pos in remove_spines_list:
            if spine_pos in ax.spines: ax.spines[spine_pos].set_visible(False)
        if ax == ax3:
            if 'right' in remove_spines_list and 'right' in ax.spines: ax.spines['right'].set_visible(False)
            if 'left' in remove_spines_list and 'left' in ax.spines: ax.spines['left'].set_visible(False)

    # --- 调整布局 ---
    # ... [调整布局代码不变] ...
    try:
        rect_top = 0.95 if fig_title else 1.0
        fig.tight_layout(rect=[0, 0.03, 1, rect_top])
    except Exception as e: print(f"Tight layout failed: {e}")

    # --- 保存或显示 ---
    # ... [保存显示代码不变] ...
    if save_path: plt.savefig(save_path, dpi=cfg['dpi'], bbox_inches='tight'); plt.close(fig); print(f"图像已保存至: {save_path}")
    else: plt.show()

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

def plot_spatial_rmse(
    spatial_rmse: np.ndarray,
    cfg: DictConfig,
    mask: np.ndarray,
    save_path: str,
    title: str = "空间RMSE分布 (Spatial RMSE Distribution)"):

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
    add_colorbar(fig, pcm, ax, label="RMSE (PSU)")

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

# 创建评估/visualization.py中的新函数

def plot_multi_horizon_comparison(predictions, targets, horizons, cfg, save_path):
    """
    绘制多步预测结果对比图
    """
    plt.figure(figsize=(12, 8))
    
    # 绘制实际值
    plt.plot(range(len(targets)), targets, 'k-', label='实际值', linewidth=2)
    
    # 绘制不同预测步长的结果
    colors = ['r', 'g', 'b', 'c', 'm']
    for i, h in enumerate(horizons):
        color = colors[i % len(colors)]
        plt.plot(range(len(predictions[h])), predictions[h], f'{color}-', 
                 label=f'{h}天预测', alpha=0.7)
    
    plt.title('多步预测对比')
    plt.xlabel('时间步')
    plt.ylabel('值')
    plt.legend()
    plt.grid(True)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

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

def plot_multi_step_predictions(metrics, horizon_steps, cfg, save_dir):
    """
    绘制多步预测比较图
    
    参数:
        metrics: 评估指标字典
        horizon_steps: 预测步长列表
        cfg: 配置对象
        save_dir: 保存目录
    """
    import matplotlib.pyplot as plt
    import os
    
    # 设置图表样式
    plt.style.use('ggplot')
    
    # 绘制不同预测步长的RMSE/MAE比较
    plt.figure(figsize=(10, 6))
    
    # RMSE比较
    rmse_values = [metrics[h]['rmse'] for h in horizon_steps]
    mae_values = [metrics[h]['mae'] for h in horizon_steps]
    
    plt.subplot(1, 2, 1)
    plt.bar(range(len(horizon_steps)), rmse_values, tick_label=[f"{h}天" for h in horizon_steps])
    plt.title('不同预测步长的RMSE比较')
    plt.ylabel('RMSE')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.bar(range(len(horizon_steps)), mae_values, tick_label=[f"{h}天" for h in horizon_steps])
    plt.title('不同预测步长的MAE比较')
    plt.ylabel('MAE')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'multi_step_metrics_comparison.png'), dpi=300)
    plt.close()
    
    # 为每个预测步长绘制预测与实际值比较图
    for h in horizon_steps:
        plt.figure(figsize=(12, 6))
        
        # 选择前100个样本进行可视化
        n_samples = min(100, len(metrics[h]['predictions']))
        time_steps = range(n_samples)
        
        # 为每个特征绘制一条线
        n_features = metrics[h]['predictions'].shape[1]
        for i in range(n_features):
            plt.subplot(n_features, 1, i+1)
            
            preds = metrics[h]['predictions'][:n_samples, i]
            targets = metrics[h]['targets'][:n_samples, i]
            
            plt.plot(time_steps, targets, 'b-', label='实际值')
            plt.plot(time_steps, preds, 'r-', label='预测值')
            
            plt.title(f'特征 {i+1} - {h}天预测')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'prediction_vs_actual_{h}day.png'), dpi=300)
        plt.close()

if __name__ == "__main__":
    main()