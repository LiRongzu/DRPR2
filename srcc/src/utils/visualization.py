"""可视化工具。"""
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Optional, Union, List, Dict, Any, Tuple
from omegaconf import DictConfig
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)
# src/utils/visualization.py
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from omegaconf import DictConfig, OmegaConf
import logging
import numpy as np
import matplotlib.colors as mcolors # 用于 set_bad

logger = logging.getLogger(__name__)

# --- 新的 Cartopy 相关函数 ---

def get_cartopy_projection(cfg: DictConfig):
    """从配置获取 Cartopy CRS 投影对象。"""
    map_cfg = cfg.visualization.map
    proj_name = map_cfg.get("cartopy_projection", "PlateCarree")
    try:
        proj_class = getattr(ccrs, proj_name)
        kwargs = {}
        # 检查是否需要中心经度
        if 'central_longitude' in map_cfg:
            # 只有某些投影接受 central_longitude
            accepted_projs = ['Mercator', 'LambertConformal', 'Stereographic', 'Orthographic'] # 可能不全
            if proj_name in accepted_projs:
                 kwargs['central_longitude'] = map_cfg.central_longitude
            elif map_cfg.get("central_longitude") is not None:
                 logger.warning(f"Projection {proj_name} might not accept 'central_longitude'.")

        return proj_class(**kwargs)
    except AttributeError:
        logger.error(f"无效的 Cartopy 投影名称: {proj_name}. 使用 PlateCarree 作为备选。")
        return ccrs.PlateCarree()
    except Exception as e:
        logger.error(f"创建投影 {proj_name} 时出错: {e}. 使用 PlateCarree 作为备选。")
        return ccrs.PlateCarree()

def get_data_crs(cfg: DictConfig):
    """从配置获取输入数据的 Cartopy CRS 对象。"""
    map_cfg = cfg.visualization.map
    crs_name = map_cfg.get("data_crs", "PlateCarree")
    try:
        return getattr(ccrs, crs_name)()
    except AttributeError:
        logger.error(f"无效的数据 CRS 名称: {crs_name}. 使用 PlateCarree 作为备选。")
        return ccrs.PlateCarree()

def add_cartopy_features(ax: plt.Axes, cfg: DictConfig):
    """向 Cartopy GeoAxes 添加地图特征。"""
    map_cfg = cfg.visualization.map
    features_cfg = map_cfg.get("features", {})

    feature_map = {
        "coastline": cfeature.COASTLINE,
        "land": cfeature.LAND,
        "ocean": cfeature.OCEAN,
        "borders": cfeature.BORDERS,
        "rivers": cfeature.RIVERS,
        "lakes": cfeature.LAKES
    }

    for feature_name, feature_obj in feature_map.items():
        feature_conf = features_cfg.get(feature_name)
        # 检查配置是否存在且 enabled 标志为 True
        if isinstance(feature_conf, DictConfig) and feature_conf.get("enabled", False):
            scale = feature_conf.get("scale", "50m") # 默认 '50m'
            kwargs = {k: v for k, v in feature_conf.items() if k not in ['scale', 'enabled']}

            # 修正颜色参数: Cartopy 特征通常用 facecolor/edgecolor
            if 'color' in kwargs:
                if feature_name in ['land', 'ocean', 'lakes']:
                     kwargs['facecolor'] = kwargs.pop('color')
                elif feature_name in ['coastline', 'borders', 'rivers']:
                     kwargs['edgecolor'] = kwargs.pop('color')

            try:
                ax.add_feature(feature_obj.with_scale(scale), **kwargs)
                # logger.debug(f"Added feature '{feature_name}' with scale '{scale}' and options {kwargs}")
            except ValueError as e: # 通常是 scale 无效
                logger.warning(f"无法以 scale '{scale}' 添加特征 '{feature_name}': {e}. 尝试默认 scale。")
                try:
                    ax.add_feature(feature_obj, **kwargs)
                except Exception as e_inner:
                     logger.error(f"添加特征 '{feature_name}' (默认 scale) 失败: {e_inner}")
            except Exception as e:
                logger.error(f"添加特征 '{feature_name}' 失败: {e}")


def add_gridlines(ax: plt.Axes, cfg: DictConfig):
    """向 Cartopy GeoAxes 添加经纬网格线。"""
    map_cfg = cfg.visualization.map
    grid_cfg = map_cfg.get("gridlines")

    if isinstance(grid_cfg, DictConfig) and grid_cfg.get("enabled", False):
        try:
            gl = ax.gridlines(crs=get_data_crs(cfg), draw_labels=grid_cfg.get("draw_labels", False),
                              linewidth=grid_cfg.get("linewidth", 0.5),
                              color=grid_cfg.get("color", 'gray'),
                              alpha=grid_cfg.get("alpha", 0.5),
                              linestyle=grid_cfg.get("linestyle", '--'))
            # 控制标签位置
            gl.top_labels = False
            gl.right_labels = False
            # 控制标签格式
            gl.xformatter = LONGITUDE_FORMATTER
            gl.yformatter = LATITUDE_FORMATTER
            # logger.debug("Added gridlines.")
        except Exception as e:
             logger.error(f"添加网格线失败: {e}")


def add_colorbar(fig, mappable, ax, label="", orientation='vertical', shrink=0.6,aspect=20,  pad=0.04):
    """
    向指定的 Axes 添加颜色条。

    Args:
        fig: Matplotlib Figure 对象。
        mappable: 颜色映射的对象 (例如 pcolormesh 或 contourf 的返回值)。
        ax: Matplotlib Axes 对象。
        label: 颜色条的标签 (可选)。
        orientation: 颜色条方向 ('vertical' 或 'horizontal')。
        fraction: 颜色条相对于原始 Axes 的大小比例。
        pad: 颜色条与 Axes 之间的间距。

    Returns:
        创建的 Colorbar 对象。
    """
    try:
        cbar = fig.colorbar(mappable, ax=ax, orientation=orientation, shrink=shrink,aspect=aspect, pad=pad)
        if label:
            cbar.set_label(label)
        logger.debug("颜色条添加成功。")
        return cbar
    except Exception as e:
        logger.error(f"添加颜色条失败: {e}", exc_info=True)
        return None



def setup_figure(
    nrows: int = 1,
    ncols: int = 1,
    figsize: Optional[Tuple[int, int]] = None,
    **kwargs
) -> Tuple[plt.Figure, Union[plt.Axes, np.ndarray]]:
    """设置matplotlib图形。"""
    if figsize is None:
        figsize = (6 * ncols, 4 * nrows)
    return plt.subplots(nrows, ncols, figsize=figsize, **kwargs)

def save_figure(
    fig: plt.Figure,
    name: str,
    config: DictConfig,
    timestamp: bool = True,
    close: bool = True
) -> None:
    """
    保存图形到Hydra运行目录下的指定位置。

    参数:
        fig: matplotlib图形对象
        name: 图形基本名称
        config: Hydra配置对象
        timestamp: 是否在文件名中添加时间戳
        close: 保存后是否关闭图形
    """
    if not config.visualization.save_figures:
        return

    # 创建保存目录（相对于Hydra运行目录）
    save_dir = config.visualization.save.get("dir", "figures")
    os.makedirs(save_dir, exist_ok=True)

    # 构建文件名
    ext = config.visualization.get("output_format", "png")
    if timestamp:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{name}_{ts}.{ext}"
    else:
        filename = f"{name}.{ext}"

    # 保存图形
    save_path = os.path.join(save_dir, filename)
    fig.savefig(
        save_path,
        dpi=config.visualization.get("dpi", 300),
        bbox_inches='tight'
    )

    if close:
        plt.close(fig)

def plot_field_timeseries(
    predictions: np.ndarray,
    ground_truth: Optional[np.ndarray] = None,
    timestamps: Optional[np.ndarray] = None,
    config: Optional[DictConfig] = None,
    title: str = "时间序列对比",
    **kwargs
) -> plt.Figure:
    """绘制预测值与真实值的时间序列对比图。"""
    fig, ax = setup_figure()
    
    if timestamps is None:
        timestamps = np.arange(len(predictions))
    
    # 获取样式设置
    if config is not None:
        style = config.visualization.timeseries
        colors = config.visualization.colors
    else:
        style = {}
        colors = {"prediction": "red", "ground_truth": "blue"}
    
    # 绘制预测值
    ax.plot(timestamps, predictions, 
            label='预测值',
            color=colors.get("prediction", "red"),
            linestyle=style.get("line_style", "-"),
            marker=style.get("marker", "."),
            alpha=style.get("alpha", 0.8))
    
    # 如果有真实值，也绘制出来
    if ground_truth is not None:
        ax.plot(timestamps, ground_truth, 
                label='真实值',
                color=colors.get("ground_truth", "blue"),
                linestyle=style.get("line_style", "-"),
                marker=style.get("marker", "."),
                alpha=style.get("alpha", 0.8))
    
    ax.set_title(title)
    ax.grid(style.get("grid", True))
    ax.legend()
    
    if config is not None:
        save_figure(fig, "timeseries_comparison", config)
    
    return fig

def plot_reconstruction_comparison(
    original: np.ndarray,
    reconstructed: np.ndarray,
    config: Optional[DictConfig] = None,
    title: str = "重建结果对比",
    **kwargs
) -> plt.Figure:
    """绘制原始场与重建场的对比图。"""
    fig, (ax1, ax2) = setup_figure(1, 2, figsize=(12, 5))
    
    # 获取颜色映射设置
    if config is not None:
        cmap = config.visualization.som.get("colormap", "viridis")
    else:
        cmap = "viridis"
    
    # 绘制原始场
    im1 = ax1.imshow(original, cmap=cmap)
    ax1.set_title("原始场")
    plt.colorbar(im1, ax=ax1)
    
    # 绘制重建场
    im2 = ax2.imshow(reconstructed, cmap=cmap)
    ax2.set_title("重建场")
    plt.colorbar(im2, ax=ax2)
    
    fig.suptitle(title)
    
    if config is not None:
        save_figure(fig, "reconstruction_comparison", config)
    
    return fig

def plot_som_components(
    model: Any,
    config: Optional[DictConfig] = None,
    **kwargs
) -> plt.Figure:
    """绘制SOM的分量平面。"""
    # 获取SOM权重
    weights = model.get_weights()  # 假设模型有此方法
    n_components = weights.shape[-1]
    
    # 创建子图
    fig, axes = setup_figure(
        nrows=(n_components + 1) // 2,
        ncols=2,
        figsize=(12, 4 * ((n_components + 1) // 2))
    )
    axes = axes.flatten()
    
    # 获取可视化设置
    if config is not None:
        cmap = config.visualization.som.get("colormap", "viridis")
    else:
        cmap = "viridis"
    
    # 绘制每个分量
    for i in range(n_components):
        im = axes[i].imshow(weights[:, :, i], cmap=cmap)
        axes[i].set_title(f"分量 {i+1}")
        plt.colorbar(im, ax=axes[i])
    
    # 如果子图数量多于分量数，删除多余的子图
    for i in range(n_components, len(axes)):
        fig.delaxes(axes[i])
    
    fig.suptitle("SOM分量平面")
    plt.tight_layout()
    
    if config is not None:
        save_figure(fig, "som_components", config)
    
    return fig