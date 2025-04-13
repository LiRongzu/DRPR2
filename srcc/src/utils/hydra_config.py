"""
配置实用工具。
"""
import os
from dataclasses import dataclass
from typing import Dict, Any, Optional
from omegaconf import DictConfig, OmegaConf

@dataclass
class DrprConfig:
    """为项目配置提供结构化访问的配置类。"""

    def __init__(self, config: Dict[str, Any]):
        """
        从配置字典初始化配置。

        参数:
            config: 配置字典
        """
        self._config = config
        self._setup_attributes()

    @classmethod
    def from_hydra_config(cls, cfg: DictConfig) -> 'DrprConfig':
        """
        从Hydra配置对象创建DrprConfig实例。

        参数:
            cfg: Hydra配置对象。

        返回:
            DrprConfig实例。
        """
        # 转换为普通字典以避免OmegaConf访问限制
        config_dict = OmegaConf.to_container(cfg, resolve=True)
        return cls(config_dict)

    def _setup_attributes(self):
        """设置实例属性以允许点访问。"""
        for key, value in self._config.items():
            if isinstance(value, dict):
                setattr(self, key, DrprConfig(value))
            else:
                setattr(self, key, value)

    def get(self, key: str, default: Any = None) -> Any:
        """
        安全地获取配置值。

        参数:
            key: 配置键
            default: 如果键不存在时返回的默认值

        返回:
            配置值或默认值
        """
        return getattr(self, key, default)

    def __getattr__(self, name: str) -> Any:
        """
        处理属性访问。

        参数:
            name: 属性名

        返回:
            属性值

        抛出:
            AttributeError: 如果属性不存在
        """
        try:
            return self._config[name]
        except KeyError:
            raise AttributeError(f"'{self.__class__.__name__}' 对象没有属性 '{name}'")

    def __str__(self) -> str:
        """返回配置的字符串表示。"""
        return str(self._config)

    def __repr__(self) -> str:
        """返回配置的详细字符串表示。"""
        return f"DrprConfig({self._config})"