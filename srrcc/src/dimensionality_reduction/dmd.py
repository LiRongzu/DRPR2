import numpy as np
import matplotlib.pyplot as plt
from pydmd import DMD
from sklearn.preprocessing import StandardScaler
import time

class DMDDimensionalityReduction:
    """
    基于动态模态分解(DMD)的降维方法
    
    DMD是一种用于分析动态系统的技术，可以从时间序列数据中提取关键动态特征和模态。
    它特别适合于提取时变系统中的时空相干结构，如流体动力学中的相干结构。
    """
    
    def __init__(self, n_modes=10, svd_rank=None, tlsq_rank=None, exact=True, random_seed=None):
        """
        初始化DMD降维模型
        
        参数:
            n_modes: 要保留的DMD模态数量
            svd_rank: SVD分解的秩，如果为None则自动确定
            tlsq_rank: 总体最小二乘的秩，用于去噪，如果为None则不使用
            exact: 是否使用精确DMD
            random_seed: 随机种子
        """
        self.n_modes = n_modes
        self.svd_rank = svd_rank
        self.tlsq_rank = tlsq_rank
        self.exact = exact
        self.random_seed = random_seed
        self.dmd = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.training_time = None
        self.eigenvalues = None
        self.modes = None
        self.dynamics = None
        
        # 设置随机种子
        if random_seed is not None:
            np.random.seed(random_seed)
        
    def fit(self, X, feature_names=None):
        """
        训练DMD模型
        
        参数:
            X: 输入数据，形状为(样本数, 特征数)
            feature_names: 特征名称列表
            
        返回:
            self: 训练好的模型
        """
        # 记录特征名称
        self.feature_names = feature_names
        
        # 数据预处理：标准化
        X_scaled = self.scaler.fit_transform(X)
        
        # 重塑数据为DMD所需的格式：(特征数, 样本数)
        X_reshaped = X_scaled.T
        
        # 记录训练开始时间
        start_time = time.time()
        
        # 初始化并训练DMD模型
        self.dmd = DMD(svd_rank=self.svd_rank, tlsq_rank=self.tlsq_rank, exact=self.exact)
        self.dmd.fit(X_reshaped)
        
        # 记录训练时间
        self.training_time = time.time() - start_time
        
        # 提取DMD特征
        self.eigenvalues = self.dmd.eigs
        self.modes = self.dmd.modes
        self.dynamics = self.dmd.dynamics
        
        return self
    
    def transform(self, X):
        """
        将数据转换到低维空间
        
        参数:
            X: 输入数据，形状为(样本数, 特征数)
            
        返回:
            X_reduced: 降维后的数据，形状为(样本数, n_modes)
        """
        # 数据预处理：标准化
        X_scaled = self.scaler.transform(X)
        
        # 重塑数据为DMD所需的格式：(特征数, 样本数)
        X_reshaped = X_scaled.T
        
        # 使用DMD模态投影数据
        # 计算每个时间点的模态系数
        X_reduced = np.zeros((X.shape[0], self.n_modes))
        
        for i in range(X.shape[0]):
            # 对每个时间点，计算其在DMD模态上的投影
            coeffs = np.linalg.lstsq(self.modes[:, :self.n_modes], X_reshaped[:, i], rcond=None)[0]
            X_reduced[i] = coeffs
        
        return X_reduced
    
    def inverse_transform(self, X_reduced):
        """
        将低维数据转换回原始空间
        
        参数:
            X_reduced: 降维后的数据，形状为(样本数, n_modes)
            
        返回:
            X_reconstructed: 重构的数据，形状为(样本数, 特征数)
        """
        # 使用DMD模态重构数据
        X_reconstructed = np.zeros((X_reduced.shape[0], self.modes.shape[0]))
        
        for i in range(X_reduced.shape[0]):
            # 使用模态系数重构原始数据
            X_reconstructed[i] = np.real(self.modes[:, :self.n_modes] @ X_reduced[i])
        
        # 反标准化
        X_reconstructed = self.scaler.inverse_transform(X_reconstructed)
        
        return X_reconstructed
    
    def fit_transform(self, X, feature_names=None):
        """
        训练模型并转换数据
        
        参数:
            X: 输入数据，形状为(样本数, 特征数)
            feature_names: 特征名称列表
            
        返回:
            X_reduced: 降维后的数据，形状为(样本数, n_modes)
        """
        self.fit(X, feature_names)
        return self.transform(X)
    
    def get_feature_importance(self):
        """
        获取特征重要性
        
        返回:
            feature_importance: 特征重要性数组
        """
        if self.modes is None:
            return None
        
        # 计算每个特征在所有模态中的平均贡献
        feature_importance = np.abs(self.modes).mean(axis=1)
        
        return feature_importance
    
    def plot_modes(self, n_modes=None, figsize=(12, 8)):
        """
        绘制DMD模态
        
        参数:
            n_modes: 要绘制的模态数量，默认为None（绘制所有模态）
            figsize: 图形大小
        """
        if self.modes is None:
            print("模型尚未训练")
            return
        
        if n_modes is None:
            n_modes = min(self.n_modes, 10)  # 默认最多绘制10个模态
        
        plt.figure(figsize=figsize)
        for i in range(n_modes):
            plt.subplot(n_modes, 1, i+1)
            plt.plot(np.real(self.modes[:, i]))
            plt.title(f'模态 {i+1}')
            if self.feature_names is not None and len(self.feature_names) == self.modes.shape[0]:
                plt.xticks(range(len(self.feature_names)), self.feature_names, rotation=90)
        
        plt.tight_layout()
        plt.show()
    
    def plot_eigenvalues(self, figsize=(8, 8)):
        """
        绘制DMD特征值
        
        参数:
            figsize: 图形大小
        """
        if self.eigenvalues is None:
            print("模型尚未训练")
            return
        
        plt.figure(figsize=figsize)
        
        # 绘制单位圆
        theta = np.linspace(0, 2*np.pi, 100)
        plt.plot(np.cos(theta), np.sin(theta), 'k--')
        
        # 绘制特征值
        plt.scatter(np.real(self.eigenvalues), np.imag(self.eigenvalues), c='r', alpha=0.5)
        plt.xlabel('实部')
        plt.ylabel('虚部')
        plt.title('DMD特征值')
        plt.grid(True)
        plt.axis('equal')
        
        plt.tight_layout()
        plt.show()
    
    def get_model_info(self):
        """
        获取模型信息
        
        返回:
            info: 包含模型信息的字典
        """
        info = {
            'method': 'DMD',
            'n_modes': self.n_modes,
            'svd_rank': self.svd_rank,
            'tlsq_rank': self.tlsq_rank,
            'exact': self.exact,
            'training_time': self.training_time
        }
        
        return info