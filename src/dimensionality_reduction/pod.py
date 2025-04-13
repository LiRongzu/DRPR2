import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import time

class PODDimensionalityReduction:
    """
    基于本征正交分解(POD)的降维方法
    
    POD是一种用于提取数据中主要模态的技术，也被称为主成分分析(PCA)或Karhunen-Loève分解。
    它在流体力学和其他物理系统中广泛应用，用于识别空间相干结构。
    """
    
    def __init__(self, n_modes=10, random_seed=None):
        """
        初始化POD降维模型
        
        参数:
            n_modes: 要保留的POD模态数量
            random_seed: 随机种子
        """
        self.n_modes = n_modes
        self.random_seed = random_seed
        self.scaler = StandardScaler()
        self.feature_names = None
        self.training_time = None
        self.modes = None  # POD模态（特征向量）
        self.singular_values = None  # 奇异值
        self.explained_variance_ratio = None  # 解释方差比
        
        # 设置随机种子
        if random_seed is not None:
            np.random.seed(random_seed)
        
    def fit(self, X, feature_names=None):
        """
        训练POD模型
        
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
        
        # 记录训练开始时间
        start_time = time.time()
        
        # 计算协方差矩阵
        # 对于大型数据集，可以使用随机化SVD或增量SVD来提高效率
        cov_matrix = np.cov(X_scaled, rowvar=False)
        
        # 计算特征值和特征向量
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # 按特征值降序排序
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # 保存POD模态（特征向量）
        self.modes = eigenvectors[:, :self.n_modes]
        
        # 保存奇异值（特征值的平方根）
        self.singular_values = np.sqrt(eigenvalues[:self.n_modes])
        
        # 计算解释方差比
        total_variance = np.sum(eigenvalues)
        self.explained_variance_ratio = eigenvalues[:self.n_modes] / total_variance
        
        # 记录训练时间
        self.training_time = time.time() - start_time
        
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
        
        # 使用POD模态投影数据
        X_reduced = X_scaled @ self.modes
        
        return X_reduced
    
    def inverse_transform(self, X_reduced):
        """
        将低维数据转换回原始空间
        
        参数:
            X_reduced: 降维后的数据，形状为(样本数, n_modes)
            
        返回:
            X_reconstructed: 重构的数据，形状为(样本数, 特征数)
        """
        # 使用POD模态重构数据
        X_reconstructed = X_reduced @ self.modes.T
        
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
        
        # 计算每个特征在所有模态中的加权贡献
        # 使用奇异值作为权重
        feature_importance = np.abs(self.modes) @ np.diag(self.singular_values)
        feature_importance = np.mean(feature_importance, axis=1)
        
        return feature_importance
    
    def plot_modes(self, n_modes=None, figsize=(12, 8)):
        """
        绘制POD模态
        
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
            plt.plot(self.modes[:, i])
            plt.title(f'模态 {i+1}')
            if self.feature_names is not None and len(self.feature_names) == self.modes.shape[0]:
                plt.xticks(range(len(self.feature_names)), self.feature_names, rotation=90)
        
        plt.tight_layout()
        plt.show()
    
    def plot_explained_variance(self, figsize=(10, 6)):
        """
        绘制解释方差比
        
        参数:
            figsize: 图形大小
        """
        if self.explained_variance_ratio is None:
            print("模型尚未训练")
            return
        
        plt.figure(figsize=figsize)
        
        # 绘制单个解释方差比
        plt.bar(range(1, len(self.explained_variance_ratio) + 1), 
                self.explained_variance_ratio, 
                alpha=0.5, 
                label='单个解释方差比')
        
        # 绘制累积解释方差比
        cumulative = np.cumsum(self.explained_variance_ratio)
        plt.step(range(1, len(cumulative) + 1), cumulative, where='mid', label='累积解释方差比')
        
        plt.xlabel('主成分数量')
        plt.ylabel('解释方差比')
        plt.title('POD模态的解释方差比')
        plt.legend(loc='best')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def get_model_info(self):
        """
        获取模型信息
        
        返回:
            info: 包含模型信息的字典
        """
        info = {
            'method': 'POD',
            'n_modes': self.n_modes,
            'training_time': self.training_time,
            'explained_variance_ratio': self.explained_variance_ratio,
            'cumulative_explained_variance': np.sum(self.explained_variance_ratio) if self.explained_variance_ratio is not None else None
        }
        
        return info