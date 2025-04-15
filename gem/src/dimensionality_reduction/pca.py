import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA as SklearnPCA
from sklearn.preprocessing import StandardScaler
import time

class PCADimensionalityReduction:
    """
    基于主成分分析(PCA)的降维方法
    
    PCA是一种线性降维技术，通过正交变换将可能相关的变量转换为线性不相关的变量，
    这些新变量称为主成分。PCA可以找出数据中的主要变化方向，并保留数据的最大方差。
    """
    
    def __init__(self, n_components=2, random_seed=None):
        """
        初始化PCA降维模型
        
        参数:
            n_components: 主成分数量，即降维后的维度
            random_seed: 随机种子
        """
        self.n_components = n_components
        self.random_seed = random_seed
        self.pca = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.training_time = None
        self.explained_variance_ratio = None
        self.cumulative_explained_variance = None
        
    def fit(self, X, feature_names=None):
        """
        训练PCA模型
        
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
        
        # 初始化PCA
        self.pca = SklearnPCA(n_components=self.n_components, random_state=self.random_seed)
        
        # 记录训练开始时间
        start_time = time.time()
        
        # 训练PCA
        self.pca.fit(X_scaled)
        
        # 记录训练结束时间
        self.training_time = time.time() - start_time
        
        # 记录解释方差比
        self.explained_variance_ratio = self.pca.explained_variance_ratio_
        self.cumulative_explained_variance = np.cumsum(self.explained_variance_ratio)
        
        print(f"PCA训练完成，用时{self.training_time:.2f}秒")
        print(f"累积解释方差: {self.cumulative_explained_variance[-1]:.4f}")
        
        return self
    
    def transform(self, X):
        """
        将数据转换为低维表示
        
        参数:
            X: 输入数据，形状为(样本数, 特征数)
            
        返回:
            transformed_X: 转换后的数据，形状为(样本数, n_components)
        """
        # 数据预处理：标准化
        X_scaled = self.scaler.transform(X)
        
        # 使用PCA进行转换
        transformed_X = self.pca.transform(X_scaled)
        
        return transformed_X
    
    def fit_transform(self, X, feature_names=None):
        """
        训练模型并转换数据
        
        参数:
            X: 输入数据，形状为(样本数, 特征数)
            feature_names: 特征名称列表
            
        返回:
            transformed_X: 转换后的数据，形状为(样本数, n_components)
        """
        self.fit(X, feature_names)
        return self.transform(X)
    
    def inverse_transform(self, transformed_X):
        """
        将低维表示转换回原始特征空间
        
        参数:
            transformed_X: 低维表示，形状为(样本数, n_components)
            
        返回:
            X_reconstructed: 重构的原始特征空间数据，形状为(样本数, 特征数)
        """
        # 使用PCA进行逆变换
        X_reconstructed_scaled = self.pca.inverse_transform(transformed_X)
        
        # 反标准化
        X_reconstructed = self.scaler.inverse_transform(X_reconstructed_scaled)
        
        return X_reconstructed
    
    def plot_explained_variance(self, figsize=(10, 6)):
        """
        绘制解释方差比和累积解释方差
        
        参数:
            figsize: 图表大小
        """
        plt.figure(figsize=figsize)
        
        # 绘制解释方差比
        plt.bar(range(1, len(self.explained_variance_ratio) + 1), 
                self.explained_variance_ratio, alpha=0.7, label='解释方差比')
        
        # 绘制累积解释方差
        plt.step(range(1, len(self.cumulative_explained_variance) + 1), 
                 self.cumulative_explained_variance, where='mid', label='累积解释方差')
        
        plt.axhline(y=0.9, color='r', linestyle='--', label='90%解释方差')
        plt.axhline(y=0.95, color='g', linestyle='--', label='95%解释方差')
        
        plt.xlabel('主成分数量')
        plt.ylabel('解释方差比')
        plt.title('PCA解释方差分析')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def plot_components(self, figsize=(12, 10)):
        """
        绘制主成分的特征权重
        
        参数:
            figsize: 图表大小
        """
        if self.feature_names is None:
            feature_names = [f'Feature {i}' for i in range(self.pca.components_.shape[1])]
        else:
            feature_names = self.feature_names
        
        plt.figure(figsize=figsize)
        
        # 计算子图布局
        n_components = min(4, self.n_components)  # 最多显示前4个主成分
        n_cols = min(2, n_components)
        n_rows = (n_components + n_cols - 1) // n_cols
        
        for i in range(n_components):
            plt.subplot(n_rows, n_cols, i + 1)
            plt.bar(range(len(feature_names)), self.pca.components_[i])
            plt.xticks(range(len(feature_names)), feature_names, rotation=90)
            plt.title(f'主成分 {i+1} (解释方差: {self.explained_variance_ratio[i]:.2f})')
            plt.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def plot_transformed_data(self, X, labels=None, figsize=(10, 8)):
        """
        绘制转换后的数据
        
        参数:
            X: 输入数据
            labels: 样本标签
            figsize: 图表大小
        """
        if self.n_components < 2:
            print("降维后的维度小于2，无法进行二维可视化")
            return
        
        # 转换数据
        transformed_X = self.transform(X)
        
        plt.figure(figsize=figsize)
        
        # 绘制转换后的数据
        if labels is not None:
            # 如果有标签，按标签分类绘制
            for label in np.unique(labels):
                mask = labels == label
                plt.scatter(transformed_X[mask, 0], transformed_X[mask, 1], 
                           label=f'Class {label}', alpha=0.7)
            plt.legend()
        else:
            # 否则统一绘制
            plt.scatter(transformed_X[:, 0], transformed_X[:, 1], alpha=0.7)
        
        plt.title('PCA降维结果')
        plt.xlabel(f'主成分1 (解释方差: {self.explained_variance_ratio[0]:.2f})')
        plt.ylabel(f'主成分2 (解释方差: {self.explained_variance_ratio[1]:.2f})')
        plt.grid(True)
        plt.show()
    
    def get_model_info(self):
        """
        获取模型信息
        
        返回:
            info: 包含模型信息的字典
        """
        info = {
            'name': 'PCA',
            'n_components': self.n_components,
            'input_dim': self.pca.n_features_ if self.pca is not None else None,
            'training_time': self.training_time,
            'explained_variance_ratio': self.explained_variance_ratio.tolist() if self.explained_variance_ratio is not None else None,
            'cumulative_explained_variance': self.cumulative_explained_variance[-1] if self.cumulative_explained_variance is not None else None
        }
        
        return info


# 示例用法
def example():
    # 生成示例数据
    np.random.seed(42)
    X = np.random.randn(500, 10)  # 500个样本，10个特征
    
    # 初始化PCA模型
    pca = PCADimensionalityReduction(n_components=5)
    
    # 训练模型并转换数据
    transformed_X = pca.fit_transform(X)
    
    # 绘制解释方差
    pca.plot_explained_variance()
    
    # 绘制主成分
    pca.plot_components()
    
    # 绘制转换后的数据
    pca.plot_transformed_data(X)
    
    # 获取模型信息
    info = pca.get_model_info()
    print("模型信息:")
    for key, value in info.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    example()