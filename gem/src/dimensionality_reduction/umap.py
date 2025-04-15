import numpy as np
import matplotlib.pyplot as plt
from umap import UMAP as UMAPImpl
from sklearn.preprocessing import StandardScaler
import time

class UMAPDimensionalityReduction:
    """
    基于统一流形近似和投影(UMAP)的降维方法
    
    UMAP是一种非线性降维技术，可以保留数据的局部和全局结构，
    特别适合于可视化高维数据。
    """
    
    def __init__(self, n_components=2, n_neighbors=15, min_dist=0.1, 
                 metric='euclidean', random_seed=None):
        """
        初始化UMAP降维模型
        
        参数:
            n_components: 降维后的维度
            n_neighbors: 局部邻域大小
            min_dist: 嵌入中点之间的最小距离
            metric: 距离度量方式
            random_seed: 随机种子
        """
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.metric = metric
        self.random_seed = random_seed
        self.umap = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.training_time = None
        
    def fit(self, X, feature_names=None):
        """
        训练UMAP模型
        
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
        
        # 初始化UMAP
        self.umap = UMAPImpl(
            n_components=self.n_components,
            n_neighbors=self.n_neighbors,
            min_dist=self.min_dist,
            metric=self.metric,
            random_state=self.random_seed
        )
        
        # 记录训练开始时间
        start_time = time.time()
        
        # 训练UMAP
        self.umap.fit(X_scaled)
        
        # 记录训练结束时间
        self.training_time = time.time() - start_time
        
        print(f"UMAP训练完成，用时{self.training_time:.2f}秒")
        
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
        
        # 使用UMAP进行转换
        transformed_X = self.umap.transform(X_scaled)
        
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
        return self.umap.embedding_
    
    def plot_embedding(self, X=None, labels=None, figsize=(10, 8), title='UMAP Embedding'):
        """
        绘制UMAP嵌入结果
        
        参数:
            X: 输入数据，如果为None则使用训练数据的嵌入
            labels: 样本标签
            figsize: 图表大小
            title: 图表标题
        """
        if self.n_components < 2:
            print("降维后的维度小于2，无法进行二维可视化")
            return
        
        # 获取嵌入结果
        if X is not None:
            embedding = self.transform(X)
        else:
            if not hasattr(self.umap, 'embedding_'):
                print("模型尚未训练，无法获取嵌入结果")
                return
            embedding = self.umap.embedding_
        
        plt.figure(figsize=figsize)
        
        # 绘制嵌入结果
        if labels is not None:
            # 如果有标签，按标签分类绘制
            for label in np.unique(labels):
                mask = labels == label
                plt.scatter(embedding[mask, 0], embedding[mask, 1], 
                           label=f'Class {label}', alpha=0.7)
            plt.legend()
        else:
            # 否则统一绘制
            plt.scatter(embedding[:, 0], embedding[:, 1], alpha=0.7)
        
        plt.title(title)
        plt.xlabel('UMAP1')
        plt.ylabel('UMAP2')
        plt.grid(True)
        plt.show()
    
    def get_model_info(self):
        """
        获取模型信息
        
        返回:
            info: 包含模型信息的字典
        """
        info = {
            'name': 'UMAP',
            'n_components': self.n_components,
            'n_neighbors': self.n_neighbors,
            'min_dist': self.min_dist,
            'metric': self.metric,
            'training_time': self.training_time
        }
        
        return info


# 示例用法
def example():
    # 生成示例数据
    np.random.seed(42)
    X = np.random.randn(500, 10)  # 500个样本，10个特征
    
    # 初始化UMAP模型
    umap = UMAPDimensionalityReduction(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        random_seed=42
    )
    
    # 训练模型并转换数据
    transformed_X = umap.fit_transform(X)
    
    # 绘制嵌入结果
    umap.plot_embedding(title='UMAP Embedding of Random Data')
    
    # 获取模型信息
    info = umap.get_model_info()
    print("模型信息:")
    for key, value in info.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    example()