"""
基于流形的重建模块

该模块实现了基于局部流形结构的数据重建方法。
主要包括Path 1和Path 2两种重建方法：
1. Path 1: 降维预测后重建到原始空间（先预测后重建）
2. Path 2: 先重建后预测（先重建到原始空间再预测）
"""

import os
import numpy as np
import logging
import time
import pickle
from scipy.optimize import minimize
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


class ManifoldReconstruction:
    """
    基于流形的数据重建类
    
    提供了通过局部PCA流形结构将低维表示重建回原始高维空间的方法。
    """
    
    def __init__(self, neighborhood_size=5, n_components=5, optimization_method='L-BFGS-B',
                 max_iterations=100, gradient_tolerance=1e-5, use_mean_constraint=True):
        """
        初始化流形重建器
        
        参数:
            neighborhood_size: 局部邻域大小，用于构建局部流形
            n_components: 局部PCA保留的主成分数量
            optimization_method: 优化方法，用于重建过程中的优化
            max_iterations: 最大迭代次数
            gradient_tolerance: 梯度容忍度，优化停止条件
            use_mean_constraint: 是否使用平均值约束确保重建结果与原始数据具有相同的平均值
        """
        self.neighborhood_size = neighborhood_size
        self.n_components = n_components
        self.optimization_method = optimization_method
        self.max_iterations = max_iterations
        self.gradient_tolerance = gradient_tolerance
        self.use_mean_constraint = use_mean_constraint
        
        self.X_train = None  # 训练数据
        self.mean_value = None  # 平均场值
        self.nn_model = None  # 最近邻模型
        self.local_pca_models = {}  # 局部PCA模型字典
    
    def fit(self, X_train, distance_vectors=None):
        """
        训练流形重建模型
        
        参数:
            X_train: 训练数据，形状为(n_samples, n_features)
            distance_vectors: 对应的距离向量（如果有），形状为(n_samples, n_distance_features)
        
        返回:
            self: 训练好的模型
        """
        logging.info("开始训练流形重建模型...")
        start_time = time.time()
        
        self.X_train = X_train
        
        # 计算全局平均值（用于平均值约束）
        if self.use_mean_constraint:
            self.mean_value = np.mean(X_train, axis=0)
            logging.info(f"计算了全局平均值，形状: {self.mean_value.shape}")
        
        # 如果提供了距离向量，使用它们作为邻域搜索的特征
        X_for_neighbors = distance_vectors if distance_vectors is not None else X_train
        
        # 构建最近邻模型
        logging.info(f"构建最近邻模型, 使用{X_for_neighbors.shape}形状的数据...")
        self.nn_model = NearestNeighbors(n_neighbors=self.neighborhood_size, algorithm='auto')
        self.nn_model.fit(X_for_neighbors)
        
        # 预计算每个样本的局部PCA模型
        logging.info(f"开始预计算局部PCA模型，局部邻域大小: {self.neighborhood_size}, 主成分数量: {self.n_components}")
        
        # 为了提高效率，只为部分关键样本构建局部PCA模型
        sample_indices = np.random.choice(len(X_train), min(1000, len(X_train)), replace=False)
        
        for idx in sample_indices:
            # 获取当前样本的邻域
            distances, indices = self.nn_model.kneighbors([X_for_neighbors[idx]])
            neighbors = self.X_train[indices[0]]
            
            # 对邻域数据进行PCA
            pca = PCA(n_components=min(self.n_components, neighbors.shape[0], neighbors.shape[1]))
            pca.fit(neighbors)
            
            # 存储PCA模型
            self.local_pca_models[idx] = pca
        
        training_time = time.time() - start_time
        logging.info(f"流形重建模型训练完成，用时{training_time:.2f}秒")
        return self
    
    def get_local_pca(self, query_vector, distance_vectors=None):
        """
        为给定的查询向量获取或构建局部PCA模型
        
        参数:
            query_vector: 查询向量，形状为(n_features,)或(n_distance_features,)
            distance_vectors: 训练数据的距离向量，如果提供
        
        返回:
            pca: 局部PCA模型
            neighborhood: 邻域样本
        """
        # 确定用于邻域搜索的特征
        X_for_neighbors = distance_vectors if distance_vectors is not None else self.X_train
        query_for_neighbors = query_vector
        
        # 找到最近邻
        distances, indices = self.nn_model.kneighbors([query_for_neighbors])
        nearest_idx = indices[0][0]  # 最近邻的索引
        
        # 检查是否已有预计算的PCA模型
        if nearest_idx in self.local_pca_models:
            return self.local_pca_models[nearest_idx], self.X_train[indices[0]]
        
        # 否则构建新的局部PCA模型
        neighbors = self.X_train[indices[0]]
        
        # 对邻域数据进行PCA
        pca = PCA(n_components=min(self.n_components, neighbors.shape[0], neighbors.shape[1]))
        pca.fit(neighbors)
        
        return pca, neighbors
    
    def _reconstruction_error(self, coefficients, pca, neighborhood_mean):
        """
        计算给定系数下的重建误差
        
        参数:
            coefficients: PCA系数
            pca: 局部PCA模型
            neighborhood_mean: 邻域均值
        
        返回:
            error: 重建误差
        """
        # 通过PCA系数重建数据
        reconstructed = pca.inverse_transform([coefficients])[0]
        
        # 如果使用平均值约束，添加到目标函数
        if self.use_mean_constraint and self.mean_value is not None:
            # 惩罚与全局均值的偏差
            mean_penalty = np.sum((reconstructed - self.mean_value) ** 2)
            return np.sum(coefficients ** 2) + 0.1 * mean_penalty
        else:
            # 简单的L2正则化
            return np.sum(coefficients ** 2)
    
    def reconstruct(self, low_dim_vector, distance_vector=None):
        """
        将低维向量重建为原始高维空间中的向量
        
        参数:
            low_dim_vector: 低维表示向量
            distance_vector: 对应的距离向量（如果有）
        
        返回:
            reconstructed: 重建后的高维向量
        """
        if self.X_train is None or self.nn_model is None:
            raise ValueError("模型尚未训练，请先调用fit方法")
        
        # 获取局部PCA模型
        pca, neighborhood = self.get_local_pca(
            low_dim_vector if distance_vector is None else distance_vector,
            distance_vectors=None if distance_vector is None else self.X_train
        )
        
        # 计算邻域均值
        neighborhood_mean = np.mean(neighborhood, axis=0)
        
        # 使用优化方法求解最佳系数
        initial_coefficients = np.zeros(pca.n_components_)
        
        result = minimize(
            self._reconstruction_error,
            initial_coefficients,
            args=(pca, neighborhood_mean),
            method=self.optimization_method,
            options={'maxiter': self.max_iterations, 'gtol': self.gradient_tolerance}
        )
        
        if not result.success:
            logging.warning(f"优化未成功收敛: {result.message}")
        
        # 使用最优系数重建数据
        reconstructed = pca.inverse_transform([result.x])[0]
        
        # 如果使用平均值约束，调整重建结果以匹配全局均值
        if self.use_mean_constraint and self.mean_value is not None:
            # 调整重建结果以保持全局平均值
            current_mean = np.mean(reconstructed)
            target_mean = np.mean(self.mean_value)
            reconstructed = reconstructed + (target_mean - current_mean)
        
        return reconstructed
    
    def save(self, filepath):
        """
        保存模型到文件
        
        参数:
            filepath: 保存路径
        """
        model_data = {
            'neighborhood_size': self.neighborhood_size,
            'n_components': self.n_components,
            'optimization_method': self.optimization_method,
            'max_iterations': self.max_iterations,
            'gradient_tolerance': self.gradient_tolerance,
            'use_mean_constraint': self.use_mean_constraint,
            'mean_value': self.mean_value,
            'local_pca_models': self.local_pca_models
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logging.info(f"流形重建模型已保存到: {filepath}")
    
    def load(self, filepath):
        """
        从文件加载模型
        
        参数:
            filepath: 模型文件路径
        
        返回:
            self: 加载了模型的实例
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.neighborhood_size = model_data['neighborhood_size']
        self.n_components = model_data['n_components']
        self.optimization_method = model_data['optimization_method']
        self.max_iterations = model_data['max_iterations']
        self.gradient_tolerance = model_data['gradient_tolerance']
        self.use_mean_constraint = model_data['use_mean_constraint']
        self.mean_value = model_data['mean_value']
        self.local_pca_models = model_data['local_pca_models']
        
        logging.info(f"流形重建模型已从{filepath}加载")
        return self


def create_manifold_reconstruction_model(config, X_train, distance_vectors=None, model_save_path=None):
    """
    创建并训练流形重建模型
    
    参数:
        config: 配置对象
        X_train: 训练数据，形状为(n_samples, n_features)
        distance_vectors: 对应的距离向量（如果有），形状为(n_samples, n_distance_features)
        model_save_path: 模型保存路径
    
    返回:
        model: 训练好的流形重建模型
    """
    logging.info("创建流形重建模型...")
    
    # 创建模型
    model = ManifoldReconstruction(
        neighborhood_size=config.path1_reconstruction.NEIGHBORHOOD_SIZE,
        n_components=config.path1_reconstruction.LOCAL_PCA_COMPONENTS,
        optimization_method=config.path1_reconstruction.OPTIMIZATION_METHOD,
        max_iterations=config.path1_reconstruction.MAX_ITERATIONS,
        gradient_tolerance=config.path1_reconstruction.GRADIENT_TOLERANCE,
        use_mean_constraint=config.path1_reconstruction.USE_MEAN_CONSTRAINT
    )
    
    # 训练模型
    model.fit(X_train, distance_vectors)
    
    # 保存模型
    if model_save_path is not None:
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        model.save(model_save_path)
    
    return model


def reconstruct_from_manifold(low_dim_data, reference_data, n_neighbors=5, n_components=5,
                            optimization_method='L-BFGS-B', max_iterations=100,
                            gradient_tolerance=1e-5, use_mean_constraint=True):
    """
    将低维数据重建回原始空间
    
    参数:
        low_dim_data: 需要重建的低维数据，形状为(n_samples, n_features_low)
        reference_data: 参考数据（原始空间），形状为(n_samples_ref, n_features_high)
        n_neighbors: 局部邻域大小
        n_components: 局部PCA保留的主成分数量
        optimization_method: 优化方法
        max_iterations: 最大迭代次数
        gradient_tolerance: 梯度容忍度
        use_mean_constraint: 是否使用平均值约束
    
    返回:
        reconstructed_data: 重建后的高维数据，形状为(n_samples, n_features_high)
    """
    # 创建重建模型
    model = ManifoldReconstruction(
        neighborhood_size=n_neighbors,
        n_components=n_components,
        optimization_method=optimization_method,
        max_iterations=max_iterations,
        gradient_tolerance=gradient_tolerance,
        use_mean_constraint=use_mean_constraint
    )
    
    # 训练模型
    model.fit(reference_data)
    
    # 重建每个样本
    reconstructed_data = np.array([
        model.reconstruct(low_dim_vector)
        for low_dim_vector in low_dim_data
    ])
    
    return reconstructed_data