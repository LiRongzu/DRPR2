"""
隐马尔可夫模型预测模块 (hmm.py)

该模块实现了基于隐马尔可夫模型(HMM)的时序预测功能，用于对降维后的河口盐度场数据进行预测。
HMM特别适合于处理具有明显状态转换特征的时序数据，能够捕捉数据中的隐含状态转移规律。
本模块使用hmmlearn库实现，支持高斯观测的HMM模型。

主要功能:
- 将输入特征与目标变量进行联合建模
- 通过Viterbi算法进行状态推断和预测
- 提供模型评估、保存和加载功能
- 支持随机种子设置，确保结果可重现性
"""

import numpy as np
from hmmlearn import hmm
import logging
import time

class HMMPredictionModel:
    """
    基于隐马尔可夫模型(HMM)的时序预测模型
    用于根据风场和径流的BMU预测盐度场的BMU
    """
    
    def __init__(self, model=None, n_states=5, n_iter=100, random_seed=None, input_dim=None, output_dim=None):
        """
        初始化HMM预测模型
        
        参数:
            model: 预训练的hmmlearn模型（可选）
            n_states: 隐状态数量
            n_iter: 最大迭代次数
            random_seed: 随机种子
            input_dim: 输入特征维度（风场+径流BMU的维度）
            output_dim: 输出特征维度（盐度场BMU的维度）
        """
        self.n_components = n_states
        self.n_iter = n_iter
        self.random_seed = random_seed
        self.model = model
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # 设置随机种子
        if random_seed is not None:
            np.random.seed(random_seed)
    
    def fit(self, X, y):
        """
        训练HMM模型
        
        参数:
            X: 输入特征（风场和径流BMU），形状为(n_samples, n_input_features)
            y: 目标特征（盐度场BMU），形状为(n_samples, n_output_features)
        """
        logging.info("准备HMM训练数据...")
        
        # 存储特征维度信息
        if self.input_dim is None:
            self.input_dim = X.shape[1]
        if self.output_dim is None:
            self.output_dim = y.shape[1]
        
        # 准备训练数据 - 合并输入和输出特征
        X_hmm = self._prepare_data(X, y)
        logging.info(f"HMM输入数据形状: {X_hmm.shape}")
        
        # 初始化并训练HMM模型
        if self.model is None:
            self.model = hmm.GaussianHMM(
                n_components=self.n_components,
                covariance_type="full",
                n_iter=self.n_iter,
                random_state=self.random_seed
            )
        
        # 训练模型
        logging.info("开始训练HMM模型...")
        start_time = time.time()
        self.model.fit(X_hmm)
        training_time = time.time() - start_time
        
        logging.info(f"HMM模型训练完成，耗时: {training_time:.2f}秒")
        if hasattr(self.model, 'monitor_'):
            logging.info(f"HMM模型收敛: {self.model.monitor_.converged}")
        return self
    
    def _prepare_data(self, X, y=None):
        """
        准备HMM输入数据
        
        参数:
            X: 输入特征（风场和径流BMU）
            y: 目标特征（盐度场BMU），可选
        """
        if y is not None:
            # 将输入特征和目标变量合并
            X_hmm = np.concatenate([X, y], axis=1)
            return X_hmm
        return X
    
    def predict(self, X):
        """
        使用训练好的模型进行预测
        
        参数:
            X: 输入特征（风场和径流BMU），形状为(n_samples, n_input_features)
            
        返回:
            y_pred: 预测的盐度场BMU，形状为(n_samples, n_output_features)
        """
        if self.model is None:
            logging.error("模型尚未训练，无法进行预测")
            return None
        
        logging.info("使用HMM模型进行预测...")
        start_time = time.time()
        
        # 对每个时间步进行预测
        y_pred = []
        
        # 使用Viterbi算法找出最可能的隐状态序列
        hidden_states = self.model.predict(X)
        
        # 从模型的均值中提取对应于输出维度的部分（盐度场BMU）
        means = self.model.means_
        for state in hidden_states:
            # 取均值的后output_dim列作为盐度场BMU预测
            output_means = means[state, -self.output_dim:]
            y_pred.append(output_means)
        
        y_pred = np.array(y_pred)
        prediction_time = time.time() - start_time
        
        logging.info(f"预测完成，耗时: {prediction_time:.2f}秒")
        return y_pred
        
    def evaluate(self, X, y):
        """
        评估模型性能
        
        参数:
            X: 输入特征（风场和径流BMU）
            y: 真实的盐度场BMU
            
        返回:
            metrics: 评估指标字典
        """
        if self.model is None:
            logging.error("模型尚未训练，无法进行评估")
            return None
        
        from evaluation.metrics import evaluate_prediction
        
        # 进行预测
        y_pred = self.predict(X)
        
        # 使用统一的评估指标函数
        metrics = evaluate_prediction(y, y_pred)
        
        logging.info(f"HMM模型评估结果: MSE={metrics['mse']:.6f}, MAE={metrics['mae']:.6f}")
        return metrics