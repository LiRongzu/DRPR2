import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
import time

class AutoencoderDimensionalityReduction:
    """
    基于自动编码器(Autoencoder)的降维方法
    
    自动编码器是一种神经网络，通过学习将输入数据压缩到低维潜在空间，
    然后重构回原始空间，从而学习数据的有效表示。
    """
    
    def __init__(self, encoding_dim=2, hidden_layers=[128, 64, 32], 
                 activation='relu', dropout_rate=0.2, random_seed=None,
                 epochs=100, batch_size=32, verbose=1):
        """
        初始化自动编码器降维模型
        
        参数:
            encoding_dim: 编码维度，即降维后的维度
            hidden_layers: 编码器和解码器中的隐藏层单元数列表
            activation: 激活函数
            dropout_rate: Dropout比率，用于防止过拟合
            random_seed: 随机种子
            epochs: 训练轮数
            batch_size: 批次大小
            verbose: 训练过程中的信息显示级别
        """
        self.encoding_dim = encoding_dim
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.random_seed = random_seed
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        
        # 设置随机种子
        if random_seed is not None:
            tf.random.set_seed(random_seed)
            np.random.seed(random_seed)
        
        self.autoencoder = None
        self.encoder = None
        self.decoder = None
        self.scaler = StandardScaler()
        self.input_dim = None
        self.feature_names = None
        self.training_time = None
        self.reconstruction_error = None
        self.history = None
        
    def _build_model(self, input_dim):
        """
        构建自动编码器模型
        
        参数:
            input_dim: 输入维度
            
        返回:
            autoencoder: 自动编码器模型
            encoder: 编码器部分
            decoder: 解码器部分
        """
        # 编码器部分
        input_layer = Input(shape=(input_dim,), name='input')
        x = input_layer
        
        # 添加编码器隐藏层
        for i, units in enumerate(self.hidden_layers):
            x = Dense(units, activation=self.activation, name=f'encoder_{i}')(x)
            if self.dropout_rate > 0:
                x = Dropout(self.dropout_rate, name=f'encoder_dropout_{i}')(x)
        
        # 编码层
        encoded = Dense(self.encoding_dim, activation=self.activation, name='encoded')(x)
        
        # 定义编码器模型
        encoder = Model(input_layer, encoded, name='encoder')
        
        # 解码器部分
        decoder_input = Input(shape=(self.encoding_dim,), name='decoder_input')
        x = decoder_input
        
        # 添加解码器隐藏层（与编码器对称）
        for i, units in enumerate(reversed(self.hidden_layers)):
            x = Dense(units, activation=self.activation, name=f'decoder_{i}')(x)
            if self.dropout_rate > 0:
                x = Dropout(self.dropout_rate, name=f'decoder_dropout_{i}')(x)
        
        # 输出层
        decoded = Dense(input_dim, activation='linear', name='decoded')(x)
        
        # 定义解码器模型
        decoder = Model(decoder_input, decoded, name='decoder')
        
        # 定义完整的自动编码器模型
        autoencoder_output = decoder(encoder(input_layer))
        autoencoder = Model(input_layer, autoencoder_output, name='autoencoder')
        
        # 编译模型
        autoencoder.compile(optimizer='adam', loss='mse')
        
        return autoencoder, encoder, decoder
    
    def fit(self, X, feature_names=None):
        """
        训练自动编码器模型
        
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
        
        # 获取输入维度
        self.input_dim = X_scaled.shape[1]
        
        # 构建模型
        self.autoencoder, self.encoder, self.decoder = self._build_model(self.input_dim)
        
        # 设置早停回调
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # 记录训练开始时间
        start_time = time.time()
        
        # 训练模型
        self.history = self.autoencoder.fit(
            X_scaled, X_scaled,
            epochs=self.epochs,
            batch_size=self.batch_size,
            shuffle=True,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=self.verbose
        )
        
        # 记录训练结束时间
        self.training_time = time.time() - start_time
        
        # 计算重构误差
        X_reconstructed = self.autoencoder.predict(X_scaled)
        self.reconstruction_error = np.mean(np.square(X_scaled - X_reconstructed))
        
        print(f"自动编码器训练完成，用时{self.training_time:.2f}秒，重构误差: {self.reconstruction_error:.4f}")
        
        return self
    
    def transform(self, X):
        """
        将数据转换为低维表示
        
        参数:
            X: 输入数据，形状为(样本数, 特征数)
            
        返回:
            transformed_X: 转换后的数据，形状为(样本数, encoding_dim)
        """
        # 数据预处理：标准化
        X_scaled = self.scaler.transform(X)
        
        # 使用编码器进行转换
        transformed_X = self.encoder.predict(X_scaled)
        
        return transformed_X
    
    def fit_transform(self, X, feature_names=None):
        """
        训练模型并转换数据
        
        参数:
            X: 输入数据，形状为(样本数, 特征数)
            feature_names: 特征名称列表
            
        返回:
            transformed_X: 转换后的数据，形状为(样本数, encoding_dim)
        """
        self.fit(X, feature_names)
        return self.transform(X)
    
    def inverse_transform(self, transformed_X):
        """
        将低维表示转换回原始特征空间
        
        参数:
            transformed_X: 低维表示，形状为(样本数, encoding_dim)
            
        返回:
            X_reconstructed: 重构的原始特征空间数据，形状为(样本数, 特征数)
        """
        # 使用解码器进行逆变换
        X_reconstructed_scaled = self.decoder.predict(transformed_X)
        
        # 反标准化
        X_reconstructed = self.scaler.inverse_transform(X_reconstructed_scaled)
        
        return X_reconstructed
    
    def plot_loss_history(self, figsize=(10, 6)):
        """
        绘制训练过程中的损失曲线
        
        参数:
            figsize: 图表大小
        """
        if self.history is None:
            print("模型尚未训练，无法绘制损失曲线")
            return
        
        plt.figure(figsize=figsize)
        plt.plot(self.history.history['loss'], label='Training Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.title('Autoencoder Loss History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def plot_original_vs_reconstructed(self, X, n_samples=5, figsize=(15, 6)):
        """
        绘制原始数据与重构数据的对比图
        
        参数:
            X: 输入数据
            n_samples: 要显示的样本数
            figsize: 图表大小
        """
        # 数据预处理
        X_scaled = self.scaler.transform(X)
        
        # 随机选择样本
        indices = np.random.choice(X.shape[0], min(n_samples, X.shape[0]), replace=False)
        X_sample = X_scaled[indices]
        
        # 重构样本
        X_reconstructed = self.autoencoder.predict(X_sample)
        
        # 绘制对比图
        plt.figure(figsize=figsize)
        
        for i in range(len(indices)):
            # 原始数据
            plt.subplot(2, n_samples, i + 1)
            plt.bar(range(X_sample.shape[1]), X_sample[i])
            plt.title(f'Original {indices[i]}')
            if i == 0:
                plt.ylabel('Standardized Value')
            
            # 重构数据
            plt.subplot(2, n_samples, n_samples + i + 1)
            plt.bar(range(X_reconstructed.shape[1]), X_reconstructed[i])
            plt.title(f'Reconstructed {indices[i]}')
            if i == 0:
                plt.ylabel('Standardized Value')
        
        plt.tight_layout()
        plt.show()
    
    def plot_latent_space(self, X, labels=None, figsize=(10, 8)):
        """
        绘制潜在空间分布
        
        参数:
            X: 输入数据
            labels: 样本标签
            figsize: 图表大小
        """
        if self.encoding_dim != 2:
            print(f"潜在空间维度为{self.encoding_dim}，无法直接可视化。请使用encoding_dim=2进行训练或使用降维方法进行可视化。")
            return
        
        # 转换数据到潜在空间
        latent_repr = self.transform(X)
        
        plt.figure(figsize=figsize)
        
        # 绘制潜在空间分布
        if labels is not None:
            # 如果有标签，按标签分类绘制
            for label in np.unique(labels):
                mask = labels == label
                plt.scatter(latent_repr[mask, 0], latent_repr[mask, 1], 
                           label=f'Class {label}', alpha=0.7)
            plt.legend()
        else:
            # 否则统一绘制
            plt.scatter(latent_repr[:, 0], latent_repr[:, 1], alpha=0.7)
        
        plt.title('Latent Space Representation')
        plt.xlabel('Latent Dimension 1')
        plt.ylabel('Latent Dimension 2')
        plt.grid(True)
        plt.show()
    
    def get_model_info(self):
        """
        获取模型信息
        
        返回:
            info: 包含模型信息的字典
        """
        info = {
            'name': 'Autoencoder',
            'input_dim': self.input_dim,
            'encoding_dim': self.encoding_dim,
            'hidden_layers': self.hidden_layers,
            'activation': self.activation,
            'dropout_rate': self.dropout_rate,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'training_time': self.training_time,
            'reconstruction_error': self.reconstruction_error
        }
        
        return info


# 示例用法
def example():
    # 生成示例数据
    np.random.seed(42)
    X = np.random.randn(500, 10)  # 500个样本，10个特征
    
    # 初始化自动编码器模型
    autoencoder = AutoencoderDimensionalityReduction(
        encoding_dim=2,
        hidden_layers=[32, 16, 8],
        epochs=50,
        verbose=1
    )
    
    # 训练模型并转换数据
    transformed_X = autoencoder.fit_transform(X)
    
    # 绘制损失曲线
    autoencoder.plot_loss_history()
    
    # 绘制原始数据与重构数据的对比
    autoencoder.plot_original_vs_reconstructed(X, n_samples=5)
    
    # 绘制潜在空间