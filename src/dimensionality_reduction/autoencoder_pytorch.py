# src/dimensionality_reduction/autoencoder_pytorch.py

import numpy as np
# import matplotlib.pyplot as plt # 如果绘图方法被移除，则不需要
import torch
import torch.nn as nn
import torch.optim as optim
# from torch.utils.data import DataLoader, TensorDataset # 不在此文件中直接使用
# from sklearn.preprocessing import StandardScaler # 不再需要内部 Scaler
# import time # 如果不使用
# import optuna # 如果不使用超参数优化方法
import logging
# 为了与旧代码兼容，暂时保留 BaseEstimator, TransformerMixin，但注意 fit/transform 签名已改变
from sklearn.base import BaseEstimator, TransformerMixin

log = logging.getLogger(__name__) # 使用标准日志

class Encoder(nn.Module):
    """
    编码器网络 (保持不变)
    """
    def __init__(self, input_dim, encoding_dim, hidden_layers, activation='ReLU', dropout_rate=0.2): # 将 activation 改为大写 ReLU
        super(Encoder, self).__init__()
        layers = []
        prev_dim = input_dim

        # 获取激活函数实例
        if activation == 'ReLU':
            act_fn = nn.ReLU()
        elif activation == 'Tanh':
            act_fn = nn.Tanh()
        elif activation == 'Sigmoid':
            act_fn = nn.Sigmoid()
        else:
            # 尝试从 torch.nn 获取，如果失败则报错
            try:
                 act_fn = getattr(nn, activation)()
            except AttributeError:
                 log.error(f"无效的激活函数名称: {activation}")
                 raise ValueError(f"无效的激活函数名称: {activation}")

        # 添加编码器隐藏层
        if hidden_layers: # 确保 hidden_layers 不是 None 或空列表
             for units in hidden_layers:
                 layers.append(nn.Linear(prev_dim, units))
                 layers.append(act_fn) # 添加激活函数实例
                 if dropout_rate > 0:
                     layers.append(nn.Dropout(dropout_rate))
                 prev_dim = units

        # 添加编码层
        layers.append(nn.Linear(prev_dim, encoding_dim))
        # 通常编码层的输出不加 Dropout，但可以加激活函数
        layers.append(act_fn) # 在编码层后也添加激活函数

        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    """
    解码器网络 (保持不变)
    """
    def __init__(self, encoding_dim, output_dim, hidden_layers, activation='ReLU', dropout_rate=0.2): # 将 activation 改为大写 ReLU
        super(Decoder, self).__init__()
        layers = []
        prev_dim = encoding_dim

        # 获取激活函数实例
        if activation == 'ReLU':
            act_fn = nn.ReLU()
        elif activation == 'Tanh':
            act_fn = nn.Tanh()
        elif activation == 'Sigmoid':
            act_fn = nn.Sigmoid()
        else:
            try:
                 act_fn = getattr(nn, activation)()
            except AttributeError:
                 log.error(f"无效的激活函数名称: {activation}")
                 raise ValueError(f"无效的激活函数名称: {activation}")

        # 添加解码器隐藏层（与编码器对称）
        if hidden_layers: # 确保 hidden_layers 不是 None 或空列表
             for units in reversed(hidden_layers):
                 layers.append(nn.Linear(prev_dim, units))
                 layers.append(act_fn) # 添加激活函数实例
                 if dropout_rate > 0:
                     layers.append(nn.Dropout(dropout_rate))
                 prev_dim = units

        # 添加输出层
        layers.append(nn.Linear(prev_dim, output_dim))
        # 输出层通常不加激活函数（除非是特定情况如 Sigmoid 用于 [0,1] 输出）
        # 或许需要根据数据特性决定是否添加，这里先不加

        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.decoder(x)


class Autoencoder(nn.Module):
    """
    自动编码器网络 (基本保持不变)
    """
    def __init__(self, input_dim, encoding_dim, hidden_layers, activation='ReLU', dropout_rate=0.2):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(input_dim, encoding_dim, hidden_layers, activation, dropout_rate)
        # !!! 确保 Decoder 的 output_dim 是 input_dim !!!
        self.decoder = Decoder(encoding_dim, input_dim, hidden_layers, activation, dropout_rate)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def encode(self, x): # 保留 encode 方法
        return self.encoder(x)

    def decode(self, z): # 添加显式的 decode 方法
         return self.decoder(z)


# === 修改后的主类 ===
class AutoencoderDimensionalityReduction(BaseEstimator, TransformerMixin): # 保持继承关系
    """
    基于自动编码器的降维方法 - PyTorch 实现。
    此类现在假设输入数据在外部已标准化。
    fit 方法接收 PyTorch DataLoader。
    encode/decode 方法处理 NumPy 数组。
    """

    def __init__(self, input_dim, encoding_dim=64, hidden_layers=[128, 64], # 示例默认值
                 activation='ReLU', dropout_rate=0.2, learning_rate=0.001, weight_decay=0.0,
                 device=None, random_seed=None):
        """
        初始化自动编码器降维模型。

        参数:
            input_dim: 输入特征维度 (必需)
            encoding_dim: 编码维度 (降维后的维度)
            hidden_layers: 编码器隐藏层单元数列表 (解码器将对称使用)
            activation: 隐藏层和编码层的激活函数 ('ReLU', 'Tanh', 'Sigmoid' 或 torch.nn 中的类名)
            dropout_rate: Dropout 比率
            learning_rate: 优化器学习率
            weight_decay: 优化器权重衰减 (L2 正则化)
            device: 计算设备 ('cpu', 'cuda', 'cuda:0' 等)
            random_seed: 随机种子，用于可复现性
        """
        self.input_dim = input_dim # *必需*
        self.encoding_dim = encoding_dim
        # 确保 hidden_layers 是列表或 None
        self.hidden_layers = list(hidden_layers) if hidden_layers is not None else []
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate # 用于 fit 方法中的优化器
        self.weight_decay = weight_decay   # 用于 fit 方法中的优化器
        self.random_seed = random_seed

        # --- 设备设置 ---
        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log.info(f"AE 模型将使用设备: {self.device}")

        # --- 设置随机种子 ---
        if random_seed is not None:
            torch.manual_seed(random_seed)
            np.random.seed(random_seed)
            if self.device.type == 'cuda':
                torch.cuda.manual_seed_all(random_seed) # 对于多 GPU
                # 以下两项可能影响性能，但提高复现性
                # torch.backends.cudnn.deterministic = True
                # torch.backends.cudnn.benchmark = False
            log.info(f"设置随机种子为: {random_seed}")

        # --- 创建模型 ---
        # !! 在 __init__ 中立即创建模型 !!
        self.model = self._build_model(self.input_dim)
        log.info("Autoencoder 模型已构建。")
        # print("DEBUG: self.model in __init__:", self.model) # 调试后可删除

        # --- 其他状态变量 ---
        # self.scaler = None # 不再需要内部 scaler
        self.is_fitted_ = False
        self.history = {'train_loss': [], 'val_loss': []} # 保留训练历史记录

    def _build_model(self, input_dim):
        """
        构建自动编码器模型。 (基本保持不变)
        """
        model = Autoencoder(
            input_dim=input_dim,
            encoding_dim=self.encoding_dim,
            hidden_layers=self.hidden_layers,
            activation=self.activation,
            dropout_rate=self.dropout_rate
        ).to(self.device) # 直接移动到设备
        return model

    # 不再需要 _train_epoch，fit 方法包含循环

    def fit(self, train_loader, val_loader=None, epochs=100): # 添加 epochs 参数
        """
        使用 PyTorch DataLoader 训练模型。
        假设 DataLoader 中的数据已标准化。
        """
        if self.model is None:
             log.error("模型在 __init__ 中未能创建。无法训练。")
             raise ValueError("模型未被初始化。")

        self.model.to(self.device) # 再次确保模型在正确的设备上

        criterion = nn.MSELoss()
        # 在 fit 方法中创建优化器
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        log.info(f"开始训练 Autoencoder，共 {epochs} 个 epochs...")
        self.history = {'train_loss': [], 'val_loss': []} # 重置历史记录

        for epoch in range(epochs):
            # --- 训练循环 ---
            self.model.train() # 设置为训练模式
            epoch_train_loss = 0.0
            for batch_data in train_loader:
                # DataLoader 可能返回一个包含特征的列表/元组
                if isinstance(batch_data, (list, tuple)):
                    batch_features = batch_data[0].to(self.device) # 获取第一个元素并移动到设备
                else:
                    batch_features = batch_data.to(self.device) # 假设直接是张量

                optimizer.zero_grad()
                outputs = self.model(batch_features)
                loss = criterion(outputs, batch_features) # 重建损失
                loss.backward()
                optimizer.step()
                epoch_train_loss += loss.item() # * batch_features.size(0) ? 如果损失是平均值则不需要乘

            avg_train_loss = epoch_train_loss / len(train_loader) # 计算平均损失
            self.history['train_loss'].append(avg_train_loss)

            # --- 验证循环 ---
            epoch_val_loss = None
            if val_loader:
                self.model.eval() # 设置为评估模式
                val_loss_sum = 0.0
                with torch.no_grad():
                    for batch_data in val_loader:
                        if isinstance(batch_data, (list, tuple)):
                            batch_features = batch_data[0].to(self.device)
                        else:
                            batch_features = batch_data.to(self.device)
                        outputs = self.model(batch_features)
                        loss = criterion(outputs, batch_features)
                        val_loss_sum += loss.item() # * batch_features.size(0) ?
                epoch_val_loss = val_loss_sum / len(val_loader) # 计算平均损失
                self.history['val_loss'].append(epoch_val_loss)
                log.info(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.6f}, Val Loss: {epoch_val_loss:.6f}")
            else:
                self.history['val_loss'].append(None) # 如果没有验证集，记录 None
                log.info(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.6f}")

            # (可选: 在此添加 Early Stopping 或 Model Checkpoint 逻辑)

        log.info("Autoencoder 训练完成。")
        self.is_fitted_ = True
        return self

    def encode(self, X): # 替换 transform
        """
        将输入数据编码为低维潜空间表示。
        假设输入 X 是 NumPy 数组，并且已在外部标准化。
        """
        if not self.is_fitted_:
            raise ValueError("模型尚未训练 (fit)，无法进行编码。")
        if self.model is None:
            raise ValueError("模型未初始化。")

        self.model.eval() # 设置为评估模式
        X_encoded_list = []
        # 处理可能的大型 NumPy 数组，分批进行编码
        batch_size = 1024 # 或者根据显存调整
        num_samples = X.shape[0]

        with torch.no_grad():
            for i in range(0, num_samples, batch_size):
                batch_X = X[i:min(i + batch_size, num_samples)]
                X_tensor = torch.tensor(batch_X, dtype=torch.float32, device=self.device)
                encoded_batch = self.model.encode(X_tensor)
                X_encoded_list.append(encoded_batch.cpu().numpy())

        X_encoded = np.concatenate(X_encoded_list, axis=0)
        return X_encoded

    def decode(self, Z): # 替换 inverse_transform
        """
        将低维潜空间表示解码回原始（标准化）空间。
        输入 Z 是 NumPy 数组。
        输出是 NumPy 数组 (仍在标准化尺度)。
        """
        if not self.is_fitted_:
             raise ValueError("模型尚未训练 (fit)，无法进行解码。")
        if self.model is None:
            raise ValueError("模型未初始化。")

        self.model.eval() # 设置为评估模式
        X_decoded_list = []
        batch_size = 1024 # 可调整
        num_samples = Z.shape[0]

        with torch.no_grad():
             for i in range(0, num_samples, batch_size):
                 batch_Z = Z[i:min(i + batch_size, num_samples)]
                 Z_tensor = torch.tensor(batch_Z, dtype=torch.float32, device=self.device)
                 decoded_batch = self.model.decode(Z_tensor) # 使用显式 decode
                 X_decoded_list.append(decoded_batch.cpu().numpy())

        X_decoded = np.concatenate(X_decoded_list, axis=0)
        return X_decoded # 返回的是标准化尺度的数据

    # 移除 fit_transform 方法
    # def fit_transform(...): pass

    # 移除 score 方法
    # def score(...): pass

    def save(self, path):
        """
        保存模型状态和必要的配置参数。不保存 Scaler。
        """
        if not self.is_fitted_:
            log.warning("模型尚未训练，但仍将保存当前状态和配置。")
        if self.model is None:
            raise ValueError("模型未初始化，无法保存。")

        # 保存必需的配置以便能够重建模型
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'config': { # 保存用于重建模型的参数
                'input_dim': self.input_dim,
                'encoding_dim': self.encoding_dim,
                'hidden_layers': self.hidden_layers,
                'activation': self.activation,
                'dropout_rate': self.dropout_rate,
                # 注意：不再保存 scaler
            },
            'history': self.history # 可以选择保存训练历史
        }
        try:
            torch.save(save_dict, path)
            log.info(f"模型状态和配置已保存到: {path}")
        except Exception as e:
            log.error(f"保存模型到 {path} 时出错: {e}", exc_info=True)


    @classmethod # 使用类方法以便在加载时创建新实例
    def load(cls, path, device=None): # device 参数用于指定加载到的设备
        """
        加载模型状态和配置参数。不加载 Scaler。
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log.info(f"尝试从 {path} 加载模型到设备 {device}...")

        try:
            save_dict = torch.load(path, map_location=device) # 加载到指定设备

            config = save_dict['config']
            # 使用加载的配置创建类的实例
            # 注意：__init__ 现在需要 input_dim 等参数
            instance = cls(
                input_dim=config['input_dim'],
                encoding_dim=config['encoding_dim'],
                hidden_layers=config['hidden_layers'],
                activation=config['activation'],
                dropout_rate=config['dropout_rate'],
                device=device # 传递设备参数
                # 初始化时不再需要 learning_rate, weight_decay 等训练参数
            )

            # 加载模型状态字典
            instance.model.load_state_dict(save_dict['model_state_dict'])
            instance.model.to(device) # 确保模型在目标设备
            instance.is_fitted_ = True # 标记为已加载/训练
            instance.history = save_dict.get('history', {'train_loss': [], 'val_loss': []}) # 加载历史记录

            log.info(f"模型已成功从 {path} 加载。")
            return instance
        except FileNotFoundError:
             log.error(f"模型文件未找到: {path}")
             raise
        except Exception as e:
             log.error(f"从 {path} 加载模型时出错: {e}", exc_info=True)
             raise

    # 移除 plot_loss_curve 方法，绘图逻辑移到 pipeline 或 notebook 中
    # def plot_loss_curve(...): pass

    # 移除 optimize_hyperparameters 方法，超参数优化是独立的任务
    # def optimize_hyperparameters(...): pass