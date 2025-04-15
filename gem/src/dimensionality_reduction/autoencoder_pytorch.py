import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import time
import optuna
import logging

class Encoder(nn.Module):
    """
    编码器网络
    """
    def __init__(self, input_dim, encoding_dim, hidden_layers, activation='relu', dropout_rate=0.2):
        super(Encoder, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # 添加编码器隐藏层
        for i, units in enumerate(hidden_layers):
            layers.append(nn.Linear(prev_dim, units))
            
            # 添加激活函数
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'sigmoid':
                layers.append(nn.Sigmoid())
            
            # 添加Dropout
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
                
            prev_dim = units
        
        # 添加编码层
        layers.append(nn.Linear(prev_dim, encoding_dim))
        if activation == 'relu':
            layers.append(nn.ReLU())
        elif activation == 'tanh':
            layers.append(nn.Tanh())
        elif activation == 'sigmoid':
            layers.append(nn.Sigmoid())
        
        self.encoder = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    """
    解码器网络
    """
    def __init__(self, encoding_dim, output_dim, hidden_layers, activation='relu', dropout_rate=0.2):
        super(Decoder, self).__init__()
        
        layers = []
        prev_dim = encoding_dim
        
        # 添加解码器隐藏层（与编码器对称）
        for i, units in enumerate(reversed(hidden_layers)):
            layers.append(nn.Linear(prev_dim, units))
            
            # 添加激活函数
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'sigmoid':
                layers.append(nn.Sigmoid())
            
            # 添加Dropout
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
                
            prev_dim = units
        
        # 添加输出层
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.decoder = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.decoder(x)


class Autoencoder(nn.Module):
    """
    自动编码器网络
    """
    def __init__(self, input_dim, encoding_dim, hidden_layers, activation='relu', dropout_rate=0.2):
        super(Autoencoder, self).__init__()
        
        self.encoder = Encoder(input_dim, encoding_dim, hidden_layers, activation, dropout_rate)
        self.decoder = Decoder(encoding_dim, input_dim, hidden_layers, activation, dropout_rate)
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def encode(self, x):
        return self.encoder(x)


class AutoencoderDimensionalityReduction:
    """
    基于自动编码器(Autoencoder)的降维方法 - PyTorch实现
    
    自动编码器是一种神经网络，通过学习将输入数据压缩到低维潜在空间，
    然后重构回原始空间，从而学习数据的有效表示。
    """
    
    def __init__(self, encoding_dim=2, hidden_layers=[128, 64, 32], 
                 activation='relu', dropout_rate=0.2, random_seed=None,
                 epochs=100, batch_size=32, learning_rate=0.001, verbose=1,
                 device=None, use_gpu=True, cuda_device=0):
        """
        初始化自动编码器降维模型
        
        参数:
            encoding_dim: 编码维度，即降维后的维度
            hidden_layers: 编码器和解码器中的隐藏层单元数列表
            activation: 激活函数 ('relu', 'tanh', 或 'sigmoid')
            dropout_rate: Dropout比率，用于防止过拟合
            random_seed: 随机种子
            epochs: 训练轮数
            batch_size: 批次大小
            learning_rate: 学习率
            verbose: 训练过程中的信息显示级别
            device: 计算设备 ('cpu' 或 'cuda')
            use_gpu: 是否使用GPU加速
            cuda_device: 使用的CUDA设备ID
        """
        self.encoding_dim = encoding_dim
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.random_seed = random_seed
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.verbose = verbose
        self.use_gpu = use_gpu
        self.cuda_device = cuda_device
        
        # 设置设备
        if device is not None:
            self.device = torch.device(device)
        else:
            if self.use_gpu and torch.cuda.is_available():
                self.device = torch.device(f'cuda:{self.cuda_device}')
                logging.info(f"使用GPU加速训练: {torch.cuda.get_device_name(self.cuda_device)}")
            else:
                self.device = torch.device('cpu')
                if self.use_gpu and not torch.cuda.is_available():
                    logging.warning("GPU不可用，将使用CPU进行训练")
                else:
                    logging.info("使用CPU进行训练")
        
        # 设置随机种子
        if random_seed is not None:
            torch.manual_seed(random_seed)
            np.random.seed(random_seed)
            if self.device.type == 'cuda':
                torch.cuda.manual_seed(random_seed)
                torch.cuda.manual_seed_all(random_seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
        
        self.model = None
        self.scaler = StandardScaler()
        self.input_dim = None
        self.feature_names = None
        self.training_time = None
        self.reconstruction_error = None
        self.history = {'train_loss': [], 'val_loss': []}
        self.optimizer = None
        self.best_params = None
    
    def _build_model(self, input_dim):
        """
        构建自动编码器模型
        
        参数:
            input_dim: 输入维度
            
        返回:
            model: 自动编码器模型
        """
        model = Autoencoder(
            input_dim=input_dim,
            encoding_dim=self.encoding_dim,
            hidden_layers=self.hidden_layers,
            activation=self.activation,
            dropout_rate=self.dropout_rate
        ).to(self.device)
        
        return model
    
    def _train_epoch(self, train_loader, val_loader=None):
        """
        训练一个epoch
        
        参数:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            
        返回:
            train_loss: 训练损失
            val_loss: 验证损失
        """
        # 训练模式
        self.model.train()
        train_loss = 0.0
        
        # 训练
        for batch_X in train_loader:
            # 获取输入数据
            if isinstance(batch_X, list):
                batch_X = batch_X[0]
            
            # 前向传播
            outputs = self.model(batch_X)
            loss = torch.nn.functional.mse_loss(outputs, batch_X)
            
            # 反向传播和优化
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            train_loss += loss.item() * batch_X.size(0)
        
        train_loss /= len(train_loader.dataset)
        
        # 验证
        val_loss = None
        if val_loader is not None:
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_X in val_loader:
                    # 获取输入数据
                    if isinstance(batch_X, list):
                        batch_X = batch_X[0]
                    
                    # 前向传播
                    outputs = self.model(batch_X)
                    loss = torch.nn.functional.mse_loss(outputs, batch_X)
                    
                    val_loss += loss.item() * batch_X.size(0)
                
                val_loss /= len(val_loader.dataset)
        
        return train_loss, val_loss
    
    def fit(self, X, y=None, validation_split=0.2, feature_names=None):
        """
        训练自动编码器模型
        
        参数:
            X: 输入特征，形状为(样本数, 特征数)
            y: 目标变量，对于自动编码器不需要，保留是为了与其他降维方法接口一致
            validation_split: 验证集比例
            feature_names: 特征名称列表
            
        返回:
            self: 训练好的模型实例
        """
        # 记录开始时间
        start_time = time.time()
        
        # 保存特征名称
        self.feature_names = feature_names
        
        # 标准化
        X_scaled = self.scaler.fit_transform(X)
        
        # 转换为PyTorch张量
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        # 创建数据集和数据加载器
        dataset = TensorDataset(X_tensor)
        
        # 划分训练集和验证集
        train_size = int((1 - validation_split) * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size)
        
        # 构建模型
        self.input_dim = X.shape[1]
        self.model = self._build_model(self.input_dim)
        
        # 定义优化器
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # 训练模型
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        for epoch in range(self.epochs):
            # 训练一个epoch
            train_loss, val_loss = self._train_epoch(train_loader, val_loader)
            
            # 记录损失
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            
            # 打印训练信息
            if self.verbose > 0 and (epoch + 1) % 10 == 0:
                logging.info(f"Epoch {epoch+1}/{self.epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            
            # 早停
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logging.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        # 计算重构误差
        self.model.eval()
        with torch.no_grad():
            X_reconstructed = self.model(X_tensor).cpu().numpy()
        
        self.reconstruction_error = np.mean((X_scaled - X_reconstructed) ** 2)
        
        # 记录训练时间
        self.training_time = time.time() - start_time
        
        logging.info(f"训练完成，耗时: {self.training_time:.2f}秒，重构误差: {self.reconstruction_error:.6f}")
        
        return self
    
    def transform(self, X):
        """
        将数据转换为低维表示
        
        参数:
            X: 输入特征，形状为(样本数, 特征数)
            
        返回:
            X_encoded: 低维表示，形状为(样本数, encoding_dim)
        """
        if self.model is None:
            raise ValueError("模型尚未训练，请先调用fit方法")
        
        # 标准化
        X_scaled = self.scaler.transform(X)
        
        # 转换为PyTorch张量并移动到指定设备
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        # 编码
        self.model.eval()
        with torch.no_grad():
            X_encoded = self.model.encode(X_tensor).cpu().numpy()
        
        return X_encoded
    
    def inverse_transform(self, X_encoded):
        """
        将低维表示转换回原始空间
        
        参数:
            X_encoded: 低维表示，形状为(样本数, encoding_dim)
            
        返回:
            X_reconstructed: 重构后的数据，形状为(样本数, 特征数)
        """
        if self.model is None:
            raise ValueError("模型尚未训练，请先调用fit方法")
        
        # 转换为PyTorch张量并移动到指定设备
        X_encoded_tensor = torch.FloatTensor(X_encoded).to(self.device)
        
        # 解码
        self.model.eval()
        with torch.no_grad():
            X_reconstructed_scaled = self.model.decoder(X_encoded_tensor).cpu().numpy()
        
        # 反标准化
        X_reconstructed = self.scaler.inverse_transform(X_reconstructed_scaled)
        
        return X_reconstructed
    
    def fit_transform(self, X, y=None, validation_split=0.2, feature_names=None):
        """
        训练模型并转换数据
        
        参数:
            X: 输入特征，形状为(样本数, 特征数)
            y: 目标变量，对于自动编码器不需要，保留是为了与其他降维方法接口一致
            validation_split: 验证集比例
            feature_names: 特征名称列表
            
        返回:
            X_encoded: 低维表示，形状为(样本数, encoding_dim)
        """
        self.fit(X, y, validation_split, feature_names)
        return self.transform(X)
    
    def score(self, X):
        """
        计算重构得分（负重构误差）
        
        参数:
            X: 输入特征，形状为(样本数, 特征数)
            
        返回:
            score: 重构得分，越高越好
        """
        if self.model is None:
            raise ValueError("模型尚未训练，请先调用fit方法")
        
        # 标准化
        X_scaled = self.scaler.transform(X)
        
        # 转换为PyTorch张量并移动到指定设备
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        # 重构
        self.model.eval()
        with torch.no_grad():
            X_reconstructed = self.model(X_tensor).cpu().numpy()
        
        # 计算重构误差
        reconstruction_error = np.mean((X_scaled - X_reconstructed) ** 2)
        
        # 返回负重构误差作为得分（越高越好）
        return -reconstruction_error
    
    def save(self, path):
        """
        保存模型
        
        参数:
            path: 模型保存路径
        """
        if self.model is None:
            raise ValueError("模型尚未训练，无法保存")
        
        # 将模型移动到CPU
        model_cpu = Autoencoder(
            input_dim=self.input_dim,
            encoding_dim=self.encoding_dim,
            hidden_layers=self.hidden_layers,
            activation=self.activation,
            dropout_rate=self.dropout_rate
        )
        model_cpu.load_state_dict(self.model.state_dict())
        
        # 保存模型和相关参数
        save_dict = {
            'model_state_dict': model_cpu.state_dict(),
            'scaler': self.scaler,
            'input_dim': self.input_dim,
            'encoding_dim': self.encoding_dim,
            'hidden_layers': self.hidden_layers,
            'activation': self.activation,
            'dropout_rate': self.dropout_rate,
            'feature_names': self.feature_names,
            'reconstruction_error': self.reconstruction_error,
            'history': self.history
        }
        
        torch.save(save_dict, path)
        logging.info(f"模型已保存到: {path}")
    
    def load(self, path):
        """
        加载模型
        
        参数:
            path: 模型加载路径
            
        返回:
            self: 加载好的模型实例
        """
        # 加载模型和相关参数
        save_dict = torch.load(path, map_location=self.device)
        
        self.input_dim = save_dict['input_dim']
        self.encoding_dim = save_dict['encoding_dim']
        self.hidden_layers = save_dict['hidden_layers']
        self.activation = save_dict['activation']
        self.dropout_rate = save_dict['dropout_rate']
        self.scaler = save_dict['scaler']
        self.feature_names = save_dict['feature_names']
        self.reconstruction_error = save_dict['reconstruction_error']
        self.history = save_dict['history']
        
        # 构建模型
        self.model = self._build_model(self.input_dim)
        self.model.load_state_dict(save_dict['model_state_dict'])
        
        logging.info(f"模型已从{path}加载")
        return self
    
    def plot_loss_curve(self):
        """
        绘制损失曲线
        """
        if not self.history['train_loss']:
            logging.warning("模型尚未训练，无法绘制损失曲线")
            return
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.history['train_loss'], label='Training Loss')
        plt.plot(self.history['val_loss'], label='Validation Loss')
        plt.title('Autoencoder Loss Curve')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def optimize_hyperparameters(self, X, n_trials=20, timeout=600):
        """
        使用Optuna优化超参数
        
        参数:
            X: 输入特征，形状为(样本数, 特征数)
            n_trials: 优化试验次数
            timeout: 优化超时时间（秒）
            
        返回:
            best_params: 最佳超参数
        """
        def objective(trial):
            # 超参数空间
            encoding_dim = trial.suggest_int('encoding_dim', 2, 20)
            n_layers = trial.suggest_int('n_layers', 1, 3)
            hidden_layers = []
            for i in range(n_layers):
                units = trial.suggest_int(f'units_{i}', 16, 128)
                hidden_layers.append(units)
            dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.5)
            learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
            batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
            
            # 创建模型
            model = AutoencoderDimensionalityReduction(
                encoding_dim=encoding_dim,
                hidden_layers=hidden_layers,
                dropout_rate=dropout_rate,
                learning_rate=learning_rate,
                batch_size=batch_size,
                epochs=50,  # 减少训练轮数以加快优化
                verbose=0,
                device=self.device,
                use_gpu=self.use_gpu,
                cuda_device=self.cuda_device
            )
            
            # 训练模型
            model.fit(X, validation_split=0.2)
            
            # 返回验证损失
            return model.history['val_loss'][-1]
        
        # 创建优化器
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials, timeout=timeout)
        
        # 获取最佳超参数
        best_params = study.best_params
        best_value = study.best_value
        
        # 更新超参数
        self.encoding_dim = best_params['encoding_dim']
        hidden_layers = []
        for i in range(best_params['n_layers']):
            hidden_layers.append(best_params[f'units_{i}'])
        self.hidden_layers = hidden_layers
        self.dropout_rate = best_params['dropout_rate']
        self.learning_rate = best_params['learning_rate']
        self.batch_size = best_params['batch_size']
        
        self.best_params = best_params
        
        logging.info(f"超参数优化完成，最佳验证损失: {best_value:.6f}")
        logging.info(f"最佳超参数: {best_params}")
        
        return best_params