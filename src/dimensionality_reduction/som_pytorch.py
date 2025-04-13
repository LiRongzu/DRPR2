"""
PyTorch实现的自组织映射(SOM) 
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import os
import time
import logging
from typing import Tuple, Optional, Callable, List, Union

class SOMTorch(nn.Module):
    def __init__(self, 
                 input_dim: int = None, 
                 map_size: Tuple[int, int] = (10, 10), 
                 sigma: float = 1.0, 
                 learning_rate: float = 0.5,
                 random_seed: Optional[int] = None,
                 n_iterations: int = 1000,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 feature_name: str = 'salinity'):
        """
        PyTorch实现的自组织映射
        
        Args:
            input_dim: 输入数据的维度，可以在fit时自动推断
            map_size: SOM网格的大小 (width, height)
            sigma: 初始邻域半径
            learning_rate: 初始学习率
            random_seed: 随机种子
            n_iterations: 训练迭代次数
            device: 使用的设备 ('cpu' 或 'cuda')
        """
        super(SOMTorch, self).__init__()
        
        self.input_dim = input_dim
        self.map_size = map_size
        self.sigma = sigma
        self.lr = learning_rate
        self.device = device
        self.n_iterations = n_iterations
        self.random_seed = random_seed
        self.feature_name = feature_name

        # 设置随机种子
        if random_seed is not None:
            torch.manual_seed(random_seed)
            np.random.seed(random_seed)
                
        # 初始权重在fit中进行
        self.weights = None
        self.grid_x = None
        self.grid_y = None
        
        # 用于跟踪训练统计信息
        self.qerrors = []
        self.training_errors = []  # Add this line to store training errors
        self.training_time = None
        self.reconstruction_error = None
        
    def _initialize_network(self):
        """初始化SOM网络结构"""
        if self.input_dim is None:
            raise ValueError("input_dim必须在初始化时提供或在fit前设置")
        
        # 初始化权重
        self.weights = nn.Parameter(
            torch.randn(self.map_size[0], self.map_size[1], self.input_dim, device=self.device),
            requires_grad=False  # SOM训练不使用梯度下降
        )
        
        # 预计算网格坐标
        self.grid_x, self.grid_y = torch.meshgrid(
            torch.arange(0, self.map_size[0], device=self.device),
            torch.arange(0, self.map_size[1], device=self.device),
            indexing='ij'
        )
        
    def _compute_bmu(self, x):
        """
        计算输入数据的BMU（最佳匹配单元）的内部方法
        
        参数:
            x: 输入数据张量，形状为 [n_samples, input_dim]
            
        返回:
            linear_indices: BMU的线性索引列表
            grid_indices: BMU的网格索引列表，每个元素为(x, y)坐标
        """
        # 确保输入是2D张量 [batch_size, input_dim]
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
            
        batch_size = x.shape[0]
        
        # 展开权重以方便计算 [width*height, input_dim]
        flat_weights = self.weights.reshape(-1, self.input_dim)
        
        # 为批处理中的每个样本找到BMU
        linear_indices = []
        grid_indices = []
        for i in range(batch_size):
            distances = torch.sum((flat_weights - x[i])**2, dim=1)
            bmu_idx = torch.argmin(distances).item()
            linear_indices.append(bmu_idx)
            
            # 计算网格坐标并确保在有效范围内
            x_coord = bmu_idx % self.map_size[0]
            y_coord = bmu_idx // self.map_size[0]
            x_coord = min(max(x_coord, 0), self.map_size[0] - 1)
            y_coord = min(max(y_coord, 0), self.map_size[1] - 1)
            grid_indices.append((x_coord, y_coord))
        
        return linear_indices, grid_indices

    def forward(self, x):
        """计算输入数据的BMU（最佳匹配单元）"""
        return self._compute_bmu(x)
        
    def get_bmu_indices(self, X_tensor):
        """
        获取每个样本的最佳匹配单元(BMU)的索引
        
        参数:
            X_tensor: 输入数据张量，形状为 [n_samples, input_dim]
            
        返回:
            bmu_indices: 线性索引列表，表示每个样本的BMU在展平地图中的位置
        """
        linear_indices, _ = self._compute_bmu(X_tensor)
        return linear_indices
    
    def fit(self, X, callback=None, verbose=True):
        """
        训练SOM模型
        
        Args:
            X: 输入数据，形状为 [n_samples, n_features]
            feature_name: 特征名称列表
            callback: 每次迭代后调用的回调函数
            verbose: 是否打印训练进度
            
        Returns:
            self: 训练好的模型
        """

        
        # 设置输入维度（如果未指定）
        if self.input_dim is None:
            self.input_dim = X.shape[1]
            
        # 初始化网络
        self._initialize_network()
        
        # 转换为PyTorch张量
        # X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device) # Original line causing warning
        # --- MODIFICATION START ---
        # Use torch.from_numpy for better efficiency and adhere to warning recommendation
        if not isinstance(X, np.ndarray):
            # If X is already a tensor, just ensure correct type and device
            X_tensor = X.to(dtype=torch.float32, device=self.device)
        else:
            # Convert NumPy array to tensor
            X_tensor = torch.from_numpy(X).float().to(self.device)
        # --- MODIFICATION END ---

        # 记录训练开始时间
        start_time = time.time()
        
        # 调用内部训练函数
        self._fit_tensor(X_tensor, self.n_iterations, callback=callback, verbose=verbose)
        
        # 记录训练结束时间
        self.training_time = time.time() - start_time
        
        # 计算重构误差
        self.reconstruction_error = self._calculate_reconstruction_error(X)
        
        if verbose:
            print(f"SOM训练完成，用时{self.training_time:.2f}秒，重构误差: {self.reconstruction_error:.6f}")
        
        return self
    
    def _fit_tensor(self,
                   data,
                   n_iterations: int = 1000,
                   batch_size: int = 32,
                   callback: Optional[Callable] = None,
                   verbose: bool = True):
        """
        内部训练函数，处理PyTorch张量

        Args:
            data: 训练数据张量，形状为 [n_samples, input_dim]
            n_iterations: 训练迭代次数
            batch_size: 每次迭代的批大小
            callback: 每次迭代后调用的回调函数
            verbose: 是否打印训练进度
        """
        n_samples = data.shape[0]

        # --- MODIFICATION START ---
        # Calculate decay_factor robustly before the loop
        if self.sigma <= 0:
             raise ValueError("SOM sigma 必须为正数")
        elif np.isclose(self.sigma, 1.0):
            decay_factor = float('inf') # sigma doesn't decay
            logging.debug("Sigma is close to 1.0, decay_factor set to infinity.")
        else:
             # Calculate log(sigma) first
             log_sigma = np.log(self.sigma)
             # Check if log_sigma is close to zero (handles sigma close to 1)
             if np.isclose(log_sigma, 0.0):
                 decay_factor = float('inf')
                 logging.debug("Log(sigma) is close to 0, decay_factor set to infinity.")
             else:
                 # Perform division only if log_sigma is not zero
                 decay_factor = n_iterations / log_sigma
                 if not np.isfinite(decay_factor): # Add a check for safety
                     logging.warning(f"Calculated decay_factor is not finite ({decay_factor}). Setting to infinity.")
                     decay_factor = float('inf')

        # Remove the misplaced block that calculated sigma_t using 'i' here
        # --- MODIFICATION END ---

        # 创建一个专用的随机数生成器，确保每次运行获得相同的批次
        if self.random_seed is not None:
            # 为批次选择创建一个独立的生成器，使其不受其他操作影响
            batch_generator = torch.Generator(device=self.device)
            batch_generator.manual_seed(self.random_seed)
        else:
            batch_generator = None

        # 训练循环
        for i in range(n_iterations):
            # 计算当前迭代的sigma和学习率
            # --- Correct calculation inside the loop ---
            if np.isinf(decay_factor):
                sigma_t = self.sigma # sigma 接近 1 或计算失败，不衰减
            else:
                # Use the pre-calculated decay_factor
                sigma_t = self.sigma * np.exp(-i / decay_factor)
            # --- End correct calculation ---

            lr_t = self.lr * np.exp(-i / n_iterations)

            # 随机选择一批样本 - 使用专用生成器以确保结果可重现
            if batch_generator is not None:
                batch_indices = torch.randint(0, n_samples, (batch_size,), generator=batch_generator)
            else:
                batch_indices = torch.randint(0, n_samples, (batch_size,))
            batch = data[batch_indices]

            # 找到这批样本的BMU
            _, bmu_grid_indices = self(batch)

            # 更新权重
            for j, (x, y) in enumerate(bmu_grid_indices):
                # 计算网格中所有点到BMU的距离
                grid_dist = torch.sqrt((self.grid_x - x)**2 + (self.grid_y - y)**2)

                # 计算邻域函数 (Use sigma_t calculated inside the loop)
                # Add a small epsilon to prevent division by zero if sigma_t becomes exactly zero
                sigma_t_sq = sigma_t**2
                neighborhood = torch.exp(-(grid_dist**2) / (2 * sigma_t_sq + 1e-9)) # Use sigma_t

                # 扩展形状以便广播
                neighborhood = neighborhood.unsqueeze(-1)

                # 更新权重
                delta = lr_t * neighborhood * (batch[j].unsqueeze(0).unsqueeze(0) - self.weights)
                self.weights.data += delta

            # 计算量化误差（每n次迭代）
            if i % 10 == 0 or i == n_iterations - 1:
                with torch.no_grad():
                    qerror = self._quantization_error(data)
                    self.qerrors.append(qerror)
                    if verbose and i % 100 == 0:
                        print(f"迭代 {i}/{n_iterations}, 量化误差: {qerror:.6f}")

            # 如果提供了回调函数，则调用它
            if callback is not None:
                callback(i, self, data)
    
    def transform(self, X):
        """
        将数据转换为SOM网格上的位置坐标
        
        Args:
            X: 输入数据，形状为 [n_samples, n_features]
            
        Returns:
            转换后的数据，形状为 [n_samples, 2]
        """
        # 检查模型是否已训练
        if self.weights is None:
            raise ValueError("模型尚未训练，请先调用fit方法")
        

        X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
        
        # 获取每个样本在SOM网格上的位置
        _, bmu_grid_indices = self(X_tensor)
        transformed_X = np.array(bmu_grid_indices)
        
        return transformed_X
    
    def fit_transform(self, X, feature_name=None, callback=None, verbose=True):
        """
        训练模型并转换数据
        
        Args:
            X: 输入数据，形状为 [n_samples, n_features]
            feature_name: 特征名称列表
            callback: 可选的回调函数，用于训练过程中的监控
            verbose: 是否显示训练进度
            
        Returns:
            转换后的数据，形状为 [n_samples, 2]
        """
        self.fit(X, feature_name, callback, verbose)
        return self.transform(X)
    
    def inverse_transform(self, transformed_X):
        """
        将SOM网格上的位置坐标转换回原始特征空间
        
        Args:
            transformed_X: SOM网格上的位置坐标，形状为 [n_samples, 2]
            
        Returns:
            重构的原始特征空间数据，形状为 [n_samples, n_features]
        """
        # 检查模型是否已训练
        if self.weights is None:
            raise ValueError("模型尚未训练，请先调用fit方法")
        
        # 获取每个位置对应的权重向量
        X_reconstructed = np.array([
            self.weights[int(x[0]), int(x[1])].cpu().numpy() for x in transformed_X
        ])
        
        return X_reconstructed
    
    def _calculate_reconstruction_error(self, X):
        """
        计算重构误差
        
        Args:
            X: 归一化后的输入数据，numpy数组或PyTorch张量
            
        Returns:
            error: 重构误差
        """
        try:
            # 尝试使用正确的项目导入路径
            from src.evaluation.metrics import calculate_mse
        except ImportError:
            # 如果失败，使用相对导入
            try:
                from ..evaluation.metrics import calculate_mse
            except ImportError:
                # 如果导入仍然失败，使用一个简单的MSE实现作为后备方案
                def calculate_mse(y_true, y_pred):
                    return np.mean((y_true - y_pred) ** 2)
        
        if isinstance(X, np.ndarray):
            X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
        else:
            X_tensor = X
            
        # 获取每个样本在SOM网格上的位置
        _, bmu_grid_indices = self(X_tensor)
        
        # 获取每个位置对应的权重向量
        reconstructed = torch.stack([
            self.weights[x, y] for x, y in bmu_grid_indices
        ]).cpu().numpy()
        
        # 计算均方误差
        error = calculate_mse(X_tensor.cpu().numpy(), reconstructed)
        
        return error
    
    def _quantization_error(self, data):
        """计算量化误差 - 样本与其BMU之间的平均距离"""
        if isinstance(data, np.ndarray):
            data = torch.tensor(data, dtype=torch.float32, device=self.device)
            
        n_samples = data.shape[0]
        error_sum = 0.0
        
        # 对于大型数据集，分批计算以节省内存
        batch_size = 1000
        for i in range(0, n_samples, batch_size):
            batch = data[i:i+batch_size]
            batch_size_actual = batch.shape[0]
            
            # 获取BMU索引
            _, bmu_grid_indices = self(batch)
            
            # 计算每个样本到其BMU的距离
            for j, (x, y) in enumerate(bmu_grid_indices):
                # 确保索引在有效范围内
                x = min(max(int(x), 0), self.map_size[0] - 1)
                y = min(max(int(y), 0), self.map_size[1] - 1)
                bmu_weights = self.weights[x, y]
                error = torch.sum((batch[j] - bmu_weights)**2).sqrt().item()
                error_sum += error
                
        return error_sum / n_samples
    
    def plot_u_matrix(self, title='SOM U-Matrix', figsize=(10, 8)):
        """
        Plot U-Matrix to show distance relationships on SOM grid
        
        Args:
            title: Plot title
            figsize: Figure size
        """
        if self.weights is None:
            raise ValueError("Model not trained")
            
        plt.figure(figsize=figsize)
        umatrix = self._calculate_umatrix().cpu().numpy()
        plt.pcolor(umatrix.T, cmap='bone_r')  # 转置以匹配坐标系
        plt.colorbar()
        plt.title(title)
        plt.show()
    
    def _calculate_umatrix(self):
        """计算U-matrix"""
        umatrix = torch.zeros((self.map_size[0], self.map_size[1]), device=self.device)
        
        # 对于每个节点，计算与相邻节点的平均距离
        for i in range(self.map_size[0]):
            for j in range(self.map_size[1]):
                # 获取相邻坐标（考虑边界）
                neighbors = []
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        if di == 0 and dj == 0:
                            continue  # 跳过自身
                        ni, nj = i + di, j + dj
                        if 0 <= ni < self.map_size[0] and 0 <= nj < self.map_size[1]:
                            neighbors.append((ni, nj))
                
                # 计算到所有相邻节点的平均距离
                if neighbors:
                    dists = []
                    for ni, nj in neighbors:
                        dist = torch.sum((self.weights[i, j] - self.weights[ni, nj])**2).sqrt()
                        dists.append(dist)
                    umatrix[i, j] = torch.mean(torch.stack(dists))
                    
        return umatrix
    
    def plot_component_planes(self, figsize=(15, 10)):
        """
        Plot distribution of each input feature on SOM grid
        
        Args:
            figsize: Figure size
        """
        if self.weights is None:
            raise ValueError("Model not trained")
            
        if self.feature_name is None:
            feature_name = [f'Feature {i}' for i in range(self.input_dim)]
        else:
            feature_name = self.feature_name
        
        # 计算子图布局
        n_features = min(16, self.input_dim)  # 限制最多显示16个组件
        n_cols = min(3, n_features)
        n_rows = (n_features + n_cols - 1) // n_cols
        
        plt.figure(figsize=figsize)
        
        # 获取权重数据
        all_weights = self.weights.cpu().numpy()
        
        for i, feature_name in enumerate(feature_name[:n_features]):
            plt.subplot(n_rows, n_cols, i + 1)
            plt.pcolor(all_weights[:, :, i].T, cmap='coolwarm')
            plt.colorbar()
            plt.title(feature_name)
        
        plt.tight_layout()
        plt.show()
    
    def plot_sample_distribution(self, X, labels=None, figsize=(10, 8)):
        """
        Plot sample distribution on SOM grid
        
        Args:
            X: Input data
            labels: Sample labels (length must match number of samples in X)
            figsize: Figure size
        """
        if self.weights is None:
            raise ValueError("Model not trained")
            
        # Data preprocessing
        X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
        
        # Get position of each sample on SOM grid
        _, bmu_grid_indices = self(X_tensor)
        positions = np.array(bmu_grid_indices)
        
        # Validate labels shape
        if labels is not None:
            if len(labels) != len(positions):
                raise ValueError(f"Number of labels ({len(labels)}) does not match number of samples ({len(positions)})")
        
        plt.figure(figsize=figsize)
        
        # Plot U-Matrix as background
        umatrix = self._calculate_umatrix().cpu().numpy()
        plt.pcolor(umatrix.T, cmap='bone_r', alpha=0.5)
        plt.colorbar()
        
        # Plot sample distribution
        if labels is not None:
            # If labels exist, plot by label category
            unique_labels = np.unique(labels)
            for label in unique_labels:
                mask = np.array(labels) == label  # Ensure mask is numpy array
                plt.scatter(positions[mask, 0] + 0.5, positions[mask, 1] + 0.5, 
                          label=f'Class {label}', alpha=0.7)
            plt.legend()
        else:
            # Otherwise plot all samples uniformly
            plt.scatter(positions[:, 0] + 0.5, positions[:, 1] + 0.5, alpha=0.7)
        
        plt.title('Sample Distribution on SOM Grid')
        plt.show()
    
    def get_model_info(self):
        """
        获取模型信息
        
        Returns:
            info: 包含模型信息的字典
        """
        if self.weights is None:
            raise ValueError("模型尚未训练")
            
        info = {
            'name': 'SOM-PyTorch',
            'map_size': self.map_size,
            'input_dim': self.input_dim,
            'output_dim': 2,  # SOM输出是二维网格上的位置
            'n_iterations': self.n_iterations,
            'training_time': self.training_time,
            'reconstruction_error': self.reconstruction_error,
            'device': self.device
        }
        
        return info
    
    def save(self, path):
        """保存模型到文件"""
        if self.weights is None:
            raise ValueError("模型尚未训练，无法保存")

        # 保存模型状态
        state_dict = {
            'weights': self.weights.cpu(), # 保存权重到 CPU
            'map_size': self.map_size,
            'input_dim': self.input_dim, # ********** 添加这一行 **********
            'sigma': self.sigma,
            'lr': self.lr,
            'n_iterations': self.n_iterations,
            'random_seed': self.random_seed,
            'qerrors': self.qerrors, # 确保 qerrors 是可序列化的 (例如 list of floats)
            'training_time': self.training_time,
            'reconstruction_error': self.reconstruction_error,
            'feature_name': self.feature_name, # 添加特征名称

        }


        try:
            torch.save(state_dict, path)
            logging.info(f"模型已保存到: {path}")
        except Exception as e:
            logging.error(f"保存模型失败: {e}")
            raise e
        
    @classmethod
    def load(cls, path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """从文件加载模型"""
        # weights_only=False 是为了加载包含任意对象的字典
        loaded_data = torch.load(path, map_location=device, weights_only=False) # 重命名变量以示区分

        # 检查加载的是否为字典
        if not isinstance(loaded_data, dict):
             raise TypeError(f"加载的文件 {path} 内容不是预期的字典格式, 而是 {type(loaded_data)}")

        # 创建模型实例 (使用字典中的元数据)
        try:
            model = cls(
                input_dim=loaded_data['input_dim'],
                map_size=loaded_data['map_size'],
                sigma=loaded_data['sigma'],
                learning_rate=loaded_data['lr'],
                n_iterations=loaded_data['n_iterations'],
                random_seed=loaded_data['random_seed'],
                device=device,
                feature_name=loaded_data['feature_name'] 
            )
        except KeyError as e:
             raise KeyError(f"加载的字典缺少用于实例化模型的键: {e}") from e

        # 手动加载权重
        weights_key = 'weights'
        if weights_key not in loaded_data:
            raise KeyError(f"加载的字典缺少权重键 '{weights_key}'")
        if isinstance(loaded_data[weights_key], torch.Tensor):
            model.weights = nn.Parameter(loaded_data[weights_key].to(device), requires_grad=False)
        else:
            raise TypeError(f"加载的 '{weights_key}' 类型不是 Tensor: {type(loaded_data[weights_key])}")

        # 预计算网格坐标
        model.grid_x, model.grid_y = torch.meshgrid(
            torch.arange(0, model.map_size[0], device=device),
            torch.arange(0, model.map_size[1], device=device),
            indexing='ij'
        )

        # 加载其他属性
        model.qerrors = loaded_data.get('qerrors', [])
        model.training_time = loaded_data.get('training_time', None)
        model.reconstruction_error = loaded_data.get('reconstruction_error', None)

        # *** 关键修复：加载并设置 feature_name ***
        # 优先使用 'feature_name' (单个字符串)，其次使用 'feature_names' (列表或字符串)
        feature_name_val = loaded_data.get('feature_name', loaded_data.get('feature_names', None))

        if isinstance(feature_name_val, list) and len(feature_name_val) > 0:
            model.feature_name = feature_name_val[0] # 取列表第一个
        elif isinstance(feature_name_val, str):
            model.feature_name = feature_name_val # 直接使用字符串
        else:
            model.feature_name = None # 其他情况设为 None

        model.eval() # 设置为评估模式
        logging.info(f"SOM 模型已通过类加载方法从 {path} 加载。特征名: {model.feature_name}")
        return model
    def compute_distance_vectors(self, data):
        """
        计算输入数据到所有SOM原型向量的距离向量
        
        参数:
            data: 输入数据，numpy数组或PyTorch张量，形状为 [n_samples, n_features]
            
        返回:
            distance_vectors: 距离向量，形状为 [n_samples, map_size[0]*map_size[1]]，
                          表示每个样本到所有SOM节点的距离
        """
        if isinstance(data, np.ndarray):
            data_tensor = torch.tensor(data, dtype=torch.float32, device=self.device)
        else:
            data_tensor = data
            
        n_samples = data_tensor.shape[0]
        n_nodes = self.map_size[0] * self.map_size[1]
        distance_vectors = np.zeros((n_samples, n_nodes), dtype=np.float32)
        
        # 批处理计算以提高效率
        batch_size = 1000  # 可以根据内存情况调整
        n_batches = int(np.ceil(n_samples / batch_size))
        
        # 展平权重以方便计算 [width*height, input_dim]
        flat_weights = self.weights.reshape(-1, self.input_dim)
        
        for b in range(n_batches):
            start_idx = b * batch_size
            end_idx = min((b + 1) * batch_size, n_samples)
            current_batch = data_tensor[start_idx:end_idx]
            
            # 计算当前批次到所有原型向量的距离
            for i in range(current_batch.shape[0]):
                distances = torch.sqrt(torch.sum((flat_weights - current_batch[i])**2, dim=1))
                distance_vectors[start_idx + i] = distances.cpu().numpy()
            
            if (b + 1) % 10 == 0 or (b + 1) == n_batches:
                logging.info(f"距离向量计算进度: {end_idx}/{n_samples} 个样本")
                
        return distance_vectors