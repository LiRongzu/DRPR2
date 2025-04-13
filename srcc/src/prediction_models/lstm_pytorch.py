import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import logging
import time

class LSTMModel(nn.Module):
    """
    PyTorch LSTM模型定义
    """
    def __init__(self, input_size, hidden_size, output_size, num_layers=2, dropout=0.2):
        super(LSTMModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Dropout层
        self.dropout = nn.Dropout(dropout)
        
        # 全连接输出层
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # 初始化隐藏状态和细胞状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # LSTM前向传播
        out, _ = self.lstm(x, (h0, c0))
        
        # 只取最后一个时间步的输出
        out = self.dropout(out[:, -1, :])
        
        # 全连接层
        out = self.fc(out)
        
        return out


class LSTMPredictionModel:
    """
    基于LSTM的时序预测模型 - PyTorch实现
    
    用于预测降维后的河口盐度场数据
    """
    
    def __init__(self, units=50, dropout=0.2, epochs=100, batch_size=32, sequence_length=10, patience=10, 
                 random_seed=None, learning_rate=0.001, device=None, use_gpu=True, cuda_device=0):
        """
        初始化LSTM预测模型
        
        参数:
            units: LSTM隐藏单元数量
            dropout: Dropout比例，用于防止过拟合
            epochs: 训练轮数
            batch_size: 批次大小
            sequence_length: 输入序列长度
            random_seed: 随机种子
            learning_rate: 学习率
            device: 计算设备 ('cpu' 或 'cuda')
            use_gpu: 是否使用GPU加速
            cuda_device: 使用的CUDA设备ID
        """
        self.units = units
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.patience = patience  # 早停耐心值
        self.random_seed = random_seed
        self.learning_rate = learning_rate
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
        self.optimizer = None
        self.criterion = None
        self.history = {'train_loss': [], 'val_loss': []}
    
    
    def _build_model(self, input_size, output_size):
        """
        构建LSTM模型
        
        参数:
            input_size: 输入特征维度
            output_size: 输出维度
        """
        model = LSTMModel(
            input_size=input_size,
            hidden_size=self.units,
            output_size=output_size,
            num_layers=2,
            dropout=self.dropout
        ).to(self.device)
        
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()
        
        return model, optimizer, criterion
    
    def fit(self, X_seq, y_seq):
        """
        训练LSTM模型
        
        参数:
            X: 输入特征，形状为(样本数, 特征数)
            y: 目标变量，形状为(样本数, 目标维度)
            
        返回:
            self: 训练好的模型实例
        """

        logging.info(f"序列数据形状: X={X_seq.shape}, y={y_seq.shape}")
        
        # 构建模型
        input_size = X_seq.shape[2]  # 特征维度
        output_size = y_seq.shape[1]  # 输出维度
        self.model, self.optimizer, self.criterion = self._build_model(input_size, output_size)
        
        # 创建数据加载器
        X_tensor = torch.FloatTensor(X_seq).to(self.device)
        y_tensor = torch.FloatTensor(y_seq).to(self.device)
        dataset = TensorDataset(X_tensor, y_tensor)
        
        # 划分训练集和验证集
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size)
        
        # 训练模型
        logging.info("开始训练LSTM模型...")
        start_time = time.time()
        
        best_val_loss = float('inf')

        patience_counter = 0
        
        for epoch in range(self.epochs):
            # 训练阶段
            self.model.train()
            train_loss = 0.0
            for batch_X, batch_y in train_loader:
                # 前向传播
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                
                # 反向传播和优化
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item() * batch_X.size(0)
            
            train_loss /= train_size
            
            # 验证阶段
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    outputs = self.model(batch_X)
                    loss = self.criterion(outputs, batch_y)
                    val_loss += loss.item() * batch_X.size(0)
            
            val_loss /= val_size
            
            # 记录损失
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            
            # 打印进度
            if (epoch + 1) % 10 == 0 or epoch == 0:
                logging.info(f"Epoch {epoch+1}/{self.epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            
            # 早停
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # 保存最佳模型状态
                best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    logging.info(f"早停: {patience}个epoch内验证损失没有改善")
                    # 恢复最佳模型状态
                    self.model.load_state_dict(best_model_state)
                    break
        
        training_time = time.time() - start_time
        logging.info(f"LSTM模型训练完成，耗时: {training_time:.2f}秒")
        return self
    
    def predict(self, X):
        """
        使用训练好的模型进行预测
        
        参数:
            X: 输入特征，形状为(样本数, 特征数)
            
        返回:
            y_pred: 预测结果，形状为(样本数-序列长度, 目标维度)
        """
        if self.model is None:
            logging.error("模型尚未训练，无法进行预测")
            return None
        
        # 创建序列
        X_seq = []
        for i in range(len(X) - self.sequence_length):
            X_seq.append(X[i:i + self.sequence_length])
        X_seq = np.array(X_seq)
        
        # 转换为PyTorch张量并移动到指定设备
        X_tensor = torch.FloatTensor(X_seq).to(self.device)
        
        # 进行预测
        logging.info("使用LSTM模型进行预测...")
        start_time = time.time()
        
        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(X_tensor).cpu().numpy()
        
        prediction_time = time.time() - start_time
        logging.info(f"预测完成，耗时: {prediction_time:.2f}秒")
        
        return y_pred
    
    def evaluate(self, X_seq, y_seq):
        """
        评估模型性能
        
        参数:
            X_seq: 输入特征，形状为(样本数, 序列长度, 特征数)
            y_seq: 目标变量，形状为(样本数, 目标维度)
            
        返回:
            metrics: 评估指标字典
        """
        if self.model is None:
            logging.error("模型尚未训练，无法进行评估")
            return None
            
        from evaluation.metrics import evaluate_prediction
        
        # 转换为PyTorch张量并移动到指定设备
        X_tensor = torch.FloatTensor(X_seq).to(self.device)
        y_tensor = torch.FloatTensor(y_seq).to(self.device)
        
        # 评估模型
        logging.info("评估LSTM模型性能...")
        
        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(X_tensor)
            # 计算损失
            loss = self.criterion(y_pred, y_tensor).item()
            
        # 使用统一的评估函数计算指标
        metrics = evaluate_prediction(y_seq, y_pred.cpu().numpy())
        metrics['loss'] = loss
        
        logging.info(f"LSTM模型评估结果: 损失={loss:.6f}, MSE={metrics['mse']:.6f}, MAE={metrics['mae']:.6f}")
        return metrics
    
    def save(self, model_path):
        """
        保存模型
        
        参数:
            model_path: 模型保存路径
        """
        if self.model is None:
            logging.error("模型尚未训练，无法保存")
            return
        
        # 将模型移动到CPU以便保存
        model_cpu = self.model.to('cpu')
        
        # 保存模型状态和配置
        state = {
            'model_state_dict': model_cpu.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'units': self.units,
            'dropout': self.dropout,
            'sequence_length': self.sequence_length,
            'history': self.history,
            'input_size': self.model.lstm.input_size,
            'output_size': self.model.fc.out_features,
            'num_layers': self.model.lstm.num_layers
        }
        
        torch.save(state, model_path)
        logging.info(f"LSTM模型已保存到: {model_path}")
        
        # 将模型移回原设备
        self.model.to(self.device)
    
    def load(self, model_path):
        """
        加载模型
        
        参数:
            model_path: 模型加载路径
            
        返回:
            self: 加载了模型的实例
        """
        # 检查设备兼容性
        if not torch.cuda.is_available() and self.device.type == 'cuda':
            logging.warning("GPU不可用，将模型加载到CPU上")
            state = torch.load(model_path, map_location='cpu')
        else:
            state = torch.load(model_path, map_location=self.device)
        
        # 获取模型参数
        self.units = state['units']
        self.dropout = state['dropout']
        self.sequence_length = state['sequence_length']
        self.history = state['history']
        input_size = state['input_size']
        output_size = state['output_size']
        num_layers = state.get('num_layers', 2)  # 兼容旧版本保存的模型
        
        # 构建模型
        self.model = LSTMModel(
            input_size=input_size,
            hidden_size=self.units,
            output_size=output_size,
            num_layers=num_layers,
            dropout=self.dropout
        ).to(self.device)
        
        # 加载模型参数
        self.model.load_state_dict(state['model_state_dict'])
        
        # 创建优化器
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        if 'optimizer_state_dict' in state:
            self.optimizer.load_state_dict(state['optimizer_state_dict'])
        
        # 定义损失函数
        self.criterion = nn.MSELoss()
        
        logging.info(f"LSTM模型已从{model_path}加载")
        return self