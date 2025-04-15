# src/prediction_models/lstm_pytorch.py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import logging
import time
from typing import List, Dict, Optional, Tuple # Added List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

class LSTMModel(nn.Module):
    """
    PyTorch LSTM 模型定义 (适用于 BMU 索引预测)
    """
    def __init__(self,
                 num_embeddings_list: List[int], # 每个输入特征的 SOM 节点数 (用于 Embedding)
                 embedding_dims_list: List[int], # 每个输入特征的 Embedding 维度
                 hidden_size: int,
                 target_output_classes: int,     # 目标 SOM 的节点数 (用于最终输出)
                 num_layers: int = 2,
                 dropout: float = 0.2):
        """
        初始化 LSTM 模型

        Args:
            num_embeddings_list: List containing the number of nodes for each input SOM feature.
                                 The order must match the order of features in the input tensor.
            embedding_dims_list: List containing the desired embedding dimension for each input feature.
            hidden_size: LSTM 隐藏层大小.
            target_output_classes: Number of classes for the target BMU prediction (nodes in target SOM).
            num_layers: LSTM 层数.
            dropout: Dropout 比例.
        """
        super(LSTMModel, self).__init__()

        if not (len(num_embeddings_list) == len(embedding_dims_list)):
             raise ValueError("num_embeddings_list 和 embedding_dims_list 必须有相同的长度")

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_input_features = len(num_embeddings_list)

        # --- Embedding 层 ---
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=num_nodes, embedding_dim=emb_dim)
            for num_nodes, emb_dim in zip(num_embeddings_list, embedding_dims_list)
        ])

        # --- 计算 LSTM 的实际输入大小 (所有 embedding 维度之和) ---
        lstm_input_size = sum(embedding_dims_list)

        # --- LSTM 层 ---
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True, # 输入: (batch, seq_len, feature)
            dropout=dropout if num_layers > 1 else 0
        )

        # --- Dropout 层 ---
        self.dropout_layer = nn.Dropout(dropout) # 使用 dropout_layer 避免与参数名冲突

        # --- 全连接输出层 (预测目标 BMU 类别) ---
        self.fc = nn.Linear(hidden_size, target_output_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入张量，形状为 (batch, seq_len, num_input_features)，包含整数 BMU 索引。

        Returns:
            输出张量，形状为 (batch, target_output_classes)，包含预测的目标 BMU 类别的 logits。
        """
        if x.dtype != torch.long:
            # logger.warning("Input tensor is not LongTensor, casting to long for embedding lookup.")
            x = x.long() # Embedding 层需要 LongTensor 索引

        batch_size, seq_len, _ = x.shape

        # --- 应用 Embedding ---
        embedded_features = []
        for i in range(self.num_input_features):
            # 输入索引形状: (batch, seq_len)
            feature_indices = x[:, :, i]
            # Embedding 输出形状: (batch, seq_len, embedding_dims_list[i])
            embedded = self.embeddings[i](feature_indices)
            embedded_features.append(embedded)

        # --- 连接嵌入特征 ---
        # 输出形状: (batch, seq_len, lstm_input_size)
        embedded_x = torch.cat(embedded_features, dim=-1)

        # --- 初始化 LSTM 状态 ---
        # 形状: (num_layers, batch_size, hidden_size)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)

        # --- LSTM 前向传播 ---
        # out 形状: (batch, seq_len, hidden_size)
        lstm_out, _ = self.lstm(embedded_x, (h0, c0))

        # --- 只取最后一个时间步的输出 ---
        # last_time_step_out 形状: (batch, hidden_size)
        last_time_step_out = lstm_out[:, -1, :]

        # --- 应用 Dropout ---
        dropped_out = self.dropout_layer(last_time_step_out)

        # --- 全连接层 ---
        # out 形状: (batch, target_output_classes) - Logits
        out = self.fc(dropped_out)

        return out


class LSTMPredictionModel:
    """
    包装器类，用于训练、预测和评估 LSTM 模型 (BMU 索引版本)
    """

    def __init__(self,
                 # --- 模型架构参数 ---
                 num_embeddings_list: List[int], # 来自 input_feature_info
                 embedding_dims_list: List[int], # 来自 input_feature_info 或配置
                 hidden_size: int,               # 来自配置 model.prediction.lstm.hidden_size
                 target_som_num_nodes: int,      # 目标 SOM 节点数
                 num_layers: int = 2,            # 来自配置 model.prediction.lstm.num_layers
                 dropout: float = 0.2,           # 来自配置 model.prediction.lstm.dropout
                 # --- 训练参数 ---
                 epochs: int = 100,              # 来自配置 training.epochs
                 batch_size: int = 32,           # 来自配置 model.prediction.lstm.batch_size
                 learning_rate: float = 0.001,   # 来自配置 training.optimizer.learning_rate
                 patience: int = 10,             # 来自配置 training.early_stopping.patience
                 # --- 其他 ---
                 sequence_length: int = 10,      # 来自配置 model.prediction.lstm.sequence_length (可能冗余)
                 random_seed: Optional[int] = None, # 来自配置 training.random_seed
                 device: Optional[str] = None,   # 来自 get_device_from_config
                 ):
        """
        初始化 LSTM 预测模型包装器
        """
        self.num_embeddings_list = num_embeddings_list
        self.embedding_dims_list = embedding_dims_list
        self.hidden_size = hidden_size
        self.target_som_num_nodes = target_som_num_nodes
        self.num_layers = num_layers
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.patience = patience
        self.sequence_length = sequence_length # 保留以供参考，但序列化在外部完成
        self.random_seed = random_seed

        # --- 设备设置 ---
        if device is not None:
            self.device = torch.device(device)
        else:
            # 默认尝试使用 GPU
            if torch.cuda.is_available():
                 self.device = torch.device('cuda')
                 try: logger.info(f"使用 GPU: {torch.cuda.get_device_name(0)}")
                 except: logger.info("使用 GPU") # Fallback if name lookup fails
            else:
                 self.device = torch.device('cpu')
                 logger.info("使用 CPU")

        # --- 随机种子 ---
        if random_seed is not None:
             torch.manual_seed(random_seed)
             np.random.seed(random_seed)
             if self.device.type == 'cuda':
                  torch.cuda.manual_seed(random_seed)
                  torch.cuda.manual_seed_all(random_seed) # if multi-GPU
                  # 可选: 为了完全可复现性，但可能影响性能
                  # torch.backends.cudnn.deterministic = True
                  # torch.backends.cudnn.benchmark = False

        self.model: Optional[LSTMModel] = None
        self.optimizer: Optional[optim.Optimizer] = None
        self.criterion: Optional[nn.Module] = None
        self.history: Dict[str, List[float]] = {'train_loss': [], 'val_loss': []}

    def _build_model(self):
        """构建 LSTM 模型、优化器和损失函数"""
        model = LSTMModel(
            num_embeddings_list=self.num_embeddings_list,
            embedding_dims_list=self.embedding_dims_list,
            hidden_size=self.hidden_size,
            target_output_classes=self.target_som_num_nodes,
            num_layers=self.num_layers,
            dropout=self.dropout
        ).to(self.device)

        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss() # 使用交叉熵损失

        return model, optimizer, criterion

    def fit(self,
            X_train: np.ndarray, y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None):
        """
        训练 LSTM 模型

        Args:
            X_train: 训练输入序列, 形状 (num_samples, seq_len, num_input_features) - 包含 BMU 索引
            y_train: 训练目标序列, 形状 (num_samples,) - 包含目标 BMU 索引
            X_val: 验证输入序列 (可选)
            y_val: 验证目标序列 (可选)
        """
        if self.model is None:
             self.model, self.optimizer, self.criterion = self._build_model()

        # --- 创建数据加载器 ---
        # 输入 X 需要是 LongTensor (用于 embedding), 目标 y 需要是 LongTensor (用于 CrossEntropyLoss)
        X_train_tensor = torch.LongTensor(X_train).to(self.device)
        y_train_tensor = torch.LongTensor(y_train).to(self.device)
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        logger.info(f"训练数据加载器创建: {len(train_dataset)} 个样本")

        val_loader = None
        if X_val is not None and y_val is not None:
            X_val_tensor = torch.LongTensor(X_val).to(self.device)
            y_val_tensor = torch.LongTensor(y_val).to(self.device)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
            logger.info(f"验证数据加载器创建: {len(val_dataset)} 个样本")
        else:
            logger.info("没有提供验证数据，早停将基于训练损失（不推荐）或不使用。")
            # 如果没有验证集，早停逻辑需要调整或禁用


        # --- 训练循环 ---
        logger.info(f"开始 LSTM 训练，共 {self.epochs} epochs...")
        start_time = time.time()
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None

        for epoch in range(self.epochs):
            self.model.train()
            epoch_train_loss = 0.0
            for batch_X, batch_y in train_loader:
                # batch_X: (batch, seq_len, num_features)
                # batch_y: (batch,) - 目标 BMU 索引

                outputs = self.model(batch_X) # Logits: (batch, target_classes)
                loss = self.criterion(outputs, batch_y)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_train_loss += loss.item() * batch_X.size(0)

            avg_train_loss = epoch_train_loss / len(train_dataset)
            self.history['train_loss'].append(avg_train_loss)

            # --- 验证 ---
            avg_val_loss = float('nan') # 默认 NaN
            if val_loader:
                self.model.eval()
                epoch_val_loss = 0.0
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        outputs = self.model(batch_X)
                        loss = self.criterion(outputs, batch_y)
                        epoch_val_loss += loss.item() * batch_X.size(0)
                avg_val_loss = epoch_val_loss / len(val_dataset)
                self.history['val_loss'].append(avg_val_loss)

                log_msg = (f"Epoch {epoch+1}/{self.epochs}, "
                           f"Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")

                # --- 早停逻辑 (基于验证损失) ---
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    best_model_state = self.model.state_dict() # 保存最佳模型状态
                    log_msg += " (New best)"
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        logger.info(f"早停触发: 在 {self.patience} 个 epochs 内验证损失没有改善。")
                        if best_model_state:
                             self.model.load_state_dict(best_model_state) # 恢复最佳模型
                             logger.info("已恢复到最佳模型状态。")
                        break # 结束训练
            else:
                # 如果没有验证集，只记录训练损失
                log_msg = (f"Epoch {epoch+1}/{self.epochs}, Train Loss: {avg_train_loss:.6f}")
                # 可以在这里添加基于训练损失的早停，但不推荐
                best_val_loss = avg_train_loss # 简单处理，以便后续保存

            # --- 打印日志 ---
            if (epoch + 1) % 10 == 0 or epoch == 0 or (val_loader and patience_counter == 0): # 每10轮,第一轮,或有提升时打印
                 logger.info(log_msg)


        # --- 训练结束 ---
        # 如果从未触发早停或没有验证集，确保保存最后/最佳模型
        if not val_loader or patience_counter < self.patience:
            if val_loader and best_model_state:
                 self.model.load_state_dict(best_model_state) # 确保用的是最佳模型
            elif not val_loader:
                 logger.warning("没有验证集，将保存最后一个 epoch 的模型。")
                 best_model_state = self.model.state_dict() # 保存最后状态
                 best_val_loss = avg_train_loss # 记录最后训练损失


        training_time = time.time() - start_time
        logger.info(f"LSTM 训练完成，耗时: {training_time:.2f} 秒。最佳验证损失: {best_val_loss:.6f}")
        return self


    def predict(self, X_seq: np.ndarray) -> np.ndarray:
        """
        使用训练好的模型进行预测 (BMU 索引)

        Args:
            X_seq: 输入序列, 形状 (num_samples, seq_len, num_input_features) - BMU 索引

        Returns:
            y_pred_indices: 预测的目标 BMU 索引, 形状 (num_samples,)
        """
        if self.model is None:
            logging.error("模型尚未训练或加载，无法进行预测")
            return np.array([]) # 返回空数组

        logger.info(f"使用 LSTM 模型进行预测，输入形状: {X_seq.shape}")
        start_time = time.time()

        # 转换为 Tensor
        X_tensor = torch.LongTensor(X_seq).to(self.device)

        self.model.eval()
        all_preds = []
        with torch.no_grad():
             # 使用 DataLoader 进行批量预测，避免内存问题
             pred_dataset = TensorDataset(X_tensor)
             pred_loader = DataLoader(pred_dataset, batch_size=self.batch_size * 2, shuffle=False) # 可以用稍大 batch

             for batch_X_tuple in pred_loader:
                  batch_X = batch_X_tuple[0] # DataLoader 返回元组
                  logits = self.model(batch_X)       # Logits: (batch, target_classes)
                  preds = torch.argmax(logits, dim=1) # 预测索引: (batch,)
                  all_preds.append(preds.cpu())

        # --- 拼接预测结果 ---
        y_pred_indices = torch.cat(all_preds).numpy()

        prediction_time = time.time() - start_time
        logger.info(f"预测完成，输出形状: {y_pred_indices.shape}，耗时: {prediction_time:.2f}秒")

        return y_pred_indices


    def evaluate(self, X_seq: np.ndarray, y_seq: np.ndarray) -> Dict[str, float]:
        """
        评估模型性能 (主要关注损失)

        Args:
            X_seq: 输入序列, 形状 (num_samples, seq_len, num_input_features)
            y_seq: 真实目标 BMU 索引, 形状 (num_samples,)

        Returns:
            metrics: 包含损失和其他指标的字典 (注意：MSE/MAE 对索引意义不大)
        """
        if self.model is None or self.criterion is None:
             logging.error("模型或损失函数未初始化，无法评估。")
             return {'loss': float('nan')}

        logger.info(f"评估 LSTM 模型性能...")
        X_tensor = torch.LongTensor(X_seq).to(self.device)
        y_tensor = torch.LongTensor(y_seq).to(self.device)

        self.model.eval()
        total_loss = 0.0
        correct_preds = 0
        total_samples = 0

        with torch.no_grad():
             eval_dataset = TensorDataset(X_tensor, y_tensor)
             eval_loader = DataLoader(eval_dataset, batch_size=self.batch_size * 2, shuffle=False)

             for batch_X, batch_y in eval_loader:
                  outputs = self.model(batch_X)       # Logits
                  loss = self.criterion(outputs, batch_y)
                  total_loss += loss.item() * batch_X.size(0)

                  preds = torch.argmax(outputs, dim=1) # Predicted indices
                  correct_preds += (preds == batch_y).sum().item()
                  total_samples += batch_y.size(0)

        avg_loss = total_loss / total_samples
        accuracy = correct_preds / total_samples if total_samples > 0 else 0.0

        metrics = {'loss': avg_loss, 'accuracy': accuracy}
        logger.info(f"LSTM 模型评估结果: Loss={avg_loss:.6f}, Accuracy={accuracy:.4f}")
        return metrics


    def save(self, model_path: str):
        """保存模型状态和配置"""
        if self.model is None:
             logging.error("模型尚未训练，无法保存")
             return

        model_cpu = self.model.to('cpu') # 移动到 CPU 保存

        state = {
            # --- 模型状态 ---
            'model_state_dict': model_cpu.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            # --- 架构参数 ---
            'num_embeddings_list': self.num_embeddings_list,
            'embedding_dims_list': self.embedding_dims_list,
            'hidden_size': self.hidden_size,
            'target_som_num_nodes': self.target_som_num_nodes,
            'num_layers': self.num_layers,
            'dropout': self.dropout,
            # --- 训练参数 (参考) ---
            'epochs_run': len(self.history['train_loss']), # 实际运行的 epochs
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'sequence_length': self.sequence_length,
            # --- 历史记录 ---
            'history': self.history,
        }
        torch.save(state, model_path)
        logger.info(f"LSTM 模型已保存到: {model_path}")
        self.model.to(self.device) # 移回原设备

    @classmethod # 改为类方法以便直接加载
    def load(cls, model_path: str, device: Optional[str] = None) -> 'LSTMPredictionModel':
        """加载模型"""
        # 确定加载设备
        if device is None:
            load_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            load_device = torch.device(device)

        if not torch.cuda.is_available() and load_device.type == 'cuda':
            logger.warning("GPU 不可用，模型将加载到 CPU。")
            load_device = torch.device('cpu')

        state = torch.load(model_path, map_location=load_device)
        logger.info(f"从 {model_path} 加载 LSTM 模型状态...")

        # --- 提取参数 ---
        num_embeddings_list = state['num_embeddings_list']
        embedding_dims_list = state['embedding_dims_list']
        hidden_size = state['hidden_size']
        target_som_num_nodes = state['target_som_num_nodes']
        num_layers = state.get('num_layers', 2) # 兼容旧版本
        dropout = state['dropout']
        lr = state.get('learning_rate', 0.001) # 获取保存的学习率或默认
        batch_size = state.get('batch_size', 32)
        epochs = state.get('epochs_run', 100) # 可以用 epochs_run 恢复训练状态，或用配置的
        patience = state.get('patience', 10)
        seq_len = state.get('sequence_length', 10)
        history = state.get('history', {'train_loss': [], 'val_loss': []})

        # --- 创建实例 ---
        # 注意：这里使用加载的参数，而不是传入的 __init__ 参数（除了 device）
        instance = cls(
            num_embeddings_list=num_embeddings_list,
            embedding_dims_list=embedding_dims_list,
            hidden_size=hidden_size,
            target_som_num_nodes=target_som_num_nodes,
            num_layers=num_layers,
            dropout=dropout,
            epochs=epochs, # 或者使用配置中的 epochs
            batch_size=batch_size,
            learning_rate=lr,
            patience=patience,
            sequence_length=seq_len,
            device=str(load_device) # 传递设备字符串
            # random_seed 不从 state 加载，应由外部控制
        )
        instance.history = history # 恢复历史记录

        # --- 构建模型并加载状态 ---
        instance.model, instance.optimizer, instance.criterion = instance._build_model()
        instance.model.load_state_dict(state['model_state_dict'])
        if 'optimizer_state_dict' in state and instance.optimizer:
             try:
                 instance.optimizer.load_state_dict(state['optimizer_state_dict'])
             except Exception as e:
                 logger.warning(f"加载优化器状态失败 (可能由于模型结构变化): {e}")


        logger.info(f"LSTM 模型成功加载并配置到设备: {instance.device}")
        instance.model.eval() # 加载后默认设为评估模式
        return instance