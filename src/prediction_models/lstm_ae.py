# src/prediction_models/lstm_pca.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset # DataLoader 现在作为输入类型
import numpy as np
import logging
import time
import os
from typing import Optional, Tuple, List, Dict, Any

# logger = logging.getLogger(__name__) # 建议在调用它的地方获取 logger 或全局配置
log = logging.getLogger(__name__) # 使用 log 保持一致

class LSTMPredictionModel(nn.Module):
    """
    适用于预测连续低维向量的 LSTM 模型。
    已修改 fit/predict 方法以接收 DataLoader。
    """
    def __init__(
        self,
        input_size: int,        # 输入特征数 (例如: latent_salinity_dim + wind_pca_dim)
        hidden_size: int,
        output_size: int,       # 输出特征数 (例如: latent_salinity_dim)
        num_layers: int,
        dropout: float = 0.0,   # Dropout 比率 (构造函数中提供默认值)
        sequence_length: int = 10, # 仅供参考
        random_seed: Optional[int] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout_p = dropout
        self.sequence_length = sequence_length # 存储供参考
        self.random_seed = random_seed
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.random_seed is not None:
            torch.manual_seed(self.random_seed)
            np.random.seed(self.random_seed)
            # 注意：在多进程DataLoader中，随机种子可能需要为每个worker单独设置

        # --- 模型架构 ---
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True, # 输入形状: (batch, seq_len, input_size)
            dropout=self.dropout_p if self.num_layers > 1 else 0 # Dropout 只在层之间
        )

        self.fc = nn.Linear(self.hidden_size, self.output_size)

        # --- 损失函数 (可以在 fit 时定义，但放在这里也可以) ---
        self.criterion = nn.MSELoss()

        self.to(self.device)
        log.info(f"LSTM 模型已在 {self.device} 上初始化")
        log.info(f"Input size: {self.input_size}, Hidden size: {self.hidden_size}, Output size: {self.output_size}, Layers: {self.num_layers}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        模型的前向传播。
        Args:
            x: 输入张量，形状 (batch_size, sequence_length, input_size)。
        Returns:
            输出张量，形状 (batch_size, output_size)。
        """
        # 确保输入在正确的设备上 (虽然 DataLoader 通常处理，但加一道保险)
        x = x.to(self.device)

        # 初始化隐藏状态和细胞状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)

        # LSTM 前向传播
        lstm_out, _ = self.lstm(x, (h0, c0))

        # 只取最后一个时间步的输出
        last_time_step_out = lstm_out[:, -1, :]

        # 通过全连接层
        output = self.fc(last_time_step_out)

        return output

    # --- 修改后的 fit 方法 ---
    def fit(self,
            train_loader: DataLoader,
            val_loader: Optional[DataLoader] = None,
            epochs: int = 100,              # 从调用者传入
            learning_rate: float = 0.001,   # 从调用者传入
            patience: int = 10              # 从调用者传入
           ):
        """
        使用 DataLoader 训练 LSTM 模型。
        Args:
            train_loader: 训练数据的 DataLoader。
            val_loader: (可选) 验证数据的 DataLoader。
            epochs: 训练轮数。
            learning_rate: 优化器的学习率。
            patience: 早停的耐心轮数。
        """
        start_time = time.time()

        # --- 优化器 ---
        # 在 fit 方法内部创建优化器
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        # --- 早停相关变量 ---
        best_val_loss = float('inf')
        epochs_no_improve = 0
        best_model_state = None # 用于存储最佳模型状态

        log.info(f"开始训练 LSTM，共 {epochs} 个 epochs...")
        log.info(f"学习率: {learning_rate}, 早停耐心: {patience}")
        if val_loader:
             log.info(f"使用验证集进行早停。训练集批次数: {len(train_loader)}, 验证集批次数: {len(val_loader)}")
        else:
             log.info(f"不使用验证集。训练集批次数: {len(train_loader)}")


        # --- 训练循环 ---
        for epoch in range(epochs):
            self.train() # 设置模型为训练模式
            epoch_train_loss = 0.0

            for i, batch_data in enumerate(train_loader):
                # 从 DataLoader 获取数据
                # 假设 DataLoader 返回 (batch_X, batch_y)
                if len(batch_data) != 2:
                     log.error(f"训练 DataLoader 应返回包含 2 个元素的元组/列表，但得到 {len(batch_data)} 个。")
                     raise ValueError("训练 DataLoader 格式错误")
                batch_X, batch_y = batch_data
                batch_X = batch_X.to(self.device) # 确保在设备上
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                outputs = self(batch_X) # 前向传播

                # 确保目标形状匹配 (N, output_size)
                if self.output_size > 1 and batch_y.ndim == 1:
                     batch_y = batch_y.view(-1, 1) # Reshape (N,) -> (N, 1) if needed
                elif self.output_size == 1 and batch_y.ndim == 1:
                     batch_y = batch_y.unsqueeze(1) # Reshape (N,) -> (N, 1)

                loss = self.criterion(outputs, batch_y)
                loss.backward() # 反向传播
                optimizer.step() # 更新权重

                epoch_train_loss += loss.item()

            avg_train_loss = epoch_train_loss / len(train_loader) # 计算平均损失

            # --- 验证 ---
            epoch_val_loss = float('inf') # 默认为无穷大
            if val_loader:
                self.eval() # 设置模型为评估模式
                epoch_val_loss_sum = 0.0
                with torch.no_grad():
                    for batch_X_val, batch_y_val in val_loader:
                        batch_X_val = batch_X_val.to(self.device)
                        batch_y_val = batch_y_val.to(self.device)
                        outputs_val = self(batch_X_val)

                        # 确保目标形状匹配
                        if self.output_size > 1 and batch_y_val.ndim == 1:
                             batch_y_val = batch_y_val.view(-1, 1)
                        elif self.output_size == 1 and batch_y_val.ndim == 1:
                             batch_y_val = batch_y_val.unsqueeze(1)

                        val_loss = self.criterion(outputs_val, batch_y_val)
                        epoch_val_loss_sum += val_loss.item()

                epoch_val_loss = epoch_val_loss_sum / len(val_loader)
                log.info(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.6f}, Val Loss: {epoch_val_loss:.6f}")

                # --- 早停检查 ---
                if epoch_val_loss < best_val_loss:
                    best_val_loss = epoch_val_loss
                    epochs_no_improve = 0
                    best_model_state = self.state_dict() # 保存最佳状态
                    log.debug(f"验证损失提升至 {best_val_loss:.6f}。保存模型状态。")
                else:
                    epochs_no_improve += 1
                    log.debug(f"验证损失已连续 {epochs_no_improve} 个 epoch 未提升。")

                if epochs_no_improve >= patience:
                    log.info(f"早停触发于 epoch {epoch + 1}。")
                    if best_model_state:
                        self.load_state_dict(best_model_state) # 加载最佳模型
                        log.info("已加载早停时的最佳模型状态。")
                    break # 退出训练循环
            else:
                # 没有验证集，仅记录训练损失
                log.info(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.6f}")
                # 如果没有验证，将最后的状态视为“最佳”
                best_model_state = self.state_dict()

        # --- 训练后处理 ---
        # 如果训练正常结束（未早停）且有验证集，确保加载的是最佳验证状态
        if val_loader and epochs_no_improve < patience and best_model_state:
             self.load_state_dict(best_model_state)
             log.info("训练正常结束，已加载验证集上的最佳模型状态。")
        # 如果没有验证集，最后的状态已在循环中保存到 best_model_state
        elif not val_loader and best_model_state:
             self.load_state_dict(best_model_state) # 确保加载最后的状态
             log.info("训练正常结束（无验证），已加载最后一个 epoch 的模型状态。")

        training_time = time.time() - start_time
        log.info(f"训练完成。总时间: {training_time:.2f} 秒。")
        if val_loader:
            log.info(f"最终使用的模型的最佳验证损失: {best_val_loss:.6f}")

    # --- 修改后的 predict 方法 ---
    def predict(self, predict_loader: DataLoader) -> np.ndarray:
        """
        使用训练好的 LSTM 模型进行预测。
        Args:
            predict_loader: 包含输入序列 (X) 的 DataLoader。
                            DataLoader 应只返回 X，不需要 y。
        Returns:
            预测的连续值，形状 (N, output_size) 的 NumPy 数组。
        """
        self.eval() # 设置模型为评估模式
        predictions_list = []

        log.info(f"开始在 {len(predict_loader.dataset)} 个序列上进行预测...")
        with torch.no_grad():
            for batch_data in predict_loader:
                # 假设 predict_loader 只返回 X
                if isinstance(batch_data, (list, tuple)):
                    batch_X = batch_data[0].to(self.device)
                else:
                    batch_X = batch_data.to(self.device) # 直接是 X 张量

                outputs = self(batch_X) # 前向传播, shape (batch_size, output_size)
                predictions_list.append(outputs.cpu().numpy())

        predictions = np.concatenate(predictions_list, axis=0)
        log.info(f"预测完成。输出形状: {predictions.shape}")

        # 确保输出形状是 (N, output_size)，即使 output_size=1
        if predictions.ndim == 1 and self.output_size == 1:
             predictions = predictions.reshape(-1, 1)
        elif predictions.ndim == 2 and predictions.shape[1] != self.output_size:
             log.error(f"预测形状不匹配！预期输出大小 {self.output_size}, 得到 {predictions.shape[1]}")
             # 返回一个形状正确的空数组或根据需要处理错误
             return np.empty((len(predict_loader.dataset), self.output_size)) * np.nan

        return predictions # Shape (N, output_size)

    # --- save 和 load 方法基本保持不变，但可以移除训练参数 ---
    def save(self, file_path: str):
        """保存模型状态字典和必要的结构参数。"""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        try:
            save_content = {
                'model_state_dict': self.state_dict(),
                'input_size': self.input_size,
                'hidden_size': self.hidden_size,
                'output_size': self.output_size,
                'num_layers': self.num_layers,
                'dropout': self.dropout_p,
                'sequence_length': self.sequence_length,
            }
            torch.save(save_content, file_path)
            log.info(f"LSTM 模型状态已保存至 {file_path}")
        except Exception as e:
            log.error(f"保存 LSTM 模型到 {file_path} 失败: {e}", exc_info=True)

    @classmethod
    def load(cls, file_path: str, device: Optional[torch.device] = None) -> 'LSTMPredictionModel':
        """加载模型状态字典并创建新的模型实例。"""
        if not os.path.exists(file_path):
            log.error(f"模型文件未找到: {file_path}")
            raise FileNotFoundError(f"模型文件未找到: {file_path}")

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        try:
            checkpoint = torch.load(file_path, map_location=device)
            log.info(f"正在从 {file_path} 加载 LSTM 模型检查点...")

            # 提取参数
            input_size = checkpoint.get('input_size')
            hidden_size = checkpoint.get('hidden_size')
            output_size = checkpoint.get('output_size')
            num_layers = checkpoint.get('num_layers')
            dropout = checkpoint.get('dropout', 0.0)
            sequence_length = checkpoint.get('sequence_length', 10)

            if None in [input_size, hidden_size, output_size, num_layers]:
                raise ValueError("检查点缺少必需的模型参数。")

            # 创建模型实例
            model = cls(
                input_size=input_size,
                hidden_size=hidden_size,
                output_size=output_size,
                num_layers=num_layers,
                dropout=dropout,
                sequence_length=sequence_length,
                device=device
            )

            # 加载状态字典
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(device)
            model.eval() # 加载后设为评估模式
            log.info(f"LSTM 模型已成功从 {file_path} 加载到 {device}")
            return model

        except Exception as e:
            log.error(f"从 {file_path} 加载 LSTM 模型失败: {e}", exc_info=True)
            raise