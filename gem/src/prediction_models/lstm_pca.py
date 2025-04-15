# src/prediction_models/lstm_pca.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import logging
import time
import os
from typing import Optional, Tuple, List, Dict, Any

logger = logging.getLogger(__name__)

class LSTMPredictionModel(nn.Module):
    """
    LSTM model adapted for predicting continuous low-dimensional PCA components.
    """
    def __init__(
        self,
        input_size: int,          # Number of input features (total PCA components from all input fields)
        hidden_size: int,
        output_size: int,         # Number of output features (PCA components for the target field)
        num_layers: int,
        dropout: float,
        # --- Training Params ---
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        patience: int = 10,
        # --- Other ---
        sequence_length: int = 10, # For reference/validation if needed
        random_seed: Optional[int] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout_p = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.patience = patience
        self.sequence_length = sequence_length # Store for potential use
        self.random_seed = random_seed
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.random_seed is not None:
            torch.manual_seed(self.random_seed)
            np.random.seed(self.random_seed)

        # --- Model Architecture ---
        # Removed Embedding layers

        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True, # Input shape: (batch, seq_len, input_size)
            dropout=self.dropout_p if self.num_layers > 1 else 0 # Dropout only between layers
        )

        # Fully connected layer to map LSTM output to the desired output size (PCA components)
        self.fc = nn.Linear(self.hidden_size, self.output_size)

        # --- Loss Function ---
        self.criterion = nn.MSELoss() # Use Mean Squared Error for continuous values

        self.to(self.device)
        logger.info(f"LSTM (PCA) Model initialized on {self.device}")
        logger.info(f"Input size: {self.input_size}, Hidden size: {self.hidden_size}, Output size: {self.output_size}, Layers: {self.num_layers}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the LSTM model.
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
               containing continuous PCA component values.
        Returns:
            Output tensor of shape (batch_size, output_size) representing predicted PCA components.
        """
        # x is already continuous, no embedding needed
        # Input shape: (batch, seq_len, input_size)

        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)

        # LSTM forward pass
        # lstm_out shape: (batch, seq_len, hidden_size)
        lstm_out, _ = self.lstm(x, (h0, c0))

        # We only need the output from the last time step
        # last_time_step_out shape: (batch, hidden_size)
        last_time_step_out = lstm_out[:, -1, :]

        # Pass through the fully connected layer
        # output shape: (batch, output_size)
        output = self.fc(last_time_step_out)

        return output

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None):
        """
        Train the LSTM model.
        Args:
            X_train: Training input sequences, shape (N_train, seq_len, input_size).
            y_train: Training target values, shape (N_train, output_size) or (N_train,).
            X_val: Validation input sequences, shape (N_val, seq_len, input_size).
            y_val: Validation target values, shape (N_val, output_size) or (N_val,).
        """
        start_time = time.time()

        # --- Data Preparation ---
        # Ensure y has 2 dimensions if output_size > 1
        if self.output_size > 1 and y_train.ndim == 1:
            y_train = y_train.reshape(-1, 1) # Should already be handled by create_sequences, but double-check
        if y_val is not None and self.output_size > 1 and y_val.ndim == 1:
            y_val = y_val.reshape(-1, 1)

        # Convert numpy arrays to PyTorch tensors
        X_train_tensor = torch.from_numpy(X_train).float().to(self.device)
        y_train_tensor = torch.from_numpy(y_train).float().to(self.device)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        val_loader = None
        if X_val is not None and y_val is not None:
            X_val_tensor = torch.from_numpy(X_val).float().to(self.device)
            y_val_tensor = torch.from_numpy(y_val).float().to(self.device)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
            logger.info(f"Training with validation set (Size: {len(X_val)})")
        else:
            logger.info("Training without validation set.")

        # --- Optimizer ---
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

        # --- Early Stopping ---
        best_val_loss = float('inf')
        epochs_no_improve = 0
        best_model_state = None

        # --- Training Loop ---
        logger.info(f"Starting training for {self.epochs} epochs...")
        for epoch in range(self.epochs):
            self.train() # Set model to training mode
            epoch_train_loss = 0.0
            num_batches = 0

            for i, (batch_X, batch_y) in enumerate(train_loader):
                optimizer.zero_grad()
                outputs = self(batch_X) # Forward pass

                # Ensure target shape matches output shape for MSELoss
                if self.output_size == 1 and batch_y.ndim == 1:
                     batch_y = batch_y.unsqueeze(1) # Make it (batch_size, 1)

                loss = self.criterion(outputs, batch_y)
                loss.backward() # Backward pass
                optimizer.step() # Update weights

                epoch_train_loss += loss.item()
                num_batches += 1

            avg_train_loss = epoch_train_loss / num_batches

            # --- Validation ---
            epoch_val_loss = float('inf')
            if val_loader:
                self.eval() # Set model to evaluation mode
                epoch_val_loss = 0.0
                num_val_batches = 0
                with torch.no_grad():
                    for batch_X_val, batch_y_val in val_loader:
                        outputs_val = self(batch_X_val)
                        if self.output_size == 1 and batch_y_val.ndim == 1:
                            batch_y_val = batch_y_val.unsqueeze(1)
                        val_loss = self.criterion(outputs_val, batch_y_val)
                        epoch_val_loss += val_loss.item()
                        num_val_batches += 1
                epoch_val_loss /= num_val_batches
                logger.info(f"Epoch [{epoch+1}/{self.epochs}], Train Loss: {avg_train_loss:.6f}, Val Loss: {epoch_val_loss:.6f}")

                # --- Early Stopping Check ---
                if epoch_val_loss < best_val_loss:
                    best_val_loss = epoch_val_loss
                    epochs_no_improve = 0
                    # Save the best model state
                    best_model_state = self.state_dict()
                    logger.debug(f"Validation loss improved to {best_val_loss:.6f}. Saving model state.")
                else:
                    epochs_no_improve += 1
                    logger.debug(f"Validation loss did not improve for {epochs_no_improve} epoch(s).")

                if epochs_no_improve >= self.patience:
                    logger.info(f"Early stopping triggered after {epoch + 1} epochs.")
                    if best_model_state:
                         self.load_state_dict(best_model_state) # Load the best model
                         logger.info("Loaded best model state due to early stopping.")
                    break # Exit training loop
            else:
                # No validation, just log training loss
                logger.info(f"Epoch [{epoch+1}/{self.epochs}], Train Loss: {avg_train_loss:.6f}")
                # If no validation, save the last model state potentially
                best_model_state = self.state_dict() # Keep track of last state

        # --- Post-Training ---
        # If training finished without early stopping, ensure the last/best state is loaded
        if not val_loader and best_model_state: # No validation, load last state
             self.load_state_dict(best_model_state)
        elif val_loader and epochs_no_improve < self.patience and best_model_state: # Finished epochs, load best val state
             self.load_state_dict(best_model_state)


        training_time = time.time() - start_time
        logger.info(f"Training finished. Total time: {training_time:.2f} seconds.")
        if val_loader:
            logger.info(f"Best validation loss achieved: {best_val_loss:.6f}")


    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained LSTM model.
        Args:
            X: Input sequences, shape (N, seq_len, input_size).
        Returns:
            Predicted continuous values, shape (N, output_size).
        """
        self.eval() # Set model to evaluation mode
        X_tensor = torch.from_numpy(X).float().to(self.device)
        predictions_list = []

        # Use DataLoader for potentially large prediction sets
        predict_dataset = TensorDataset(X_tensor)
        predict_loader = DataLoader(predict_dataset, batch_size=self.batch_size, shuffle=False)

        logger.info(f"Starting prediction on {len(X)} sequences...")
        with torch.no_grad():
            for (batch_X,) in predict_loader: # Note the comma for unpacking
                outputs = self(batch_X) # Shape (batch_size, output_size)
                predictions_list.append(outputs.cpu().numpy())

        predictions = np.concatenate(predictions_list, axis=0)
        logger.info(f"Prediction finished. Output shape: {predictions.shape}")
        # Ensure output shape is (N, output_size) even if output_size is 1
        if predictions.ndim == 1 and self.output_size == 1:
             predictions = predictions.reshape(-1, 1)
        elif predictions.shape[1] != self.output_size:
             logger.error(f"Prediction shape mismatch! Expected output size {self.output_size}, got {predictions.shape[1]}")
             # Handle error appropriately, maybe return empty or raise exception
             return np.empty((len(X), self.output_size))


        return predictions # Shape (N, output_size)

    def save(self, file_path: str):
        """Saves the model state dictionary."""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        try:
            torch.save({
                'model_state_dict': self.state_dict(),
                'input_size': self.input_size,
                'hidden_size': self.hidden_size,
                'output_size': self.output_size,
                'num_layers': self.num_layers,
                'dropout': self.dropout_p,
                'sequence_length': self.sequence_length, # Save sequence length used during training
                # Add other relevant parameters if needed for reloading
            }, file_path)
            logger.info(f"Model state saved to {file_path}")
        except Exception as e:
            logger.error(f"Failed to save model to {file_path}: {e}", exc_info=True)

    @classmethod
    def load(cls, file_path: str, device: Optional[torch.device] = None) -> 'LSTMPredictionModel':
        """Loads a model state dictionary and creates a new model instance."""
        if not os.path.exists(file_path):
            logger.error(f"Model file not found: {file_path}")
            raise FileNotFoundError(f"Model file not found: {file_path}")

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        try:
            checkpoint = torch.load(file_path, map_location=device)
            logger.info(f"Loading model checkpoint from {file_path}")

            # --- Extract parameters ---
            input_size = checkpoint.get('input_size')
            hidden_size = checkpoint.get('hidden_size')
            output_size = checkpoint.get('output_size')
            num_layers = checkpoint.get('num_layers')
            dropout = checkpoint.get('dropout', 0.0) # Default dropout if not saved
            sequence_length = checkpoint.get('sequence_length', 10) # Default seq len

            if None in [input_size, hidden_size, output_size, num_layers]:
                 raise ValueError("Checkpoint missing required model parameters (input_size, hidden_size, output_size, num_layers).")


            # --- Create model instance ---
            # Note: Training parameters like epochs, lr, patience are not needed for inference
            # but sequence_length might be useful context.
            model = cls(
                input_size=input_size,
                hidden_size=hidden_size,
                output_size=output_size,
                num_layers=num_layers,
                dropout=dropout,
                sequence_length=sequence_length, # Pass loaded sequence length
                device=device
                # Other params like epochs, lr, patience use defaults or aren't needed for prediction
            )

            # --- Load state dict ---
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(device)
            model.eval() # Set to evaluation mode after loading
            logger.info(f"Model loaded successfully from {file_path} to {device}")
            return model

        except Exception as e:
            logger.error(f"Failed to load model from {file_path}: {e}", exc_info=True)
            raise