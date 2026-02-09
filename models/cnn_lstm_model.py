"""
Hybrid CNN-LSTM model architecture for malicious domain detection.
PyTorch implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import numpy as np


class AttentionLayer(nn.Module):
    """
    Custom attention layer for sequence data.
    """
    
    def __init__(self, input_dim: int, units: int = 64):
        super(AttentionLayer, self).__init__()
        self.units = units
        self.W = nn.Linear(input_dim, units)
        self.u = nn.Linear(units, 1, bias=False)
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
        Returns:
            Attention-weighted output of shape (batch_size, input_dim)
        """
        # Compute attention scores
        uit = torch.tanh(self.W(x))  # (batch, seq_len, units)
        ait = self.u(uit).squeeze(-1)  # (batch, seq_len)
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(ait, dim=1)  # (batch, seq_len)
        
        # Apply attention weights
        attention_weights = attention_weights.unsqueeze(-1)  # (batch, seq_len, 1)
        weighted_input = x * attention_weights  # (batch, seq_len, input_dim)
        
        # Sum over time steps
        output = torch.sum(weighted_input, dim=1)  # (batch, input_dim)
        
        return output


class CNN_LSTM_Model(nn.Module):
    """
    Hybrid CNN-LSTM model for domain classification.
    
    Architecture:
        Input -> Dense Embedding -> 1D CNN -> BiLSTM -> Attention -> Dense -> Output
    """
    
    def __init__(self,
                 input_dim: int,
                 num_classes: int = 3,
                 config: Optional[Dict] = None):
        super(CNN_LSTM_Model, self).__init__()
        
        if config is None:
            config = {}
        
        # Model configuration
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.embedding_dim = config.get('embedding_dim', 256)
        self.conv_filters = config.get('conv_filters', [128, 64])
        self.conv_kernel_size = config.get('conv_kernel_size', 3)
        self.pool_size = config.get('pool_size', 2)
        self.lstm_units = config.get('lstm_units', 128)
        self.bidirectional = config.get('bidirectional', True)
        self.use_attention = config.get('use_attention', True)
        self.attention_units = config.get('attention_units', 64)
        self.dense_units = config.get('dense_units', [128, 64])
        self.dropout_rate = config.get('dropout_rate', 0.3)
        self.dropout_rate_final = config.get('dropout_rate_final', 0.2)
        self.use_batch_norm = config.get('use_batch_norm', True)
        
        # Dense embedding layer
        self.embedding = nn.Linear(input_dim, self.embedding_dim)
        self.embedding_bn = nn.BatchNorm1d(self.embedding_dim) if self.use_batch_norm else None
        self.embedding_dropout = nn.Dropout(self.dropout_rate)
        
        # CNN layers
        self.conv_layers = nn.ModuleList()
        self.conv_bn_layers = nn.ModuleList()
        self.conv_dropout_layers = nn.ModuleList()
        
        in_channels = 1
        for filters in self.conv_filters:
            self.conv_layers.append(
                nn.Conv1d(in_channels, filters, kernel_size=self.conv_kernel_size, padding='same')
            )
            if self.use_batch_norm:
                self.conv_bn_layers.append(nn.BatchNorm1d(filters))
            else:
                self.conv_bn_layers.append(None)
            self.conv_dropout_layers.append(nn.Dropout(self.dropout_rate))
            in_channels = filters
        
        # Max pooling
        self.max_pool = nn.MaxPool1d(kernel_size=self.pool_size)
        
        # Calculate LSTM input size after pooling
        # Input: (batch, embedding_dim) -> (batch, 1, embedding_dim) for CNN
        # After CNN: (batch, conv_filters[-1], embedding_dim)
        # After pooling: (batch, conv_filters[-1], embedding_dim // pool_size)
        lstm_input_size = self.conv_filters[-1]
        
        # BiLSTM layer
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=self.lstm_units,
            num_layers=1,
            batch_first=True,
            bidirectional=self.bidirectional
        )
        self.lstm_dropout = nn.Dropout(self.dropout_rate)
        
        # Attention or pooling
        lstm_output_size = self.lstm_units * 2 if self.bidirectional else self.lstm_units
        if self.use_attention:
            self.attention = AttentionLayer(lstm_output_size, self.attention_units)
            dense_input_size = lstm_output_size
        else:
            self.attention = None
            dense_input_size = lstm_output_size
        
        # Dense layers
        self.dense_layers = nn.ModuleList()
        self.dense_dropout_layers = nn.ModuleList()
        
        prev_units = dense_input_size
        for i, units in enumerate(self.dense_units):
            self.dense_layers.append(nn.Linear(prev_units, units))
            dropout_rate = self.dropout_rate if i < len(self.dense_units) - 1 else self.dropout_rate_final
            self.dense_dropout_layers.append(nn.Dropout(dropout_rate))
            prev_units = units
        
        # Output layer
        self.output_layer = nn.Linear(prev_units, num_classes)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim, 1)
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # Reshape input: (batch, input_dim, 1) -> (batch, input_dim)
        batch_size = x.size(0)
        x = x.squeeze(-1)  # (batch, input_dim)
        
        # Dense embedding
        x = self.embedding(x)  # (batch, embedding_dim)
        x = F.relu(x)
        if self.embedding_bn is not None:
            x = self.embedding_bn(x)
        x = self.embedding_dropout(x)
        
        # Reshape for CNN: (batch, embedding_dim) -> (batch, 1, embedding_dim)
        x = x.unsqueeze(1)  # (batch, 1, embedding_dim)
        
        # CNN layers
        for conv, bn, dropout in zip(self.conv_layers, self.conv_bn_layers, self.conv_dropout_layers):
            x = conv(x)
            x = F.relu(x)
            if bn is not None:
                x = bn(x)
            x = dropout(x)
        
        # Max pooling
        x = self.max_pool(x)  # (batch, conv_filters[-1], seq_len)
        
        # Transpose for LSTM: (batch, channels, seq_len) -> (batch, seq_len, channels)
        x = x.transpose(1, 2)  # (batch, seq_len, conv_filters[-1])
        
        # LSTM
        x, _ = self.lstm(x)  # (batch, seq_len, lstm_units * 2 if bidirectional)
        x = self.lstm_dropout(x)
        
        # Attention or pooling
        if self.use_attention:
            x = self.attention(x)  # (batch, lstm_output_size)
        else:
            x = torch.mean(x, dim=1)  # Global average pooling
        
        # Dense layers
        for dense, dropout in zip(self.dense_layers, self.dense_dropout_layers):
            x = dense(x)
            x = F.relu(x)
            x = dropout(x)
        
        # Output layer
        x = self.output_layer(x)
        
        return x


class SimpleDenseModel(nn.Module):
    """
    Simple fully-connected baseline model.
    """
    
    def __init__(self,
                 input_dim: int,
                 num_classes: int = 3,
                 config: Optional[Dict] = None):
        super(SimpleDenseModel, self).__init__()
        
        if config is None:
            config = {}
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.dense_units = config.get('dense_units', [256, 128, 64])
        self.dropout_rate = config.get('dropout_rate', 0.3)
        self.use_batch_norm = config.get('use_batch_norm', True)
        
        # Build dense layers
        self.layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        
        prev_units = input_dim
        for units in self.dense_units:
            self.layers.append(nn.Linear(prev_units, units))
            if self.use_batch_norm:
                self.bn_layers.append(nn.BatchNorm1d(units))
            else:
                self.bn_layers.append(None)
            self.dropout_layers.append(nn.Dropout(self.dropout_rate))
            prev_units = units
        
        # Output layer
        self.output_layer = nn.Linear(prev_units, num_classes)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim) or (batch_size, input_dim, 1)
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # Flatten if needed
        if len(x.shape) == 3:
            x = x.squeeze(-1)
        
        # Dense layers
        for layer, bn, dropout in zip(self.layers, self.bn_layers, self.dropout_layers):
            x = layer(x)
            x = F.relu(x)
            if bn is not None:
                x = bn(x)
            x = dropout(x)
        
        # Output
        x = self.output_layer(x)
        
        return x


def build_cnn_lstm_model(
    input_dim: int,
    num_classes: int = 3,
    config: Optional[Dict] = None
) -> nn.Module:
    """
    Build hybrid CNN-LSTM model for domain classification.
    
    Args:
        input_dim: Number of input features
        num_classes: Number of output classes
        config: Model configuration dictionary
        
    Returns:
        PyTorch model
    """
    model = CNN_LSTM_Model(input_dim, num_classes, config)
    return model


def build_simple_dense_model(
    input_dim: int,
    num_classes: int = 3,
    config: Optional[Dict] = None
) -> nn.Module:
    """
    Build a simple fully-connected baseline model.
    
    Args:
        input_dim: Number of input features
        num_classes: Number of output classes
        config: Model configuration dictionary
        
    Returns:
        PyTorch model
    """
    model = SimpleDenseModel(input_dim, num_classes, config)
    return model


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    Count trainable and non-trainable parameters.
    
    Args:
        model: PyTorch model
        
    Returns:
        Tuple of (trainable_params, non_trainable_params)
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    
    return trainable, non_trainable


def get_model_summary(model: nn.Module, input_size: Tuple) -> str:
    """
    Get model summary as string.
    
    Args:
        model: PyTorch model
        input_size: Input tensor size (e.g., (50, 1) for features)
        
    Returns:
        Model summary string
    """
    from io import StringIO
    import sys
    
    # Capture summary
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    
    print(model)
    print("\n" + "=" * 80)
    trainable, non_trainable = count_parameters(model)
    print(f"Trainable parameters: {trainable:,}")
    print(f"Non-trainable parameters: {non_trainable:,}")
    print(f"Total parameters: {trainable + non_trainable:,}")
    print("=" * 80)
    
    summary = sys.stdout.getvalue()
    sys.stdout = old_stdout
    
    return summary


if __name__ == "__main__":
    print("Testing CNN-LSTM Model (PyTorch)")
    print("=" * 50)
    
    # Test model building
    input_dim = 50  # Number of features
    num_classes = 3  # benign, phishing, malware
    
    print("\n1. Building CNN-LSTM model...")
    model_config = {
        'embedding_dim': 256,
        'conv_filters': [128, 64],
        'lstm_units': 128,
        'dense_units': [128, 64],
        'use_attention': True
    }
    
    model = build_cnn_lstm_model(input_dim, num_classes, model_config)
    
    print("\nModel Summary:")
    summary = get_model_summary(model, (input_dim, 1))
    print(summary)
    
    # Test with dummy data
    print("\n2. Testing model with dummy data...")
    batch_size = 32
    X_dummy = torch.randn(batch_size, input_dim, 1)
    
    model.eval()
    with torch.no_grad():
        predictions = model(X_dummy)
    
    # Apply softmax to get probabilities
    probabilities = F.softmax(predictions, dim=1)
    
    print(f"Input shape: {X_dummy.shape}")
    print(f"Output shape: {predictions.shape}")
    print(f"Sample prediction (logits): {predictions[0]}")
    print(f"Sample prediction (probabilities): {probabilities[0]}")
    print(f"Probability sum (should be ~1.0): {probabilities[0].sum():.4f}")
    
    print("\n3. Building simple dense model for comparison...")
    dense_model = build_simple_dense_model(input_dim, num_classes)
    
    print("\nDense Model Summary:")
    summary = get_model_summary(dense_model, (input_dim,))
    print(summary)
    
    # Test dense model
    X_dummy_flat = X_dummy.squeeze(-1)
    dense_model.eval()
    with torch.no_grad():
        predictions = dense_model(X_dummy_flat)
    
    print(f"\nDense model output shape: {predictions.shape}")
    
    print("\nModel test completed successfully!")
