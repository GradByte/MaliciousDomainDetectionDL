"""
Model training utilities with callbacks and monitoring.
PyTorch implementation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
from typing import Dict, List, Optional, Tuple
import time
import json
from sklearn.metrics import f1_score


class EarlyStopping:
    """Early stopping to stop training when validation metric stops improving."""
    
    def __init__(self, patience: int = 10, mode: str = 'min', min_delta: float = 0):
        """
        Args:
            patience: Number of epochs to wait before stopping
            mode: 'min' for loss, 'max' for metrics like F1
            min_delta: Minimum change to qualify as improvement
        """
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
        
    def __call__(self, score: float, epoch: int) -> bool:
        """
        Returns True if training should stop.
        """
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False
        
        if self.mode == 'min':
            improved = score < (self.best_score - self.min_delta)
        else:  # max
            improved = score > (self.best_score + self.min_delta)
        
        if improved:
            self.best_score = score
            self.counter = 0
            self.best_epoch = epoch
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        
        return False


class ModelTrainer:
    """
    Handles model training with callbacks and monitoring.
    """
    
    def __init__(self, model: nn.Module, config: Dict, device: Optional[torch.device] = None):
        """
        Initialize trainer.
        
        Args:
            model: PyTorch model to train
            config: Training configuration
            device: Device to train on (CPU/GPU)
        """
        self.model = model
        self.config = config
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        self.model.to(self.device)
        
        print(f"Training on device: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        
        # Training state
        self.history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_f1_score': [],
            'val_f1_weighted': [],
            'epoch_time': [],
            'learning_rate': []
        }
        self.best_model_state = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.writer = None
    
    def setup_training(self, class_weights: Optional[Dict] = None):
        """
        Setup optimizer, loss function, and scheduler.
        
        Args:
            class_weights: Class weights for imbalanced data
        """
        training_config = self.config.get('training', {})
        
        # Optimizer
        learning_rate = training_config.get('learning_rate', 0.001)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Loss function
        if class_weights is not None:
            # Convert class weights dict to tensor
            weight = torch.tensor([class_weights[i] for i in range(len(class_weights))], 
                                 dtype=torch.float32).to(self.device)
            self.criterion = nn.CrossEntropyLoss(weight=weight)
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        # Learning rate scheduler
        if training_config.get('use_lr_scheduler', True):
            lr_factor = training_config.get('lr_factor', 0.5)
            lr_patience = training_config.get('lr_patience', 5)
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=lr_factor,
                patience=lr_patience
            )
            print(f"✓ LR Scheduler: ReduceLROnPlateau (factor={lr_factor}, patience={lr_patience})")
        
        # TensorBoard
        tensorboard_dir = self.config.get('paths', {}).get('tensorboard_dir', 'logs/tensorboard')
        os.makedirs(tensorboard_dir, exist_ok=True)
        self.writer = SummaryWriter(tensorboard_dir)
        
        print(f"✓ Optimizer: Adam (lr={learning_rate})")
        print(f"✓ Loss: CrossEntropyLoss" + (f" (weighted)" if class_weights else ""))
        if self.scheduler:
            print(f"✓ LR Scheduler: ReduceLROnPlateau")
        print(f"✓ TensorBoard: {tensorboard_dir}")
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """
        Train for one epoch.
        
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.train()
        
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def validate_epoch(self, val_loader: DataLoader, num_classes: int = 3) -> Dict:
        """
        Validate for one epoch.
        
        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        
        total_loss = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                
                _, predicted = output.max(1)
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        
        # Calculate metrics
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        
        accuracy = (all_preds == all_targets).mean()
        f1_macro = f1_score(all_targets, all_preds, average='macro', zero_division=0)
        f1_weighted = f1_score(all_targets, all_preds, average='weighted', zero_division=0)
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted
        }
    
    def train(self,
              X_train: np.ndarray,
              y_train: np.ndarray,
              X_val: np.ndarray,
              y_val: np.ndarray,
              class_weights: Optional[Dict] = None) -> Dict:
        """
        Train the model.
        
        Args:
            X_train: Training features
            y_train: Training labels (integer encoded)
            X_val: Validation features
            y_val: Validation labels (integer encoded)
            class_weights: Class weights for imbalanced data
            
        Returns:
            Training history dictionary
        """
        training_config = self.config.get('training', {})
        
        batch_size = training_config.get('batch_size', 512)
        epochs = training_config.get('epochs', 50)
        
        # Setup training
        self.setup_training(class_weights)
        
        # Create data loaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.LongTensor(y_train)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val),
            torch.LongTensor(y_val)
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )
        
        # Early stopping
        early_stopping = None
        if training_config.get('early_stopping', True):
            es_patience = training_config.get('early_stopping_patience', 10)
            es_mode = training_config.get('early_stopping_mode', 'max')
            early_stopping = EarlyStopping(patience=es_patience, mode=es_mode)
            print(f"✓ EarlyStopping: patience={es_patience}, monitor=val_f1_score")
        
        # Model checkpoint path
        saved_models_dir = self.config.get('paths', {}).get('saved_models_dir', 'saved_models')
        os.makedirs(saved_models_dir, exist_ok=True)
        best_model_path = os.path.join(saved_models_dir, 'best_model.pt')
        
        print("\n" + "=" * 70)
        print("STARTING TRAINING")
        print("=" * 70)
        print(f"Training samples: {len(X_train):,}")
        print(f"Validation samples: {len(X_val):,}")
        print(f"Batch size: {batch_size}")
        print(f"Epochs: {epochs}")
        print(f"Device: {self.device}")
        print("=" * 70 + "\n")
        
        # Training loop
        start_time = time.time()
        best_val_f1 = 0.0
        
        for epoch in range(epochs):
            epoch_start = time.time()
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validate
            val_metrics = self.validate_epoch(val_loader)
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_accuracy'].append(train_acc)
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_accuracy'].append(val_metrics['accuracy'])
            self.history['val_f1_score'].append(val_metrics['f1_macro'])
            self.history['val_f1_weighted'].append(val_metrics['f1_weighted'])
            
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['learning_rate'].append(current_lr)
            
            epoch_time = time.time() - epoch_start
            self.history['epoch_time'].append(epoch_time)
            
            # Print progress
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"  train_loss: {train_loss:.4f} - train_acc: {train_acc:.4f}")
            print(f"  val_loss: {val_metrics['loss']:.4f} - val_acc: {val_metrics['accuracy']:.4f}")
            print(f"  val_f1_macro: {val_metrics['f1_macro']:.4f} - val_f1_weighted: {val_metrics['f1_weighted']:.4f}")
            print(f"  time: {epoch_time:.2f}s - lr: {current_lr:.6f}")
            
            # TensorBoard logging
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_metrics['loss'], epoch)
            self.writer.add_scalar('Accuracy/train', train_acc, epoch)
            self.writer.add_scalar('Accuracy/val', val_metrics['accuracy'], epoch)
            self.writer.add_scalar('F1/val_macro', val_metrics['f1_macro'], epoch)
            self.writer.add_scalar('F1/val_weighted', val_metrics['f1_weighted'], epoch)
            self.writer.add_scalar('Learning_Rate', current_lr, epoch)
            
            # Save best model
            if val_metrics['f1_macro'] > best_val_f1:
                best_val_f1 = val_metrics['f1_macro']
                self.best_model_state = self.model.state_dict().copy()
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_f1_macro': val_metrics['f1_macro'],
                    'history': self.history
                }, best_model_path)
                print(f"  ✓ Saved best model (val_f1={best_val_f1:.4f})")
            
            # Learning rate scheduling
            if self.scheduler:
                self.scheduler.step(val_metrics['loss'])
            
            # Early stopping
            if early_stopping:
                if early_stopping(val_metrics['f1_macro'], epoch):
                    print(f"\nEarly stopping triggered at epoch {epoch+1}")
                    print(f"Best epoch was {early_stopping.best_epoch+1} with val_f1={early_stopping.best_score:.4f}")
                    break
            
            print()
        
        total_time = time.time() - start_time
        
        # Restore best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        
        print("\n" + "=" * 70)
        print("TRAINING COMPLETED")
        print("=" * 70)
        print(f"Total training time: {total_time/60:.2f} minutes")
        print(f"Average time per epoch: {total_time/len(self.history['train_loss']):.2f}s")
        print(f"Best validation F1: {best_val_f1:.4f}")
        print("=" * 70 + "\n")
        
        # Close TensorBoard writer
        self.writer.close()
        
        return self.history
    
    def save_model(self, save_path: str):
        """
        Save the trained model.
        
        Args:
            save_path: Path to save the model
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'history': self.history
        }, save_path)
        print(f"Model saved to: {save_path}")
    
    def save_history(self, save_path: str):
        """
        Save training history.
        
        Args:
            save_path: Path to save history JSON
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Convert to JSON-serializable format
        history_dict = {key: [float(val) for val in values] 
                       for key, values in self.history.items()}
        
        with open(save_path, 'w') as f:
            json.dump(history_dict, f, indent=2)
        
        print(f"Training history saved to: {save_path}")
    
    def get_best_epoch_metrics(self) -> Dict:
        """
        Get metrics from the best epoch.
        
        Returns:
            Dictionary of best metrics
        """
        if not self.history['val_loss']:
            return {}
        
        # Find best epoch based on validation F1
        best_epoch = np.argmax(self.history['val_f1_score'])
        
        best_metrics = {
            'epoch': best_epoch + 1,
            'train_loss': self.history['train_loss'][best_epoch],
            'val_loss': self.history['val_loss'][best_epoch],
            'train_accuracy': self.history['train_accuracy'][best_epoch],
            'val_accuracy': self.history['val_accuracy'][best_epoch],
            'val_f1_score': self.history['val_f1_score'][best_epoch],
            'val_f1_weighted': self.history['val_f1_weighted'][best_epoch]
        }
        
        return best_metrics


if __name__ == "__main__":
    print("Testing Model Trainer (PyTorch)")
    print("=" * 50)
    
    # Create dummy model and data
    from models.cnn_lstm_model import build_cnn_lstm_model
    
    input_dim = 50
    num_classes = 3
    
    model = build_cnn_lstm_model(input_dim, num_classes, {'embedding_dim': 32, 'conv_filters': [16], 'lstm_units': 16})
    
    # Dummy data
    np.random.seed(42)
    X_train = np.random.randn(1000, input_dim, 1).astype(np.float32)
    y_train = np.random.randint(0, num_classes, 1000)
    X_val = np.random.randn(200, input_dim, 1).astype(np.float32)
    y_val = np.random.randint(0, num_classes, 200)
    
    # Create config
    config = {
        'training': {
            'batch_size': 32,
            'epochs': 3,
            'learning_rate': 0.001,
            'early_stopping': True,
            'early_stopping_patience': 2,
            'use_lr_scheduler': True
        },
        'paths': {
            'saved_models_dir': 'test_saved_models',
            'logs_dir': 'test_logs',
            'tensorboard_dir': 'test_logs/tensorboard'
        }
    }
    
    # Train
    trainer = ModelTrainer(model, config)
    history = trainer.train(X_train, y_train, X_val, y_val)
    
    print("\nBest epoch metrics:")
    best_metrics = trainer.get_best_epoch_metrics()
    for key, value in best_metrics.items():
        print(f"  {key}: {value}")
    
    print("\nTrainer test completed successfully!")
