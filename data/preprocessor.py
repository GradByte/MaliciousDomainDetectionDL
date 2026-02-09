"""
Data preprocessing, normalization, and train/val/test splitting.
"""

import numpy as np
import pickle
import os
from typing import Tuple, Dict, List, Optional
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight


class DNSDataPreprocessor:
    """
    Preprocess DNS features for machine learning.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize preprocessor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.class_weights = None
        self.is_fitted = False
    
    def fit(self, X: np.ndarray, y: List[str]):
        """
        Fit the preprocessor on training data.
        
        Args:
            X: Feature matrix
            y: Labels
        """
        # Fit scaler on features
        self.scaler.fit(X)
        
        # Fit label encoder
        self.label_encoder.fit(y)
        
        # Calculate class weights
        y_encoded = self.label_encoder.transform(y)
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(y_encoded),
            y=y_encoded
        )
        self.class_weights = dict(enumerate(class_weights))
        
        self.is_fitted = True
        
        print("Preprocessor fitted.")
        print(f"Classes: {self.label_encoder.classes_}")
        print(f"Class weights: {self.class_weights}")
    
    def transform(self, X: np.ndarray, y: Optional[List[str]] = None) -> Tuple:
        """
        Transform features and labels.
        
        Args:
            X: Feature matrix
            y: Labels (optional)
            
        Returns:
            Tuple of (transformed_X, transformed_y) or just transformed_X
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")
        
        # Handle NaN and infinite values
        X = np.nan_to_num(X, nan=0.0, posinf=1e10, neginf=-1e10)
        
        # Normalize features
        X_scaled = self.scaler.transform(X)
        
        if y is not None:
            # Encode labels
            y_encoded = self.label_encoder.transform(y)
            return X_scaled, y_encoded
        
        return X_scaled
    
    def fit_transform(self, X: np.ndarray, y: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit and transform in one step.
        
        Args:
            X: Feature matrix
            y: Labels
            
        Returns:
            Tuple of (transformed_X, transformed_y)
        """
        self.fit(X, y)
        return self.transform(X, y)
    
    def inverse_transform_labels(self, y_encoded: np.ndarray) -> np.ndarray:
        """
        Convert encoded labels back to original labels.
        
        Args:
            y_encoded: Encoded labels
            
        Returns:
            Original labels
        """
        return self.label_encoder.inverse_transform(y_encoded)
    
    def save(self, scaler_path: str, encoder_path: str):
        """
        Save preprocessor to disk.
        
        Args:
            scaler_path: Path to save scaler
            encoder_path: Path to save label encoder
        """
        os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
        os.makedirs(os.path.dirname(encoder_path), exist_ok=True)
        
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        with open(encoder_path, 'wb') as f:
            pickle.dump({
                'label_encoder': self.label_encoder,
                'class_weights': self.class_weights
            }, f)
        
        print(f"Preprocessor saved to {scaler_path} and {encoder_path}")
    
    def load(self, scaler_path: str, encoder_path: str):
        """
        Load preprocessor from disk.
        
        Args:
            scaler_path: Path to scaler file
            encoder_path: Path to label encoder file
        """
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        
        with open(encoder_path, 'rb') as f:
            data = pickle.load(f)
            self.label_encoder = data['label_encoder']
            self.class_weights = data['class_weights']
        
        self.is_fitted = True
        print(f"Preprocessor loaded from {scaler_path} and {encoder_path}")
    
    def get_num_classes(self) -> int:
        """Get number of classes."""
        return len(self.label_encoder.classes_)
    
    def get_class_names(self) -> List[str]:
        """Get class names."""
        return list(self.label_encoder.classes_)


def split_data(X: np.ndarray, 
               y: np.ndarray, 
               train_ratio: float = 0.7,
               val_ratio: float = 0.15,
               test_ratio: float = 0.15,
               random_seed: int = 42) -> Tuple:
    """
    Split data into train, validation, and test sets.
    
    Args:
        X: Feature matrix
        y: Labels
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        random_seed: Random seed
        
    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"
    
    # First split: train + val vs test
    test_size = test_ratio
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_seed, stratify=y
    )
    
    # Second split: train vs val
    val_size = val_ratio / (train_ratio + val_ratio)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size, random_state=random_seed, stratify=y_temp
    )
    
    print(f"\nData split:")
    print(f"  Training set: {len(X_train):,} samples ({len(X_train)/len(X)*100:.1f}%)")
    print(f"  Validation set: {len(X_val):,} samples ({len(X_val)/len(X)*100:.1f}%)")
    print(f"  Test set: {len(X_test):,} samples ({len(X_test)/len(X)*100:.1f}%)")
    
    # Print class distribution
    print(f"\nClass distribution:")
    for split_name, y_split in [('Train', y_train), ('Val', y_val), ('Test', y_test)]:
        unique, counts = np.unique(y_split, return_counts=True)
        print(f"  {split_name}:")
        for cls, count in zip(unique, counts):
            print(f"    Class {cls}: {count:,} ({count/len(y_split)*100:.1f}%)")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def prepare_data_for_cnn(X: np.ndarray, reshape: bool = True) -> np.ndarray:
    """
    Prepare data for CNN input (add channel dimension if needed).
    
    Args:
        X: Feature matrix of shape (n_samples, n_features)
        reshape: Whether to reshape for CNN
        
    Returns:
        Reshaped array for CNN input (n_samples, n_features, 1)
    """
    if reshape and len(X.shape) == 2:
        # Add channel dimension for 1D CNN
        X = np.expand_dims(X, axis=-1)
    return X


def convert_labels_to_categorical(y: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Convert integer labels to one-hot encoded categorical labels.
    
    Note: PyTorch doesn't require one-hot encoding for classification.
    This function is provided for compatibility only.
    
    Args:
        y: Integer labels
        num_classes: Number of classes
        
    Returns:
        One-hot encoded labels
    """
    # Use NumPy for framework-agnostic one-hot encoding
    one_hot = np.zeros((y.size, num_classes))
    one_hot[np.arange(y.size), y] = 1
    return one_hot


class DataAugmenter:
    """
    Data augmentation for DNS features (SMOTE-like approach).
    """
    
    def __init__(self, noise_factor: float = 0.01):
        """
        Initialize augmenter.
        
        Args:
            noise_factor: Amount of noise to add
        """
        self.noise_factor = noise_factor
    
    def augment(self, X: np.ndarray, n_samples: int) -> np.ndarray:
        """
        Augment data by adding noise and interpolation.
        
        Args:
            X: Feature matrix
            n_samples: Number of augmented samples to generate
            
        Returns:
            Augmented feature matrix
        """
        augmented = []
        
        for _ in range(n_samples):
            # Select two random samples
            idx1, idx2 = np.random.choice(len(X), size=2, replace=True)
            
            # Interpolate
            alpha = np.random.random()
            sample = alpha * X[idx1] + (1 - alpha) * X[idx2]
            
            # Add small noise
            noise = np.random.normal(0, self.noise_factor, sample.shape)
            sample = sample + noise
            
            augmented.append(sample)
        
        return np.array(augmented)


def balance_dataset(X: np.ndarray, 
                   y: np.ndarray, 
                   strategy: str = 'undersample',
                   target_ratio: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Balance dataset using undersampling or oversampling.
    
    Args:
        X: Feature matrix
        y: Labels
        strategy: 'undersample', 'oversample', or 'weighted'
        target_ratio: Target ratio for minority/majority classes
        
    Returns:
        Balanced (X, y)
    """
    unique_classes, class_counts = np.unique(y, return_counts=True)
    
    if strategy == 'weighted':
        # Don't modify data, just return as-is (will use class weights)
        return X, y
    
    elif strategy == 'undersample':
        # Undersample majority classes
        min_count = int(np.min(class_counts) / target_ratio)
        
        balanced_X = []
        balanced_y = []
        
        for cls in unique_classes:
            cls_mask = (y == cls)
            cls_X = X[cls_mask]
            cls_y = y[cls_mask]
            
            # Sample up to min_count
            n_samples = min(len(cls_X), min_count)
            indices = np.random.choice(len(cls_X), size=n_samples, replace=False)
            
            balanced_X.append(cls_X[indices])
            balanced_y.append(cls_y[indices])
        
        X_balanced = np.vstack(balanced_X)
        y_balanced = np.concatenate(balanced_y)
        
        # Shuffle
        indices = np.random.permutation(len(X_balanced))
        return X_balanced[indices], y_balanced[indices]
    
    elif strategy == 'oversample':
        # Oversample minority classes
        max_count = int(np.max(class_counts))
        
        balanced_X = []
        balanced_y = []
        
        augmenter = DataAugmenter()
        
        for cls in unique_classes:
            cls_mask = (y == cls)
            cls_X = X[cls_mask]
            cls_y = y[cls_mask]
            
            balanced_X.append(cls_X)
            balanced_y.append(cls_y)
            
            # Generate synthetic samples if needed
            if len(cls_X) < max_count:
                n_augment = max_count - len(cls_X)
                augmented_X = augmenter.augment(cls_X, n_augment)
                augmented_y = np.full(n_augment, cls)
                
                balanced_X.append(augmented_X)
                balanced_y.append(augmented_y)
        
        X_balanced = np.vstack(balanced_X)
        y_balanced = np.concatenate(balanced_y)
        
        # Shuffle
        indices = np.random.permutation(len(X_balanced))
        return X_balanced[indices], y_balanced[indices]
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


if __name__ == "__main__":
    print("Testing DNS Data Preprocessor")
    print("=" * 50)
    
    # Create dummy data
    np.random.seed(42)
    X = np.random.randn(1000, 50)
    y = np.random.choice(['benign', 'phishing', 'malware'], size=1000)
    
    print(f"\nOriginal data shape: {X.shape}")
    print(f"Original labels shape: {y.shape}")
    
    # Test preprocessor
    preprocessor = DNSDataPreprocessor()
    X_scaled, y_encoded = preprocessor.fit_transform(X, y)
    
    print(f"\nScaled data shape: {X_scaled.shape}")
    print(f"Encoded labels shape: {y_encoded.shape}")
    print(f"Unique encoded labels: {np.unique(y_encoded)}")
    
    # Test data split
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        X_scaled, y_encoded, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15
    )
    
    print("\nPreprocessor test completed successfully!")

