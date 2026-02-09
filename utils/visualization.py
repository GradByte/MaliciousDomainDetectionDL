"""
Visualization utilities for training history and model analysis.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Optional
import os


# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def plot_training_history(history: Dict,
                          metrics: List[str] = None,
                          save_path: Optional[str] = None):
    """
    Plot training history for multiple metrics.
    
    Args:
        history: Training history dictionary
        metrics: List of metrics to plot (None for all)
        save_path: Path to save plot (optional)
    """
    if metrics is None:
        # Auto-detect metrics (exclude validation metrics and special keys)
        metrics = [key for key in history.keys() 
                  if not key.startswith('val_') 
                  and key not in ['epoch_time', 'learning_rate']]
    
    n_metrics = len(metrics)
    if n_metrics == 0:
        print("No metrics to plot")
        return
    
    # Create subplots
    n_cols = min(2, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
    if n_metrics == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        # Plot training metric
        if metric in history:
            epochs = range(1, len(history[metric]) + 1)
            ax.plot(epochs, history[metric], 'b-', label=f'Training {metric}', linewidth=2)
        
        # Plot validation metric if available
        val_metric = f'val_{metric}'
        if val_metric in history:
            epochs = range(1, len(history[val_metric]) + 1)
            ax.plot(epochs, history[val_metric], 'r-', label=f'Validation {metric}', linewidth=2)
        
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=11)
        ax.set_title(f'{metric.replace("_", " ").title()} Over Epochs', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_metrics, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to: {save_path}")
    
    plt.close()


def plot_loss_and_accuracy(history: Dict,
                           save_path: Optional[str] = None):
    """
    Plot loss and accuracy in a 2-subplot figure.
    
    Args:
        history: Training history dictionary
        save_path: Path to save plot (optional)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Handle both TensorFlow ('loss') and PyTorch ('train_loss') formats
    loss_key = 'train_loss' if 'train_loss' in history else 'loss'
    acc_key = 'train_accuracy' if 'train_accuracy' in history else 'accuracy'
    
    epochs = range(1, len(history[loss_key]) + 1)
    
    # Plot loss
    ax1.plot(epochs, history[loss_key], 'b-', label='Training Loss', linewidth=2)
    if 'val_loss' in history:
        ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Model Loss', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=11)
    ax1.grid(alpha=0.3)
    
    # Plot accuracy
    ax2.plot(epochs, history[acc_key], 'b-', label='Training Accuracy', linewidth=2)
    if 'val_accuracy' in history:
        ax2.plot(epochs, history['val_accuracy'], 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Model Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=11)
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Loss and accuracy plot saved to: {save_path}")
    
    plt.close()


def plot_learning_rate(history: Dict,
                       save_path: Optional[str] = None):
    """
    Plot learning rate schedule if available.
    
    Args:
        history: Training history dictionary
        save_path: Path to save plot (optional)
    """
    if 'lr' not in history and 'learning_rate' not in history:
        print("No learning rate information in history")
        return
    
    lr_key = 'lr' if 'lr' in history else 'learning_rate'
    
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(history[lr_key]) + 1)
    plt.plot(epochs, history[lr_key], 'g-', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Learning Rate', fontsize=12)
    plt.title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    plt.yscale('log')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Learning rate plot saved to: {save_path}")
    
    plt.close()


def plot_class_distribution(y: np.ndarray,
                           class_names: List[str],
                           title: str = "Class Distribution",
                           save_path: Optional[str] = None):
    """
    Plot class distribution as a bar chart.
    
    Args:
        y: Labels (integer or one-hot)
        class_names: List of class names
        title: Plot title
        save_path: Path to save plot (optional)
    """
    # Convert one-hot to integer if needed
    if len(y.shape) > 1 and y.shape[1] > 1:
        y = np.argmax(y, axis=1)
    
    unique, counts = np.unique(y, return_counts=True)
    
    plt.figure(figsize=(10, 6))
    colors = sns.color_palette("husl", len(class_names))
    bars = plt.bar(range(len(unique)), counts, color=colors)
    
    # Add value labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(count):,}\n({count/len(y)*100:.1f}%)',
                ha='center', va='bottom', fontsize=10)
    
    plt.xlabel('Class', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xticks(range(len(unique)), [class_names[i] for i in unique], fontsize=11)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Class distribution plot saved to: {save_path}")
    
    plt.close()


def plot_feature_importance(feature_names: List[str],
                           importance_scores: np.ndarray,
                           top_n: int = 20,
                           save_path: Optional[str] = None):
    """
    Plot feature importance scores.
    
    Args:
        feature_names: List of feature names
        importance_scores: Importance scores for each feature
        top_n: Number of top features to show
        save_path: Path to save plot (optional)
    """
    # Get top N features
    indices = np.argsort(importance_scores)[-top_n:]
    top_features = [feature_names[i] for i in indices]
    top_scores = importance_scores[indices]
    
    plt.figure(figsize=(10, max(6, top_n * 0.3)))
    colors = plt.cm.viridis(np.linspace(0, 1, len(top_scores)))
    plt.barh(range(len(top_scores)), top_scores, color=colors)
    plt.yticks(range(len(top_scores)), top_features, fontsize=10)
    plt.xlabel('Importance Score', fontsize=12)
    plt.title(f'Top {top_n} Most Important Features', fontsize=14, fontweight='bold')
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Feature importance plot saved to: {save_path}")
    
    plt.close()


def plot_prediction_confidence(y_pred_proba: np.ndarray,
                               y_true: Optional[np.ndarray] = None,
                               save_path: Optional[str] = None):
    """
    Plot distribution of prediction confidence scores.
    
    Args:
        y_pred_proba: Predicted probabilities
        y_true: True labels (optional, for coloring by correctness)
        save_path: Path to save plot (optional)
    """
    # Get max probability (confidence) for each prediction
    confidences = np.max(y_pred_proba, axis=1)
    
    plt.figure(figsize=(10, 6))
    
    if y_true is not None:
        # Convert one-hot to integer if needed
        if len(y_true.shape) > 1:
            y_true = np.argmax(y_true, axis=1)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Separate correct and incorrect predictions
        correct_mask = (y_true == y_pred)
        
        plt.hist(confidences[correct_mask], bins=50, alpha=0.7, 
                label=f'Correct ({np.sum(correct_mask)})', color='green')
        plt.hist(confidences[~correct_mask], bins=50, alpha=0.7,
                label=f'Incorrect ({np.sum(~correct_mask)})', color='red')
        plt.legend(fontsize=11)
    else:
        plt.hist(confidences, bins=50, alpha=0.7, color='blue')
    
    plt.xlabel('Prediction Confidence', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title('Distribution of Prediction Confidence', fontsize=14, fontweight='bold')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Prediction confidence plot saved to: {save_path}")
    
    plt.close()


def create_evaluation_report(history: Dict,
                            metrics: Dict,
                            class_names: List[str],
                            output_dir: str):
    """
    Create comprehensive evaluation report with all plots.
    
    Args:
        history: Training history dictionary
        metrics: Evaluation metrics dictionary
        class_names: List of class names
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("\nGenerating evaluation visualizations...")
    
    # 1. Training history
    plot_training_history(
        history,
        save_path=os.path.join(output_dir, 'training_history.png')
    )
    
    # 2. Loss and accuracy
    plot_loss_and_accuracy(
        history,
        save_path=os.path.join(output_dir, 'loss_accuracy.png')
    )
    
    # 3. Learning rate (if available)
    if 'lr' in history or 'learning_rate' in history:
        plot_learning_rate(
            history,
            save_path=os.path.join(output_dir, 'learning_rate.png')
        )
    
    print(f"Evaluation visualizations saved to: {output_dir}")


if __name__ == "__main__":
    print("Testing Visualization Utilities")
    print("=" * 50)
    
    # Create dummy history
    epochs = 20
    history = {
        'loss': np.linspace(2.0, 0.5, epochs) + np.random.randn(epochs) * 0.05,
        'val_loss': np.linspace(2.0, 0.6, epochs) + np.random.randn(epochs) * 0.08,
        'accuracy': np.linspace(0.5, 0.95, epochs) + np.random.randn(epochs) * 0.02,
        'val_accuracy': np.linspace(0.5, 0.92, epochs) + np.random.randn(epochs) * 0.03,
        'precision': np.linspace(0.5, 0.94, epochs),
        'val_precision': np.linspace(0.5, 0.91, epochs),
    }
    
    print("\n1. Plotting training history...")
    plot_training_history(history)
    
    print("\n2. Plotting loss and accuracy...")
    plot_loss_and_accuracy(history)
    
    print("\n3. Plotting class distribution...")
    y = np.random.choice([0, 1, 2], size=1000, p=[0.6, 0.25, 0.15])
    class_names = ['benign', 'phishing', 'malware']
    plot_class_distribution(y, class_names)
    
    print("\n4. Plotting prediction confidence...")
    y_pred_proba = np.random.dirichlet(np.ones(3), size=1000)
    y_true = np.random.randint(0, 3, size=1000)
    plot_prediction_confidence(y_pred_proba, y_true)
    
    print("\nVisualization test completed successfully!")

