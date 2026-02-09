"""
Model evaluation utilities: metrics, confusion matrix, classification reports.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_curve, auc, roc_auc_score
)
from typing import Dict, List, Optional, Tuple
import json
import os


class ModelEvaluator:
    """
    Comprehensive model evaluation.
    """
    
    def __init__(self, class_names: List[str]):
        """
        Initialize evaluator.
        
        Args:
            class_names: List of class names
        """
        self.class_names = class_names
        self.num_classes = len(class_names)
    
    def evaluate(self, 
                y_true: np.ndarray, 
                y_pred: np.ndarray,
                y_pred_proba: Optional[np.ndarray] = None) -> Dict:
        """
        Comprehensive evaluation with multiple metrics.
        
        Args:
            y_true: True labels (integer or one-hot)
            y_pred: Predicted labels (integer or one-hot)
            y_pred_proba: Predicted probabilities (optional)
            
        Returns:
            Dictionary of metrics
        """
        # Convert one-hot to integer labels if needed
        if len(y_true.shape) > 1 and y_true.shape[1] > 1:
            y_true = np.argmax(y_true, axis=1)
        if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
            y_pred = np.argmax(y_pred, axis=1)
        
        metrics = {}
        
        # Overall metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Per-class metrics
        precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        for i, class_name in enumerate(self.class_names):
            metrics[f'precision_{class_name}'] = precision_per_class[i]
            metrics[f'recall_{class_name}'] = recall_per_class[i]
            metrics[f'f1_{class_name}'] = f1_per_class[i]
        
        # ROC-AUC if probabilities provided
        if y_pred_proba is not None:
            try:
                # One-vs-Rest ROC-AUC
                y_true_onehot = np.eye(self.num_classes)[y_true]
                metrics['roc_auc_macro'] = roc_auc_score(
                    y_true_onehot, y_pred_proba, average='macro', multi_class='ovr'
                )
                metrics['roc_auc_weighted'] = roc_auc_score(
                    y_true_onehot, y_pred_proba, average='weighted', multi_class='ovr'
                )
                
                # Per-class ROC-AUC
                for i, class_name in enumerate(self.class_names):
                    metrics[f'roc_auc_{class_name}'] = roc_auc_score(
                        y_true_onehot[:, i], y_pred_proba[:, i]
                    )
            except Exception as e:
                print(f"Could not compute ROC-AUC: {e}")
        
        return metrics
    
    def print_metrics(self, metrics: Dict):
        """
        Print metrics in a formatted way.
        
        Args:
            metrics: Dictionary of metrics
        """
        print("\n" + "=" * 70)
        print("MODEL EVALUATION METRICS")
        print("=" * 70)
        
        # Overall metrics
        print("\nOverall Metrics:")
        print(f"  Accuracy:           {metrics['accuracy']:.4f}")
        print(f"  Precision (macro):  {metrics['precision_macro']:.4f}")
        print(f"  Recall (macro):     {metrics['recall_macro']:.4f}")
        print(f"  F1 Score (macro):   {metrics['f1_macro']:.4f}")
        
        if 'roc_auc_macro' in metrics:
            print(f"  ROC-AUC (macro):    {metrics['roc_auc_macro']:.4f}")
        
        # Per-class metrics
        print("\nPer-Class Metrics:")
        for class_name in self.class_names:
            print(f"\n  {class_name.capitalize()}:")
            print(f"    Precision: {metrics[f'precision_{class_name}']:.4f}")
            print(f"    Recall:    {metrics[f'recall_{class_name}']:.4f}")
            print(f"    F1 Score:  {metrics[f'f1_{class_name}']:.4f}")
            if f'roc_auc_{class_name}' in metrics:
                print(f"    ROC-AUC:   {metrics[f'roc_auc_{class_name}']:.4f}")
        
        print("=" * 70 + "\n")
    
    def save_metrics(self, metrics: Dict, save_path: str):
        """
        Save metrics to JSON file.
        
        Args:
            metrics: Dictionary of metrics
            save_path: Path to save JSON
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"Metrics saved to: {save_path}")
    
    def plot_confusion_matrix(self, 
                             y_true: np.ndarray, 
                             y_pred: np.ndarray,
                             save_path: Optional[str] = None,
                             normalize: bool = True):
        """
        Plot confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            save_path: Path to save plot (optional)
            normalize: Whether to normalize
        """
        # Convert one-hot to integer labels if needed
        if len(y_true.shape) > 1 and y_true.shape[1] > 1:
            y_true = np.argmax(y_true, axis=1)
        if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
            y_pred = np.argmax(y_pred, axis=1)
        
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Plot
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='.2f' if normalize else 'd',
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cbar_kws={'label': 'Normalized Count' if normalize else 'Count'}
        )
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.title('Confusion Matrix' + (' (Normalized)' if normalize else ''), fontsize=14)
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to: {save_path}")
        
        plt.close()
    
    def plot_roc_curves(self,
                       y_true: np.ndarray,
                       y_pred_proba: np.ndarray,
                       save_path: Optional[str] = None):
        """
        Plot ROC curves for each class.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            save_path: Path to save plot (optional)
        """
        # Convert to one-hot if needed
        if len(y_true.shape) == 1:
            y_true = np.eye(self.num_classes)[y_true]
        
        plt.figure(figsize=(10, 8))
        
        # Plot ROC curve for each class
        for i, class_name in enumerate(self.class_names):
            fpr, tpr, _ = roc_curve(y_true[:, i], y_pred_proba[:, i])
            roc_auc = auc(fpr, tpr)
            
            plt.plot(
                fpr, tpr, 
                label=f'{class_name.capitalize()} (AUC = {roc_auc:.3f})',
                linewidth=2
            )
        
        # Plot diagonal
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves - Multi-Class', fontsize=14)
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"ROC curves saved to: {save_path}")
        
        plt.close()
    
    def generate_classification_report(self,
                                      y_true: np.ndarray,
                                      y_pred: np.ndarray,
                                      save_path: Optional[str] = None) -> str:
        """
        Generate and optionally save classification report.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            save_path: Path to save report (optional)
            
        Returns:
            Classification report string
        """
        # Convert one-hot to integer labels if needed
        if len(y_true.shape) > 1 and y_true.shape[1] > 1:
            y_true = np.argmax(y_true, axis=1)
        if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
            y_pred = np.argmax(y_pred, axis=1)
        
        report = classification_report(
            y_true, y_pred,
            target_names=self.class_names,
            digits=4
        )
        
        print("\n" + "=" * 70)
        print("CLASSIFICATION REPORT")
        print("=" * 70)
        print(report)
        print("=" * 70 + "\n")
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'w') as f:
                f.write("CLASSIFICATION REPORT\n")
                f.write("=" * 70 + "\n")
                f.write(report)
                f.write("=" * 70 + "\n")
            print(f"Classification report saved to: {save_path}")
        
        return report
    
    def analyze_errors(self,
                      y_true: np.ndarray,
                      y_pred: np.ndarray,
                      sample_indices: Optional[np.ndarray] = None,
                      top_n: int = 10) -> Dict:
        """
        Analyze misclassified samples.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            sample_indices: Original indices of samples (optional)
            top_n: Number of top errors to return
            
        Returns:
            Dictionary with error analysis
        """
        # Convert one-hot to integer labels if needed
        if len(y_true.shape) > 1 and y_true.shape[1] > 1:
            y_true = np.argmax(y_true, axis=1)
        if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
            y_pred = np.argmax(y_pred, axis=1)
        
        # Find misclassified samples
        misclassified_mask = (y_true != y_pred)
        misclassified_indices = np.where(misclassified_mask)[0]
        
        error_analysis = {
            'total_errors': len(misclassified_indices),
            'error_rate': len(misclassified_indices) / len(y_true),
            'misclassified_indices': misclassified_indices[:top_n].tolist(),
            'confusion_pairs': []
        }
        
        # Analyze confusion pairs
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                if i != j:
                    confusion_count = np.sum((y_true == i) & (y_pred == j))
                    if confusion_count > 0:
                        error_analysis['confusion_pairs'].append({
                            'true_class': self.class_names[i],
                            'predicted_class': self.class_names[j],
                            'count': int(confusion_count)
                        })
        
        # Sort by count
        error_analysis['confusion_pairs'].sort(key=lambda x: x['count'], reverse=True)
        
        return error_analysis
    
    def print_error_analysis(self, error_analysis: Dict):
        """
        Print error analysis in a formatted way.
        
        Args:
            error_analysis: Error analysis dictionary
        """
        print("\n" + "=" * 70)
        print("ERROR ANALYSIS")
        print("=" * 70)
        print(f"Total errors: {error_analysis['total_errors']:,}")
        print(f"Error rate: {error_analysis['error_rate']:.2%}")
        
        print("\nTop Confusion Pairs:")
        for i, pair in enumerate(error_analysis['confusion_pairs'][:10], 1):
            print(f"  {i}. {pair['true_class']} → {pair['predicted_class']}: {pair['count']} errors")
        
        print("=" * 70 + "\n")


if __name__ == "__main__":
    print("Testing Model Evaluator")
    print("=" * 50)
    
    # Create dummy data
    np.random.seed(42)
    n_samples = 1000
    num_classes = 3
    class_names = ['benign', 'phishing', 'malware']
    
    y_true = np.random.randint(0, num_classes, n_samples)
    y_pred = y_true.copy()
    # Add some errors
    error_indices = np.random.choice(n_samples, size=100, replace=False)
    y_pred[error_indices] = np.random.randint(0, num_classes, 100)
    
    y_pred_proba = np.random.dirichlet(np.ones(num_classes), size=n_samples)
    
    # Test evaluator
    evaluator = ModelEvaluator(class_names)
    
    print("\n1. Computing metrics...")
    metrics = evaluator.evaluate(y_true, y_pred, y_pred_proba)
    evaluator.print_metrics(metrics)
    
    print("\n2. Generating classification report...")
    report = evaluator.generate_classification_report(y_true, y_pred)
    
    print("\n3. Analyzing errors...")
    error_analysis = evaluator.analyze_errors(y_true, y_pred)
    evaluator.print_error_analysis(error_analysis)
    
    print("\nEvaluator test completed successfully!")

