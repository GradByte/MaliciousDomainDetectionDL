# Documentation and Examples

This directory contains example results and visualizations from the malicious domain detection model.

## Images

The `images/` directory contains example training results:

- **train_class_distribution.png** - Training data class distribution
- **confusion_matrix.png** - Model confusion matrix on test set
- **roc_curves.png** - ROC curves for each class
- **training_history.png** - Training metrics over epochs
- **loss_accuracy.png** - Loss and accuracy curves

These results were generated using the test configuration (`config_test.yaml`) with 100,000 samples.

## Results

| Metric | Score |
|--------|-------|
| Accuracy | 76.96% |
| F1-Score (Macro) | 76.24% |
| ROC-AUC (Macro) | 92.06% |

These results demonstrate the model's capability on a smaller dataset. Performance typically improves with larger training datasets.

