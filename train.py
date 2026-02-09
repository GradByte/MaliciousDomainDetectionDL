"""
Main training script for malicious domain detection model.
PyTorch implementation.
"""

import os
import numpy as np
import yaml
import argparse
from datetime import datetime
import sys
import torch

# Import our modules
from data.data_loader import DNSDataLoader
from data.feature_engineering import DNSFeatureExtractor
from data.preprocessor import (
    DNSDataPreprocessor, split_data, 
    prepare_data_for_cnn
)
from models.cnn_lstm_model import build_cnn_lstm_model, count_parameters
from models.model_trainer import ModelTrainer
from utils.evaluation import ModelEvaluator
from utils.visualization import (
    plot_training_history, plot_loss_and_accuracy,
    plot_class_distribution, create_evaluation_report
)


def load_config(config_path: str = 'config.yaml'):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main(args):
    """Main training pipeline."""
    
    print("\n" + "=" * 80)
    print(" " * 20 + "MALICIOUS DOMAIN DETECTION - TRAINING (PyTorch)")
    print("=" * 80)
    
    # Check PyTorch and device
    print(f"\nPyTorch version: {torch.__version__}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    
    # Load configuration
    print("\n[1/10] Loading configuration...")
    config = load_config(args.config)
    print(f"✓ Configuration loaded from: {args.config}")
    
    # Create output directories
    results_dir = config['paths'].get('results_dir', 'results')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_dir = os.path.join(results_dir, f'experiment_{timestamp}')
    os.makedirs(experiment_dir, exist_ok=True)
    print(f"✓ Experiment directory: {experiment_dir}")
    
    # Step 1: Load data
    print("\n[2/10] Loading data...")
    data_config = config['data']
    loader = DNSDataLoader(data_dir=data_config['data_dir'])
    
    # Get dataset statistics
    stats = loader.get_dataset_statistics()
    print(f"Dataset statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value:,}")
    
    # Load balanced sample
    samples_per_category = data_config['samples_per_class']
    
    # Map to file categories
    file_samples = {}
    benign_per_source = samples_per_category['benign'] // 2
    file_samples['benign_umbrella'] = benign_per_source
    file_samples['benign_cesnet'] = benign_per_source
    file_samples['phishing'] = samples_per_category['phishing']
    file_samples['malware'] = samples_per_category['malware']
    
    print(f"\nLoading samples:")
    for category, count in file_samples.items():
        print(f"  {category}: {count:,}")
    
    records, labels = loader.load_balanced_sample(
        file_samples,
        random_seed=data_config['random_seed']
    )
    
    print(f"\n✓ Loaded {len(records):,} total samples")
    print(f"✓ Label distribution:")
    unique_labels, label_counts = np.unique(labels, return_counts=True)
    for label, count in zip(unique_labels, label_counts):
        print(f"    {label}: {count:,} ({count/len(labels)*100:.1f}%)")
    
    # Step 2: Feature extraction
    print("\n[3/10] Extracting features...")
    feature_extractor = DNSFeatureExtractor(config.get('feature_engineering'))
    
    print("Extracting features from records...")
    X = feature_extractor.extract_features_batch(records)
    
    print(f"✓ Extracted features shape: {X.shape}")
    print(f"✓ Number of features: {X.shape[1]}")
    print(f"✓ Feature names: {len(feature_extractor.get_feature_names())}")
    
    # Step 3: Preprocessing
    print("\n[4/10] Preprocessing data...")
    preprocessor = DNSDataPreprocessor(config)
    
    # Fit and transform
    X_scaled, y_encoded = preprocessor.fit_transform(X, labels)
    
    print(f"✓ Scaled features shape: {X_scaled.shape}")
    print(f"✓ Encoded labels shape: {y_encoded.shape}")
    print(f"✓ Number of classes: {preprocessor.get_num_classes()}")
    print(f"✓ Class names: {preprocessor.get_class_names()}")
    
    # Save preprocessor
    preprocessor.save(
        config['paths']['feature_scaler_path'],
        config['paths']['label_encoder_path']
    )
    
    # Step 4: Split data
    print("\n[5/10] Splitting data...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        X_scaled, y_encoded,
        train_ratio=data_config['train_ratio'],
        val_ratio=data_config['val_ratio'],
        test_ratio=data_config['test_ratio'],
        random_seed=data_config['random_seed']
    )
    
    # Plot class distributions
    class_names = preprocessor.get_class_names()
    plot_class_distribution(
        y_train, class_names, 
        title="Training Set Class Distribution",
        save_path=os.path.join(experiment_dir, 'train_class_distribution.png')
    )
    
    # Step 5: Prepare data for CNN
    print("\n[6/10] Preparing data for CNN...")
    X_train = prepare_data_for_cnn(X_train)
    X_val = prepare_data_for_cnn(X_val)
    X_test = prepare_data_for_cnn(X_test)
    
    # PyTorch uses integer labels, not one-hot
    print(f"✓ Training data shape: {X_train.shape}")
    print(f"✓ Training labels shape: {y_train.shape}")
    print(f"✓ Labels are integer encoded (0-{preprocessor.get_num_classes()-1})")
    
    # Step 6: Build model
    print("\n[7/10] Building model...")
    model_config = config['model']
    model_config['input_dim'] = X_train.shape[1]
    
    model = build_cnn_lstm_model(
        input_dim=X_train.shape[1],
        num_classes=preprocessor.get_num_classes(),
        config=model_config
    )
    
    print("\nModel Architecture:")
    print(model)
    print("\n" + "=" * 80)
    trainable, non_trainable = count_parameters(model)
    print(f"Trainable parameters: {trainable:,}")
    print(f"Non-trainable parameters: {non_trainable:,}")
    print(f"Total parameters: {trainable + non_trainable:,}")
    print("=" * 80)
    
    # Step 7: Train model
    print("\n[8/10] Training model...")
    trainer = ModelTrainer(model, config, device=device)
    
    history = trainer.train(
        X_train, y_train,
        X_val, y_val,
        class_weights=preprocessor.class_weights if config['training'].get('use_class_weights') else None
    )
    
    # Save model and history
    model_path = os.path.join(experiment_dir, 'final_model.pt')
    trainer.save_model(model_path)
    
    history_path = os.path.join(experiment_dir, 'training_history.json')
    trainer.save_history(history_path)
    
    # Step 8: Evaluation
    print("\n[9/10] Evaluating model...")
    
    # Predict on test set
    print("Predicting on test set...")
    model.eval()
    
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    with torch.no_grad():
        y_test_pred_logits = model(X_test_tensor)
        y_test_pred_proba = torch.softmax(y_test_pred_logits, dim=1).cpu().numpy()
    
    y_test_pred = np.argmax(y_test_pred_proba, axis=1)
    
    # Evaluate
    evaluator = ModelEvaluator(class_names)
    
    print("\n--- Test Set Evaluation ---")
    test_metrics = evaluator.evaluate(y_test, y_test_pred, y_test_pred_proba)
    evaluator.print_metrics(test_metrics)
    
    # Save metrics
    metrics_path = os.path.join(experiment_dir, 'test_metrics.json')
    evaluator.save_metrics(test_metrics, metrics_path)
    
    # Generate classification report
    report_path = os.path.join(experiment_dir, 'classification_report.txt')
    evaluator.generate_classification_report(y_test, y_test_pred, report_path)
    
    # Plot confusion matrix
    cm_path = os.path.join(experiment_dir, 'confusion_matrix.png')
    evaluator.plot_confusion_matrix(y_test, y_test_pred, save_path=cm_path, normalize=True)
    
    # Plot ROC curves
    roc_path = os.path.join(experiment_dir, 'roc_curves.png')
    evaluator.plot_roc_curves(y_test, y_test_pred_proba, save_path=roc_path)
    
    # Error analysis
    error_analysis = evaluator.analyze_errors(y_test, y_test_pred)
    evaluator.print_error_analysis(error_analysis)
    
    # Step 9: Visualization
    print("\n[10/10] Generating visualizations...")
    
    # Training history plots
    plot_training_history(
        history,
        save_path=os.path.join(experiment_dir, 'training_history.png')
    )
    
    plot_loss_and_accuracy(
        history,
        save_path=os.path.join(experiment_dir, 'loss_accuracy.png')
    )
    
    # Prediction confidence
    from utils.visualization import plot_prediction_confidence
    plot_prediction_confidence(
        y_test_pred_proba, y_test,
        save_path=os.path.join(experiment_dir, 'prediction_confidence.png')
    )
    
    # Print best epoch metrics
    print("\n--- Best Epoch Metrics ---")
    best_metrics = trainer.get_best_epoch_metrics()
    for key, value in best_metrics.items():
        print(f"  {key}: {value}")
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print(f"\nResults saved to: {experiment_dir}")
    print(f"\nKey files:")
    print(f"  - Model: {model_path}")
    print(f"  - Best Model: saved_models/best_model.pt")
    print(f"  - Metrics: {metrics_path}")
    print(f"  - Report: {report_path}")
    print(f"  - Confusion Matrix: {cm_path}")
    print(f"  - ROC Curves: {roc_path}")
    print("\nTo view TensorBoard logs:")
    print(f"  tensorboard --logdir logs/tensorboard")
    print("=" * 80 + "\n")
    
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train malicious domain detection model (PyTorch)')
    parser.add_argument(
        '--config', 
        type=str, 
        default='config.yaml',
        help='Path to configuration file'
    )
    
    args = parser.parse_args()
    
    try:
        exit_code = main(args)
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
