"""
Test script to verify the setup and components work correctly.
Tests data loading, feature extraction, and preprocessing without training.
"""

import numpy as np
import sys
from datetime import datetime

print("\n" + "=" * 80)
print(" " * 25 + "SETUP VERIFICATION TEST")
print("=" * 80)

# Test 1: Data Loader
print("\n[Test 1/5] Testing Data Loader...")
try:
    from data.data_loader import DNSDataLoader
    
    loader = DNSDataLoader("Zenodo")
    print("✓ Data loader imported successfully")
    
    # Test loading a small sample
    print("  Loading small sample (100 records per category)...")
    samples = loader.load_sample_for_exploration(n_samples=100)
    
    total_samples = sum(len(records) for records in samples.values())
    print(f"✓ Loaded {total_samples} sample records")
    
    for category, records in samples.items():
        if records:
            print(f"  - {category}: {len(records)} records")
            print(f"    Sample domain: {records[0].get('domain_name', 'N/A')}")
    
    print("✓ Data Loader: PASSED")
    
except Exception as e:
    print(f"✗ Data Loader: FAILED - {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 2: Feature Engineering
print("\n[Test 2/5] Testing Feature Engineering...")
try:
    from data.feature_engineering import DNSFeatureExtractor
    
    extractor = DNSFeatureExtractor()
    print("✓ Feature extractor imported successfully")
    
    # Get a sample record
    sample_record = None
    for records in samples.values():
        if records:
            sample_record = records[0]
            break
    
    if sample_record:
        features = extractor.extract_features(sample_record)
        print(f"✓ Extracted {len(features)} features from sample record")
        print(f"  Feature shape: {features.shape}")
        print(f"  Feature range: [{features.min():.2f}, {features.max():.2f}]")
        print(f"  Feature names: {len(extractor.get_feature_names())}")
    
    # Test batch extraction
    all_records = []
    for records in samples.values():
        all_records.extend(records[:10])  # Take 10 from each
    
    if all_records:
        print(f"  Extracting features from {len(all_records)} records...")
        features_batch = extractor.extract_features_batch(all_records)
        print(f"✓ Batch extraction shape: {features_batch.shape}")
    
    print("✓ Feature Engineering: PASSED")
    
except Exception as e:
    print(f"✗ Feature Engineering: FAILED - {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Preprocessing
print("\n[Test 3/5] Testing Preprocessing...")
try:
    from data.preprocessor import DNSDataPreprocessor, split_data
    
    preprocessor = DNSDataPreprocessor()
    print("✓ Preprocessor imported successfully")
    
    # Create dummy data for testing
    X = np.random.randn(1000, 50)
    y = np.random.choice(['benign', 'phishing', 'malware'], size=1000)
    
    print(f"  Test data shape: {X.shape}")
    print(f"  Test labels shape: {y.shape}")
    
    # Fit and transform
    X_scaled, y_encoded = preprocessor.fit_transform(X, y)
    print(f"✓ Preprocessed data shape: {X_scaled.shape}")
    print(f"✓ Encoded labels shape: {y_encoded.shape}")
    print(f"  Classes: {preprocessor.get_class_names()}")
    print(f"  Class weights: {preprocessor.class_weights}")
    
    # Test splitting
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        X_scaled, y_encoded,
        train_ratio=0.7, val_ratio=0.15, test_ratio=0.15
    )
    print(f"✓ Data split successful")
    
    print("✓ Preprocessing: PASSED")
    
except Exception as e:
    print(f"✗ Preprocessing: FAILED - {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Check PyTorch availability
print("\n[Test 4/5] Checking PyTorch availability...")
try:
    import torch
    print(f"✓ PyTorch version: {torch.__version__}")
    
    # Check GPU
    if torch.cuda.is_available():
        print(f"✓ GPU available: {torch.cuda.device_count()} device(s)")
        for i in range(torch.cuda.device_count()):
            print(f"  - GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  CUDA version: {torch.version.cuda}")
    else:
        print("  ⚠ No GPU detected, will use CPU")
    
    # Test model import
    from models.cnn_lstm_model import build_cnn_lstm_model, count_parameters
    print("✓ Model modules imported successfully")
    
    # Build a small test model
    test_model = build_cnn_lstm_model(
        input_dim=50,
        num_classes=3,
        config={'embedding_dim': 32, 'conv_filters': [16], 'lstm_units': 16}
    )
    print(f"✓ Test model built successfully")
    
    # Count parameters
    trainable, non_trainable = count_parameters(test_model)
    print(f"  Model parameters: {trainable:,}")
    
    print("✓ PyTorch: PASSED")
    
except ImportError as e:
    print(f"✗ PyTorch: NOT INSTALLED - {e}")
    print("  Note: PyTorch is required for training.")
    print("  Install with: pip install torch torchvision")
    print("  Or visit: https://pytorch.org/get-started/locally/")
    # Don't exit, continue with other tests
except Exception as e:
    print(f"✗ PyTorch: ERROR - {e}")
    import traceback
    traceback.print_exc()

# Test 5: Evaluation and Visualization
print("\n[Test 5/5] Testing Evaluation and Visualization...")
try:
    from utils.evaluation import ModelEvaluator
    from utils.visualization import plot_class_distribution
    
    print("✓ Evaluation and visualization modules imported successfully")
    
    # Test evaluator
    class_names = ['benign', 'phishing', 'malware']
    evaluator = ModelEvaluator(class_names)
    
    # Create dummy predictions
    y_true = np.random.randint(0, 3, 100)
    y_pred = y_true.copy()
    y_pred[:10] = (y_pred[:10] + 1) % 3  # Add some errors
    y_pred_proba = np.random.dirichlet(np.ones(3), size=100)
    
    metrics = evaluator.evaluate(y_true, y_pred, y_pred_proba)
    print(f"✓ Computed {len(metrics)} evaluation metrics")
    print(f"  Sample metrics: accuracy={metrics['accuracy']:.3f}, f1={metrics['f1_macro']:.3f}")
    
    print("✓ Evaluation & Visualization: PASSED")
    
except Exception as e:
    print(f"✗ Evaluation & Visualization: FAILED - {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Summary
print("\n" + "=" * 80)
print(" " * 30 + "TEST SUMMARY")
print("=" * 80)
print("\n✓ All core components verified successfully!")
print("\nNext steps:")
print("  1. Install PyTorch if not already installed:")
print("     pip install torch torchvision")
print("     Or visit: https://pytorch.org/get-started/locally/")
print("\n  2. Run training with test configuration:")
print("     python train.py --config config_test.yaml")
print("\n  3. For full training, use the main configuration:")
print("     python train.py --config config.yaml")
print("\n" + "=" * 80 + "\n")

