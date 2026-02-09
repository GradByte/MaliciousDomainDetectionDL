# Malicious Domain Detection using Deep Learning

A deep learning system for detecting malicious domains (phishing and malware) using DNS features. Built with **PyTorch** and trained on the Zenodo DNS dataset.

## Overview

This project implements a hybrid CNN-LSTM neural network to classify domains into three categories:
- **Benign**: Legitimate, safe domains
- **Phishing**: Domains used for phishing attacks
- **Malware**: Domains associated with malware distribution

## Features

- **Efficient Data Loading**: Streaming JSON parser for handling large datasets (14.8GB+)
- **Rich Feature Engineering**: Extracts 50+ features from DNS records including:
  - Domain-based features (length, entropy, special characters)
  - DNS record counts (A, AAAA, MX, NS, TXT)
  - IP-based features (diversity, geographic distribution)
  - TTL statistics
  - DNSSEC features
  - Nameserver and mail server patterns
- **Hybrid CNN-LSTM Architecture**: Combines pattern detection with sequential learning
- **Attention Mechanism**: Focuses on most important features
- **Class Imbalance Handling**: Weighted loss and sampling strategies
- **Comprehensive Evaluation**: Accuracy, precision, recall, F1-score, ROC-AUC, confusion matrix
- **GPU Acceleration**: Optimized for CUDA-enabled GPUs

## Project Structure

```
dnsProject/
├── data/
│   ├── data_loader.py          # Efficient data loading from large JSON files
│   ├── feature_engineering.py  # Extract features from DNS records
│   └── preprocessor.py         # Data preprocessing and normalization
├── models/
│   ├── cnn_lstm_model.py       # Hybrid CNN-LSTM architecture
│   └── model_trainer.py        # Training loop with callbacks
├── utils/
│   ├── evaluation.py           # Metrics, confusion matrix, reports
│   └── visualization.py        # Training plots and analysis
├── Zenodo/                     # Dataset directory
│   ├── phishing.json
│   ├── malware.json
│   ├── benign_umbrella.json
│   └── benign_cesnet.json
├── train.py                    # Main training script
├── predict.py                  # Inference script
├── config.yaml                 # Hyperparameters and settings
└── requirements.txt            # Python dependencies
```

## Installation

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Verify dataset:**
Ensure the Zenodo dataset files are in the `Zenodo/` directory:
- `phishing.json` (~2.1GB)
- `malware.json` (~1.0GB)
- `benign_umbrella.json` (~5.7GB)
- `benign_cesnet.json` (~6.0GB)

## Usage

### Training

Train the model with default configuration:
```bash
python train.py
```

Train with custom configuration:
```bash
python train.py --config custom_config.yaml
```

### Configuration

Edit `config.yaml` to customize:
- Data sampling strategy
- Model architecture parameters
- Training hyperparameters
- Evaluation settings

Key parameters:
```yaml
data:
  sample_size: 10000000  # Total samples to load
  samples_per_class:
    benign: 5000000
    phishing: 3000000
    malware: 2000000

model:
  conv_filters: [128, 64]
  lstm_units: 128
  dense_units: [128, 64]
  dropout_rate: 0.3

training:
  batch_size: 512
  epochs: 50
  learning_rate: 0.001
  early_stopping_patience: 10
```

### Prediction

Predict on new domains from a JSON file:
```bash
python predict.py --input domains.json --output predictions.json
```

Interactive prediction mode:
```bash
python predict.py
```

## Model Architecture

```
Input (50 features)
    ↓
Dense Embedding (256) + BatchNorm + Dropout
    ↓
1D Conv (128 filters) + ReLU + BatchNorm + Dropout
    ↓
1D Conv (64 filters) + ReLU + BatchNorm + Dropout
    ↓
MaxPooling1D
    ↓
BiLSTM (128 units) + Dropout
    ↓
Attention Layer (64 units)
    ↓
Dense (128) + ReLU + Dropout
    ↓
Dense (64) + ReLU + Dropout
    ↓
Output (3 classes, softmax)
```

## Performance

The model is designed to achieve high accuracy through:
- Rich DNS feature extraction (50+ features)
- Hybrid CNN-LSTM architecture with attention
- Comprehensive training on large-scale dataset

## Results

After training, results are saved to `results/experiment_TIMESTAMP/`:
- `best_model.pt` - Best model checkpoint (PyTorch)
- `final_model.pt` - Final trained model (PyTorch)
- `training_history.json` - Training metrics
- `test_metrics.json` - Test set performance
- `classification_report.txt` - Detailed classification report
- `confusion_matrix.png` - Confusion matrix visualization
- `roc_curves.png` - ROC curves for each class
- `training_history.png` - Training curves
- `loss_accuracy.png` - Loss and accuracy plots

## Monitoring Training

Monitor training progress with TensorBoard:
```bash
tensorboard --logdir logs/tensorboard
```

## Dataset

This project uses the DNS dataset from:
**Zenodo**: https://zenodo.org/records/13330074

The dataset contains:
- ~248M benign domains (CESNET)
- ~224M benign domains (Umbrella)
- ~85M phishing domains
- ~43M malware domains

Each record includes comprehensive DNS information:
- A, AAAA, CNAME, MX, NS, TXT, SOA records
- DNSSEC validation status
- IP addresses and TTL values
- Nameserver and mail server details

## Requirements

- Python 3.8+
- **PyTorch 2.0+**
- NumPy, Pandas, Scikit-learn
- Matplotlib, Seaborn
- ijson (for streaming large JSON files)
- CUDA-compatible GPU (recommended)

## Tips for Training

1. **Start Small**: Begin with smaller samples to verify the pipeline
2. **Monitor GPU Memory**: Adjust `batch_size` if you encounter OOM errors
3. **Class Imbalance**: The `weighted` strategy works well for this dataset
4. **Early Stopping**: Prevents overfitting

## Troubleshooting

**Out of Memory Error:**
- Reduce `batch_size` in config.yaml
- Reduce `sample_size` to load fewer samples
- Use a smaller model architecture

**Slow Data Loading:**
- The first epoch may be slower due to data loading
- Consider preprocessing and caching features

**Model Not Improving:**
- Check class weights are properly applied
- Try different learning rates (0.0001 - 0.01)
- Adjust model architecture complexity

## Future Improvements

- Real-time domain monitoring
- Feature importance analysis
- Ensemble models
- Transfer learning from pretrained models
- API endpoint for production deployment
- Integration with threat intelligence feeds

## License

This project is for educational and research purposes.

## Citation

If you use this code or dataset, please cite:
- Dataset: Zenodo DNS Dataset (https://zenodo.org/records/13330074)

