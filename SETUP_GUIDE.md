# Setup and Installation Guide

## Prerequisites

- Python 3.8 or higher
- pip package manager
- CUDA-compatible GPU (optional but recommended for training)

## Step-by-Step Installation

### 1. Install Core Dependencies

First, install the required Python packages:

```bash
# Install NumPy, Pandas, and scikit-learn
pip install numpy pandas scikit-learn

# Install PyTorch (choose one based on your setup)
# For CPU-only:
pip install torch torchvision

# For GPU (with CUDA support - check https://pytorch.org):
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install ijson pyyaml matplotlib seaborn scipy tqdm
```

Alternatively, install all at once from requirements.txt:
```bash
pip install -r requirements.txt
```

### 2. Verify Dataset

Ensure the Zenodo dataset files are present in the `Zenodo/` directory:

```bash
ls -lh Zenodo/
```

You should see:
- `phishing.json` (~2.1GB)
- `malware.json` (~1.0GB)
- `benign_umbrella.json` (~5.7GB)
- `benign_cesnet.json` (~6.0GB)

**Total dataset size**: ~14.8GB

### 3. Test the Setup

Run the setup verification script:

```bash
python test_setup.py
```

This will:
- Test data loading from large JSON files
- Verify feature extraction works correctly
- Test preprocessing pipeline
- Check PyTorch availability
- Verify evaluation utilities

### 4. Quick Test Training

Start with a small sample to verify everything works:

```bash
python train.py --config config_test.yaml
```

The test configuration (`config_test.yaml`) uses:
- **100,000 total samples** (50K benign, 30K phishing, 20K malware)
- **Smaller model** architecture for faster training
- **10 epochs** with early stopping

### 5. Full Training

Once the test run succeeds, train on the full dataset:

```bash
python train.py --config config.yaml
```

The full configuration uses:
- **10 million samples** (5M benign, 3M phishing, 2M malware)
- **Full model** architecture
- **50 epochs** with early stopping

## Monitoring Training

### TensorBoard

Monitor training in real-time:

```bash
tensorboard --logdir logs/tensorboard
```

Then open your browser to: http://localhost:6006

### Training Progress

Watch the console output for:
- Loss and accuracy per epoch
- Validation metrics
- F1 scores per class
- Learning rate adjustments
- Early stopping triggers

## After Training

### Results Location

Results are saved to `results/experiment_TIMESTAMP/`:

```bash
ls -lh results/experiment_*/
```

Key files:
- `best_model.pt` - Best model checkpoint
- `final_model.pt` - Final trained model
- `test_metrics.json` - Performance metrics
- `classification_report.txt` - Detailed report
- `confusion_matrix.png` - Confusion matrix
- `roc_curves.png` - ROC curves
- `training_history.png` - Training plots

### Making Predictions

Use the trained model for predictions:

```bash
# Interactive mode
python predict.py

# From file
python predict.py --input domains.json --output predictions.json

# Custom model path
python predict.py --model results/experiment_XXX/best_model.pt
```

## Troubleshooting

### PyTorch Installation Issues

If PyTorch installation fails:

1. **Option A**: Install from PyTorch website (recommended):
Visit https://pytorch.org/get-started/locally/ and follow instructions for your system.

2. **Option B**: Install with specific CUDA version:
```bash
# Check CUDA version
nvidia-smi

# Install matching PyTorch (example for CUDA 11.8)
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

3. **Option C**: CPU-only installation:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### Out of Memory Errors

If you encounter OOM during training:

1. **Reduce batch size** in config.yaml:
```yaml
training:
  batch_size: 256  # or 128, 64
```

2. **Reduce sample size**:
```yaml
data:
  samples_per_class:
    benign: 2000000    # Reduce from 5M
    phishing: 1500000  # Reduce from 3M
    malware: 1000000   # Reduce from 2M
```

3. **Use a smaller model**:
```yaml
model:
  embedding_dim: 128       # Reduce from 256
  conv_filters: [64, 32]   # Reduce from [128, 64]
  lstm_units: 64           # Reduce from 128
  dense_units: [64, 32]    # Reduce from [128, 64]
```

### Slow Data Loading

The first epoch may be slower due to:
- Streaming large JSON files
- Feature extraction
- Data preprocessing

Subsequent epochs will be faster as data is cached in memory.

### CUDA Errors

If you get CUDA-related errors:

1. **Verify CUDA installation**:
```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"
```

2. **Check PyTorch CUDA compatibility**:
```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

3. **Use CPU-only** if GPU issues persist:
```bash
export CUDA_VISIBLE_DEVICES=-1
python train.py --config config_test.yaml
```

## Performance Tips

### For Faster Training

1. **Use GPU**: Significantly faster than CPU
2. **Increase batch size**: If you have enough GPU memory
3. **Use fewer samples**: Start with 1M samples for experimentation

### For Better Accuracy

1. **Use more training data**: Scale up to 10M+ samples
2. **Train longer**: Increase epochs or adjust early stopping patience
3. **Tune hyperparameters**: Learning rate, dropout, architecture
4. **Handle class imbalance**: Try different balancing strategies
5. **Data augmentation**: Enable in preprocessing

## Next Steps

1. ✅ **Installation** - Install all dependencies
2. ✅ **Verification** - Run `test_setup.py`
3. ✅ **Test Training** - Run with `config_test.yaml`
4. ✅ **Full Training** - Run with `config.yaml`
5. ✅ **Evaluation** - Analyze results and metrics
6. ✅ **Deployment** - Use `predict.py` for inference

## Support

For issues or questions:
1. Check this guide and README.md
2. Review error messages in console output
3. Check PyTorch and CUDA compatibility
4. Verify dataset integrity

## Dataset Citation

Dataset from: https://zenodo.org/records/13330074

Please cite the dataset if you use it in research or publications.

