# Sportka Predictor - Quick Start Guide

This guide will help you get started with the improved Sportka Predictor system.

## Installation

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

Required packages:
- `numpy` - Numerical operations
- `tensorflow>=2.13.0` - Neural network framework
- `scipy` - Scientific computing
- `pandas` - Data manipulation
- `ephem` - Astronomical calculations
- `reportlab` - PDF generation
- `h5py` - Model storage

Optional (for GUI):
- `tkinter` - Graphical interface

### Step 2: Verify Installation

```bash
python sportka_launcher.py test
```

This will check if all dependencies are installed correctly.

## Quick Start

### Method 1: Using the Launcher (Recommended)

The easiest way to use the system is through the launcher:

```bash
# Test installation
python sportka_launcher.py test

# Create configuration file
python sportka_launcher.py config

# Run prediction with command-line interface
python sportka_launcher.py predict

# Launch GUI (requires tkinter)
python sportka_launcher.py gui
```

### Method 2: Direct Python Module Usage

You can also run modules directly:

```bash
# Command-line prediction
python -m sportka.learn_improved

# GUI interface
python -m sportka.gui

# Create default configuration
python -m sportka.config
```

## Step-by-Step Tutorial

### 1. First Time Setup

```bash
# Create configuration file with default settings
python sportka_launcher.py config

# This creates config.json with default parameters
```

### 2. Download Data

The system needs historical lottery data. You can:

**Option A: Automatic Download**
```bash
python -m sportka.learn_improved
# Will automatically attempt to download from Sazka
```

**Option B: Manual Download**
1. Visit https://www.sazka.cz/loterie/sportka/statistiky
2. Download "Historie losovanych cisel" CSV
3. Save as `/tmp/sportka.csv` or update config.json with your path

### 3. Train a Model

```bash
python sportka_launcher.py predict
```

This will:
1. Load historical data
2. Apply biquaternion transformations
3. Train the neural network
4. Save the model
5. Make predictions

### 4. Make Predictions

After training, the script will automatically predict numbers for a future date.

You can customize the prediction date by editing `learn_improved.py`:
```python
PREDICTION_DATE = '15.12.2025'  # Change this date
```

### 5. Using the GUI (if tkinter is available)

```bash
python sportka_launcher.py gui
```

#### GUI Workflow:

**Step 1: Load Data**
- Go to "Data" tab
- Click "Download Latest Data from Sazka" or "Load" to use existing file
- Verify the data is loaded (you'll see statistics)

**Step 2: Configure Training**
- Go to "Training" tab
- Set epochs (100-200 recommended)
- Set batch size (32 recommended)
- Enable/disable biquaternion transformations
- Click "Start Training"
- Monitor progress in real-time

**Step 3: Save Model**
- After training completes, click "Save Model"
- Model will be saved with timestamp

**Step 4: Make Predictions**
- Go to "Prediction" tab
- Enter prediction date
- Click "Predict"
- View top 7 numbers with probabilities

**Step 5: Save Results**
- Go to "Results" tab
- Click "Add Current Prediction" to save predictions
- Select multiple predictions
- Click "Print Selected Numbers" to create PDF

## Configuration

Edit `config.json` to customize behavior:

```json
{
  "data": {
    "default_csv_path": "/tmp/sportka.csv",
    "auto_download": false
  },
  "training": {
    "default_epochs": 100,
    "default_batch_size": 32,
    "use_biquaternion": true
  },
  "model": {
    "model_dir": "./models",
    "auto_save": true
  }
}
```

### Key Configuration Options

- **use_biquaternion**: Enable advanced transformations (recommended: true)
- **default_epochs**: More epochs = better accuracy but longer training
- **default_batch_size**: 32 is good balance for most systems
- **model_dir**: Where to save trained models

## Understanding the Output

### Prediction Output Example

```
Top 7 Numbers:
------------------------------------------
  1. Number  7  0.8234  ████████████████████████████████████████
  2. Number 14  0.7891  ███████████████████████████████████████
  3. Number 21  0.7654  ██████████████████████████████████████
  ...

Recommended combination: [7, 14, 21, 28, 35, 42, 49]
```

- **Number**: The lottery number (1-49)
- **Probability**: How confident the model is (0-1)
- **Bar graph**: Visual representation of confidence
- **Recommended combination**: Top 7 numbers to play

## Advanced Usage

### Programmatic Usage

```python
from sportka.neural_network import SportkaPredictor
from sportka.learn import draw_history
from sportka.neural_network import create_training_data_with_biquaternion
import numpy as np

# Load data
dh = draw_history()

# Create training data with biquaternion
x_train, y_train = create_training_data_with_biquaternion(
    dh.draws,
    use_biquaternion=True
)

# Create and train predictor
predictor = SportkaPredictor(
    input_dim=x_train.shape[1],
    hidden_layers=32,
    hidden_units=128
)

predictor.build_model()
predictor.train(x_train, y_train, epochs=100)

# Save model
predictor.save_weights('my_custom_model')

# Load saved model later
predictor2 = SportkaPredictor()
predictor2.load_weights('./models/my_custom_model_20251102_123456.weights.h5')

# Make prediction
from sportka.learn import date_to_x
from datetime import datetime

predict_date = datetime.strptime('15.12.2025', '%d.%m.%Y').date()
x_base = date_to_x(predict_date)
# ... prepare full input with history and biquaternion features

predictions = predictor.predict(x_input)
best = predictor.get_best_numbers(predictions, n=7)
print(f"Best numbers: {[n for n, _ in best]}")
```

### Custom Biquaternion Transformations

```python
from sportka.biquaternion import (
    Biquaternion,
    numbers_to_biquaternion,
    biquaternion_transform,
    theta_orthogonalization,
    apply_biquaternion_theta_transform
)

# Convert numbers to biquaternion
numbers = [1, 7, 14, 21, 28, 35, 42]
bq = numbers_to_biquaternion(numbers)

# Get transformation features
features = biquaternion_transform(numbers)
print(f"Feature dimension: {features.shape}")  # (24,)

# Apply custom theta angle
ortho_features = theta_orthogonalization(features, theta=np.pi/6)

# Full pipeline
final_features = apply_biquaternion_theta_transform(numbers)
```

## Tips for Best Results

### Training Tips

1. **Use more data**: More historical draws = better predictions
2. **Enable biquaternion**: Provides richer features
3. **Adequate epochs**: Start with 100, increase if loss is still decreasing
4. **Save models**: Keep successful models for future use
5. **Monitor validation loss**: If it increases while training loss decreases, you're overfitting

### M1 Mac Optimization

The system is optimized for Apple Silicon:
- TensorFlow automatically uses Metal Performance Shaders
- Close other applications during training
- Monitor Activity Monitor for temperature
- Consider reducing batch size if memory issues occur

### Prediction Tips

1. **Use recent data**: Download latest data before predicting
2. **Check multiple predictions**: Generate several combinations
3. **Consider patterns**: Look at historical patterns in data
4. **Don't overtrain**: More training isn't always better

## Troubleshooting

### Problem: "ModuleNotFoundError: No module named 'tensorflow'"

**Solution**: Install dependencies
```bash
pip install tensorflow numpy scipy
```

### Problem: "No module named 'tkinter'"

**Solution**: 
- Ubuntu/Debian: `sudo apt-get install python3-tk`
- The GUI won't work, but command-line interface will
- Use: `python sportka_launcher.py predict`

### Problem: Training is very slow

**Solutions**:
1. Reduce epochs: Set to 50 for testing
2. Increase batch size: Try 64 instead of 32
3. Disable biquaternion: Faster but less accurate
4. Use CPU only: Set `TF_CPP_MIN_LOG_LEVEL=2`

### Problem: "ValueError: The filename must end in .weights.h5"

**Solution**: This is fixed in the latest version. Make sure you have the updated code.

### Problem: Model not learning (loss not decreasing)

**Solutions**:
1. Check data quality: Ensure CSV is loaded correctly
2. Increase epochs: Try 200-300
3. Adjust learning rate: Modify optimizer in neural_network.py
4. Check data preprocessing: Verify transforms are applied

### Problem: Out of memory during training

**Solutions**:
1. Reduce batch size: Try 16 instead of 32
2. Reduce hidden layers: Try 16 instead of 32
3. Reduce hidden units: Try 64 instead of 128
4. Use smaller dataset: Limit number of draws

## File Structure

```
MachineLearning/
├── sportka/
│   ├── __init__.py
│   ├── biquaternion.py          # Biquaternion transformations
│   ├── neural_network.py         # Optimized neural network
│   ├── gui.py                    # GUI application
│   ├── config.py                 # Configuration management
│   ├── learn.py                  # Original learning script
│   ├── learn_improved.py         # Improved learning script
│   ├── download.py               # Data download utilities
│   └── README.md                 # Technical documentation
├── models/                       # Saved models (created automatically)
├── predictions/                  # Saved predictions (created automatically)
├── config.json                   # Configuration file
├── requirements.txt              # Python dependencies
├── sportka_launcher.py           # Main launcher script
└── USAGE.md                      # This file
```

## Model Files

Models are saved with timestamps:
- `sportka_model_20251102_123456.weights.h5` - Model weights
- `sportka_model_20251102_123456_metadata.json` - Model metadata

Metadata includes:
- Model architecture parameters
- Training history
- Performance metrics
- Timestamp

## Next Steps

1. **Experiment with parameters**: Try different epochs, batch sizes
2. **Compare models**: Train multiple models and compare results
3. **Track performance**: Keep log of predictions vs actual results
4. **Customize features**: Modify biquaternion transformations
5. **Ensemble methods**: Combine predictions from multiple models

## Support

For issues or questions:
1. Check this USAGE guide
2. Check sportka/README.md for technical details
3. Review the code comments
4. Check GitHub issues

## Important Notes

⚠️ **Disclaimer**: This is a machine learning experiment for educational purposes. Lottery outcomes are random and cannot be reliably predicted. Use responsibly.

⚠️ **Data Privacy**: Keep your predictions private. Don't share trained models publicly as they may contain patterns from your usage.

⚠️ **Updates**: Regularly download fresh data for best results.

## Example Session

```bash
# Complete workflow example
$ python sportka_launcher.py test
✓ All components installed

$ python sportka_launcher.py config
Configuration created at ./config.json

$ python sportka_launcher.py predict
Downloading latest data...
✓ Loaded 1234 historical draws
Building model...
✓ Model built
Training...
Epoch 1/100 - loss: 0.0234
...
✓ Training completed
Model saved to: ./models/sportka_model_20251102_123456.weights.h5

Prediction for 15.12.2025:
Top 7 Numbers:
  1. Number  7  0.8234
  2. Number 14  0.7891
  ...
Recommended combination: [7, 14, 21, 28, 35, 42, 49]
```

---

**Happy Predicting! 🎰🤖**
