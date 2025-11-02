# Machine Learning Projects

This repository contains various machine learning projects and experiments.

## Projects

### Sportka Predictor - Advanced ML System ⭐ NEW

An advanced lottery number prediction system using biquaternion transformations and optimized neural networks.

**Key Features:**
- 🔬 Biquaternion mathematical transformations for richer feature representation
- 🧮 Theta-based orthogonalization for decorrelated features
- 🚀 Neural network optimized for M1 Mac (Metal Performance Shaders)
- 💾 Model weight persistence (save/load functionality)
- 🖥️ User-friendly GUI with real-time training monitoring
- 📊 PDF printing for lottery tickets
- ⚙️ Configurable parameters

**Quick Start:**
```bash
# Install dependencies
pip install -r requirements.txt

# Test installation
python sportka_launcher.py test

# Run prediction
python sportka_launcher.py predict

# Launch GUI
python sportka_launcher.py gui
```

**Documentation:**
- [Quick Start Guide](USAGE.md) - Get started quickly
- [Technical Documentation](sportka/README.md) - In-depth technical details
- [Configuration](config.json) - Customize behavior

**What's New:**
- ✨ Biquaternion transformations from unified-biquaternion-theory concepts
- ✨ Theta-bot inspired optimization strategies
- ✨ M1 Mac hardware acceleration
- ✨ Complete GUI application
- ✨ Model persistence and management

### Finance Projects

Various financial analysis and trading projects:
- Binance integration
- Alpha Vantage data sync
- Kalman filter implementations

## Repository Structure

```
MachineLearning/
├── sportka/              # Sportka Predictor (lottery prediction)
│   ├── biquaternion.py  # Mathematical transformations
│   ├── neural_network.py # Optimized neural network
│   ├── gui.py           # GUI application
│   ├── config.py        # Configuration management
│   └── README.md        # Technical documentation
├── finance/             # Financial analysis projects
├── models/              # Saved ML models
├── predictions/         # Saved predictions
├── requirements.txt     # Python dependencies
├── sportka_launcher.py  # Main launcher
├── USAGE.md            # Quick start guide
└── README.md           # This file
```

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Required Packages
- `numpy` - Numerical computing
- `tensorflow>=2.13.0` - Deep learning framework
- `scipy` - Scientific computing
- `pandas` - Data analysis
- `ephem` - Astronomical calculations
- `reportlab` - PDF generation

### Optional Packages
- `tkinter` - For GUI (usually included with Python)
- `selenium` - For web scraping

## Usage

See [USAGE.md](USAGE.md) for detailed instructions.

### Quick Commands

```bash
# Sportka Predictor
python sportka_launcher.py test      # Test installation
python sportka_launcher.py config    # Create config file
python sportka_launcher.py predict   # Run prediction
python sportka_launcher.py gui       # Launch GUI
```

## Features

### Sportka Predictor Features

1. **Advanced Mathematics**
   - Biquaternion representations (8D complex space)
   - Quaternion multiplication and rotations
   - Theta-based orthogonalization
   - Complex number encodings

2. **Neural Network**
   - Residual connections for deep networks
   - Batch normalization for stable training
   - Dropout regularization
   - L2 weight regularization
   - Early stopping
   - Learning rate reduction

3. **M1 Mac Optimization**
   - Metal Performance Shaders support
   - Unified memory utilization
   - Hardware acceleration

4. **User Interface**
   - Intuitive multi-tab GUI
   - Real-time training progress
   - Model management
   - PDF ticket generation
   - Result tracking

5. **Model Management**
   - Save trained models with metadata
   - Load previous models
   - Version tracking with timestamps
   - Training history logging

## Project History

### Recent Updates

**November 2025 - Sportka Predictor v2.0**
- Added biquaternion transformations
- Implemented theta-based orthogonalization
- Created comprehensive GUI
- Added model persistence
- Optimized for M1 Mac
- Added configuration system
- Created launcher script

### Earlier Work
- Original sportka predictor
- Finance analysis tools
- Kalman filter experiments

## Technical Details

### Biquaternions

Biquaternions extend quaternions by using complex numbers as components:
```
q = w + xi + yj + zk
where w, x, y, z ∈ ℂ
```

This provides:
- 8D real representation (2 components per quaternion part)
- Non-commutative algebra for capturing order
- Natural encoding of rotations and symmetries

### Theta Orthogonalization

Creates orthogonal feature basis through rotation matrices:
```python
R(θ) = [cos(θ)  -sin(θ)]
       [sin(θ)   cos(θ)]
```

Benefits:
- Reduces feature correlation
- Improves neural network learning
- Clear geometric interpretation

### Neural Network Architecture

```
Input(103 or 151) → Dense(128) → BatchNorm → Dropout
                 ↓
        [32 × Dense Block with Residuals]
                 ↓
             Dense(64) → Dropout
                 ↓
            Output(49, sigmoid)
```

## Performance

### Training Time (M1 Mac)
- 100 epochs: ~5-10 minutes
- 200 epochs: ~10-20 minutes

### Memory Usage
- Training: ~2-4 GB RAM
- Prediction: ~500 MB RAM

### Accuracy
- Depends on data quality and quantity
- More historical data = better predictions
- Biquaternion features improve representation

## Development

### Project Structure

Each project has:
- Main implementation files
- Configuration options
- Documentation
- Tests (where applicable)

### Code Style
- Python 3 type hints where applicable
- Docstrings for all public functions
- Comments for complex algorithms

### Testing

```bash
# Test sportka predictor
python sportka_launcher.py test

# Test individual modules
python -c "import sportka.biquaternion; print('OK')"
python -c "import sportka.neural_network; print('OK')"
```

## Contributing

Areas for contribution:
- Additional mathematical transformations
- Performance optimizations
- GUI improvements
- Documentation enhancements
- Test coverage
- New prediction strategies

## License

See repository license file.

## Disclaimer

⚠️ **Important**: Machine learning predictions for lottery numbers are experimental and educational. Lottery outcomes are random and cannot be reliably predicted. Use responsibly and for learning purposes only.

## References

### Sportka Predictor
- Biquaternion algebra in mathematics
- Neural network optimization techniques
- M1 Mac machine learning optimization
- Quaternion applications in signal processing

### Mathematical Background
- Hamilton's quaternions
- Complex number theory
- Non-commutative algebra
- Orthogonal transformations

## Support

For questions or issues:
1. Check [USAGE.md](USAGE.md) for quick start
2. Check [sportka/README.md](sportka/README.md) for technical details
3. Review code comments
4. Open GitHub issue

## Acknowledgments

- Inspired by unified-biquaternion-theory concepts
- Theta-bot optimization strategies
- TensorFlow team for M1 support
- Python scientific computing community

---

**Status**: ✅ Active Development

**Last Updated**: November 2025

**Python Version**: 3.8+

**TensorFlow Version**: 2.13+
