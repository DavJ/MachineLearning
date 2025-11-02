# Sportka Predictor - Advanced ML System

An advanced machine learning system for Sportka lottery prediction using biquaternion transformations and optimized neural networks.

## Features

### 1. Biquaternion Transformations
- **Biquaternion representation**: Converts lottery numbers into higher-dimensional mathematical space
- **Rotation transformations**: Applies multiple rotation operators for richer feature extraction
- **Complex number encoding**: Uses complex numbers for more expressive representations

### 2. Theta-based Orthogonalization
- **Orthogonal basis creation**: Reduces correlation between features
- **Configurable rotation angles**: Customizable theta parameter for optimization
- **Enhanced feature space**: Improves neural network learning capability

### 3. Optimized Neural Network
- **M1 Mac optimization**: Automatically uses Metal Performance Shaders on Apple Silicon
- **Residual connections**: Enables deeper networks with better gradient flow
- **Batch normalization**: Improves training stability and speed
- **L2 regularization**: Prevents overfitting
- **Dropout layers**: Additional regularization technique
- **Early stopping**: Automatically stops training when validation loss stops improving
- **Learning rate reduction**: Adapts learning rate during training

### 4. Model Persistence
- **Save/Load weights**: Store trained models on disk
- **Metadata tracking**: Automatically saves model configuration and training history
- **Version management**: Multiple model versions with timestamps

### 5. User-Friendly GUI
- **Data loading**: Download from Sazka or load from file
- **Training control**: Start, stop, and monitor training progress
- **Real-time progress**: Live updates during training
- **Prediction interface**: Easy date-based predictions
- **Number selection**: Save and manage predicted numbers
- **PDF printing**: Generate printable lottery tickets

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

## Usage

### Command Line Interface

Run the improved predictor:

```bash
python -m sportka.learn_improved
```

This will:
1. Download latest data from Sazka
2. Apply biquaternion transformations
3. Train the neural network
4. Make predictions
5. Save the trained model

### Graphical User Interface

Launch the GUI:

```bash
python -m sportka.gui
```

#### GUI Workflow:

1. **Data Tab**:
   - Click "Download Latest Data from Sazka" or load from file
   - View loaded data statistics

2. **Training Tab**:
   - Configure training parameters (epochs, batch size)
   - Enable/disable biquaternion transformations
   - Start training and monitor progress
   - Save trained models for later use
   - Load previously trained models

3. **Prediction Tab**:
   - Enter prediction date
   - Click "Predict" to generate recommendations
   - View top numbers with probabilities

4. **Results Tab**:
   - Add predictions to selection
   - Save selected numbers to file
   - Generate PDF for printing

## Architecture

### Biquaternion Module (`biquaternion.py`)

The biquaternion transformation provides a sophisticated mathematical representation:

- **Biquaternion class**: w + xi + yj + zk (where w,x,y,z are complex)
- **Quaternion multiplication**: Non-commutative algebra for rotations
- **Normalization**: Unit biquaternions for stable representations
- **Feature extraction**: Converts to 8D real vector (2 components per quaternion part)

### Neural Network Module (`neural_network.py`)

Optimized architecture:

```
Input → Dense(128) → BatchNorm → Dropout
    ↓
[32 x (Dense(128) → BatchNorm → Dropout + Residual)]
    ↓
Dense(64) → Dropout
    ↓
Output(49, sigmoid)
```

Key optimizations:
- **He normal initialization**: Better for ReLU activations
- **Adam optimizer**: Adaptive learning rate
- **MSE loss**: Suitable for probability distributions
- **Residual connections**: Every 2 layers for deeper networks

### GUI Module (`gui.py`)

Multi-threaded GUI application:
- **Main thread**: UI updates and user interaction
- **Background thread**: Model training without blocking UI
- **Message queue**: Thread-safe communication
- **Tkinter**: Cross-platform GUI framework

## Technical Details

### Biquaternion Transform Pipeline

1. **Number to Biquaternion**:
   - Normalize numbers to [0,1]
   - Pack into complex components
   - Create biquaternion (w,x,y,z)

2. **Rotation Applications**:
   - Apply π/4 rotation around i-axis
   - Apply π/6 rotation around j-axis
   - Extract 8D features from each

3. **Theta Orthogonalization**:
   - Create rotation matrix with angle θ
   - Apply to feature vector
   - Produce decorrelated features

### Model Input Features

**Without biquaternion** (103 dimensions):
- 5D: Date features (day, month, year, weekday, moon phase)
- 49D: Previous draw 1 (probability distribution)
- 49D: Previous draw 2 (probability distribution)

**With biquaternion** (151 dimensions):
- 5D: Date features
- 49D: Previous draw 1
- 49D: Previous draw 2
- 24D: Biquaternion transform of draw 1
- 24D: Biquaternion transform of draw 2

### M1 Mac Optimization

The system automatically detects Apple Silicon and uses:
- **Metal Performance Shaders**: Hardware-accelerated operations
- **Unified memory**: Efficient data transfer
- **Neural Engine**: When available through TensorFlow

## Model Persistence

Models are saved with:
- **Weights file**: `models/sportka_model_YYYYMMDD_HHMMSS.h5`
- **Metadata file**: `models/sportka_model_YYYYMMDD_HHMMSS_metadata.json`

Metadata includes:
- Model architecture parameters
- Training history
- Timestamp
- Performance metrics

## Advanced Usage

### Programmatic Usage

```python
from sportka.neural_network import SportkaPredictor
from sportka.learn import draw_history

# Load data
dh = draw_history()

# Create predictor
predictor = SportkaPredictor(
    input_dim=151,  # With biquaternion
    hidden_layers=32,
    hidden_units=128
)

# Train
from sportka.neural_network import create_training_data_with_biquaternion
x_train, y_train = create_training_data_with_biquaternion(dh.draws, True)
predictor.train(x_train, y_train, epochs=100)

# Save
predictor.save_weights('my_model')

# Predict
predictions = predictor.predict(x_test)
best = predictor.get_best_numbers(predictions, n=7)
```

### Custom Transformations

```python
from sportka.biquaternion import (
    numbers_to_biquaternion,
    biquaternion_transform,
    theta_orthogonalization
)

# Convert numbers to biquaternion
bq = numbers_to_biquaternion([1, 7, 14, 21, 28, 35, 42])

# Get features
features = biquaternion_transform([1, 7, 14, 21, 28, 35, 42])

# Apply orthogonalization
ortho_features = theta_orthogonalization(features, theta=np.pi/3)
```

## Theory

### Why Biquaternions?

Biquaternions provide several advantages for lottery prediction:

1. **Higher-dimensional representation**: Maps numbers to 8D complex space
2. **Rotational properties**: Naturally encodes symmetries
3. **Non-commutativity**: Captures order-dependent patterns
4. **Mathematical elegance**: Proven framework from physics and robotics

### Why Theta Orthogonalization?

Orthogonalization reduces feature correlation:

1. **Independence**: Features become more independent
2. **Better learning**: Neural networks learn faster with decorrelated inputs
3. **Reduced redundancy**: More efficient use of network capacity
4. **Geometric interpretation**: Clear mathematical meaning

## Performance Tips

1. **Training**:
   - Start with 100 epochs, increase if needed
   - Use batch size 32 for balanced speed/accuracy
   - Enable early stopping to prevent overfitting

2. **Biquaternion**:
   - Enable for best results (more features)
   - Disable for faster training (fewer parameters)

3. **M1 Mac**:
   - Ensure TensorFlow 2.13+ for Metal support
   - Close other applications during training
   - Monitor temperature if training for long periods

## Troubleshooting

### Data Download Issues
- **Problem**: Selenium cannot find ChromeDriver
- **Solution**: Install ChromeDriver manually or use existing CSV file

### Training Slow
- **Problem**: Training takes too long
- **Solution**: Reduce epochs, increase batch size, or disable biquaternion

### Out of Memory
- **Problem**: Not enough memory during training
- **Solution**: Reduce batch size or hidden units

### Model Not Learning
- **Problem**: Loss not decreasing
- **Solution**: Increase epochs, adjust learning rate, or check data quality

## Future Improvements

Potential enhancements:
- Integration with actual unified-biquaternion-theory repository (when accessible)
- Additional theta-bot strategies
- Ensemble methods
- Online learning
- Real-time result checking
- Statistical analysis tools

## References

- Biquaternions in mathematics and physics
- Neural network optimization for lottery prediction
- M1 Mac machine learning optimization
- Quaternion algebra applications

## License

See repository license.

## Contributing

Contributions welcome! Areas of interest:
- Additional mathematical transformations
- GUI improvements
- Performance optimizations
- Documentation enhancements
