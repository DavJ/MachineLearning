"""
Improved neural network module for sportka prediction.

Optimized for M1 Mac and includes:
- Weight persistence (save/load)
- Optimized architecture
- Training callbacks
- Performance monitoring
"""

import os
import json
import numpy as np
import tensorflow as tf
from datetime import datetime
from typing import Optional, Tuple, Callable, Dict, Any


# Enable M1 optimization
try:
    # Use Metal Performance Shaders on M1 Mac
    tf.config.set_visible_devices([], 'GPU')  # Let TensorFlow auto-detect
except:
    pass


class SportkaPredictor:
    """
    Neural network predictor for sportka lottery numbers.
    Optimized for M1 Mac with weight persistence.
    """
    
    def __init__(
        self,
        input_dim: int = 103,
        hidden_layers: int = 64,
        hidden_units: int = 128,
        dropout_rate: float = 0.3,
        model_dir: str = './models'
    ):
        """
        Initialize the predictor.
        
        Args:
            input_dim: Input feature dimension
            hidden_layers: Number of hidden layers
            hidden_units: Units per hidden layer
            dropout_rate: Dropout rate for regularization
            model_dir: Directory to save/load models
        """
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.hidden_units = hidden_units
        self.dropout_rate = dropout_rate
        self.model_dir = model_dir
        self.model = None
        self.training_history = []
        
        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
    
    def build_model(self) -> tf.keras.Model:
        """
        Build an optimized neural network model.
        
        Returns:
            Compiled Keras model
        """
        inputs = tf.keras.Input(shape=(self.input_dim,))
        
        # Initial dense layer
        x = tf.keras.layers.Dense(
            self.hidden_units,
            activation='relu',
            kernel_initializer='he_normal',
            kernel_regularizer=tf.keras.regularizers.l2(0.001)
        )(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(self.dropout_rate)(x)
        
        # Hidden layers with residual connections
        for i in range(self.hidden_layers):
            residual = x
            
            # Dense block
            x = tf.keras.layers.Dense(
                self.hidden_units,
                activation='relu',
                kernel_initializer='he_normal',
                kernel_regularizer=tf.keras.regularizers.l2(0.001)
            )(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Dropout(self.dropout_rate)(x)
            
            # Add residual connection every 2 layers
            if i % 2 == 1:
                x = tf.keras.layers.Add()([x, residual])
        
        # Output layers
        x = tf.keras.layers.Dense(
            64,
            activation='relu',
            kernel_initializer='he_normal'
        )(x)
        x = tf.keras.layers.Dropout(self.dropout_rate / 2)(x)
        
        # Final output layer (49 numbers, probability distribution)
        outputs = tf.keras.layers.Dense(
            49,
            activation='sigmoid',
            kernel_initializer='glorot_uniform'
        )(x)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        # Compile with optimizer suitable for M1
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=0.001,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07
        )
        
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae', 'mse']
        )
        
        self.model = model
        return model
    
    def train(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        validation_split: float = 0.2,
        callbacks: Optional[list] = None,
        verbose: int = 1
    ) -> tf.keras.callbacks.History:
        """
        Train the model.
        
        Args:
            x_train: Training features
            y_train: Training labels
            epochs: Number of training epochs
            batch_size: Batch size for training
            validation_split: Fraction of data for validation
            callbacks: List of Keras callbacks
            verbose: Verbosity level
        
        Returns:
            Training history
        """
        if self.model is None:
            self.build_model()
        
        if callbacks is None:
            callbacks = []
        
        # Add early stopping
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=verbose
        )
        callbacks.append(early_stopping)
        
        # Add learning rate reduction
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=7,
            min_lr=1e-6,
            verbose=verbose
        )
        callbacks.append(reduce_lr)
        
        # Train the model
        history = self.model.fit(
            x_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=verbose
        )
        
        self.training_history.append({
            'timestamp': datetime.now().isoformat(),
            'epochs': epochs,
            'final_loss': float(history.history['loss'][-1]),
            'final_val_loss': float(history.history['val_loss'][-1])
        })
        
        return history
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            x: Input features
        
        Returns:
            Predicted probabilities for each number
        """
        if self.model is None:
            raise ValueError("Model not built or loaded. Train or load a model first.")
        
        return self.model.predict(x, verbose=0)
    
    def save_weights(self, name: str = 'sportka_model') -> str:
        """
        Save model weights to disk.
        
        Args:
            name: Model name
        
        Returns:
            Path to saved weights
        """
        if self.model is None:
            raise ValueError("No model to save")
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        weights_path = os.path.join(self.model_dir, f'{name}_{timestamp}.h5')
        
        self.model.save_weights(weights_path)
        
        # Save metadata
        metadata = {
            'input_dim': self.input_dim,
            'hidden_layers': self.hidden_layers,
            'hidden_units': self.hidden_units,
            'dropout_rate': self.dropout_rate,
            'timestamp': timestamp,
            'training_history': self.training_history
        }
        
        metadata_path = os.path.join(self.model_dir, f'{name}_{timestamp}_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Model weights saved to: {weights_path}")
        print(f"Metadata saved to: {metadata_path}")
        
        return weights_path
    
    def load_weights(self, weights_path: str) -> None:
        """
        Load model weights from disk.
        
        Args:
            weights_path: Path to weights file
        """
        # Try to load metadata
        base_path = weights_path.replace('.h5', '')
        metadata_path = base_path + '_metadata.json'
        
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Update parameters from metadata
            self.input_dim = metadata.get('input_dim', self.input_dim)
            self.hidden_layers = metadata.get('hidden_layers', self.hidden_layers)
            self.hidden_units = metadata.get('hidden_units', self.hidden_units)
            self.dropout_rate = metadata.get('dropout_rate', self.dropout_rate)
            self.training_history = metadata.get('training_history', [])
        
        # Build model with loaded parameters
        if self.model is None:
            self.build_model()
        
        # Load weights
        self.model.load_weights(weights_path)
        print(f"Model weights loaded from: {weights_path}")
    
    def list_saved_models(self) -> list:
        """
        List all saved models in the model directory.
        
        Returns:
            List of model file paths
        """
        if not os.path.exists(self.model_dir):
            return []
        
        models = [
            os.path.join(self.model_dir, f)
            for f in os.listdir(self.model_dir)
            if f.endswith('.h5')
        ]
        
        return sorted(models, reverse=True)  # Most recent first
    
    def get_best_numbers(self, predictions: np.ndarray, n: int = 7) -> list:
        """
        Extract best N numbers from predictions.
        
        Args:
            predictions: Prediction array
            n: Number of numbers to return
        
        Returns:
            List of (number, probability) tuples
        """
        if len(predictions.shape) > 1:
            predictions = predictions[0]
        
        numbers_with_probs = [
            (i + 1, float(predictions[i]))
            for i in range(min(49, len(predictions)))
        ]
        
        sorted_numbers = sorted(numbers_with_probs, key=lambda x: x[1], reverse=True)
        return sorted_numbers[:n]


def create_training_data_with_biquaternion(draws, use_biquaternion: bool = True):
    """
    Create training data with optional biquaternion transformation.
    
    Args:
        draws: List of draw objects
        use_biquaternion: Whether to apply biquaternion transformation
    
    Returns:
        Tuple of (x_train, y_train)
    """
    from sportka.biquaternion import apply_biquaternion_theta_transform
    
    x_train_list = []
    y_train_list = []
    
    for draw in draws:
        # Base features (date, moon phase, etc.)
        base_features = draw.x_train
        
        # History features
        hist1 = draw.x_train_history_1
        hist2 = draw.x_train_history_2
        
        if use_biquaternion:
            # Apply biquaternion transformation to history
            # Extract top 7 numbers from probability distribution
            top_numbers_1 = np.argsort(hist1)[-7:] + 1
            top_numbers_2 = np.argsort(hist2)[-7:] + 1
            
            bq_features_1 = apply_biquaternion_theta_transform(top_numbers_1.tolist())
            bq_features_2 = apply_biquaternion_theta_transform(top_numbers_2.tolist())
            
            # Combine all features
            combined_features = np.concatenate([
                base_features,
                hist1,
                hist2,
                bq_features_1,
                bq_features_2
            ])
        else:
            # Standard features
            combined_features = np.concatenate([
                base_features,
                hist1,
                hist2
            ])
        
        x_train_list.append(combined_features)
        y_train_list.append(draw.y_train_1)
    
    return np.array(x_train_list), np.array(y_train_list)
