"""
Improved Sportka Predictor with Biquaternion Transformations

This script demonstrates the improved prediction system with:
- Biquaternion transformations
- Theta-based orthogonalization
- Optimized neural network
- Model persistence
"""

import numpy as np
from datetime import datetime
import os
import sys

from sportka.download import download_data_from_sazka
from sportka.learn import draw_history, date_to_x
from sportka.neural_network import SportkaPredictor, create_training_data_with_biquaternion
from sportka.biquaternion import apply_biquaternion_theta_transform


def print_separator(title=""):
    """Print a formatted separator."""
    print("\n" + "=" * 70)
    if title:
        print(f" {title}")
        print("=" * 70)
    print()


def main():
    """Main execution function."""
    print_separator("SPORTKA PREDICTOR - Advanced ML System")
    print("Features:")
    print("  - Biquaternion transformations")
    print("  - Theta-based orthogonalization")
    print("  - Neural network optimized for M1 Mac")
    print("  - Model weight persistence")
    print()
    
    # Configuration
    PREDICTION_DATE = '15.12.2025'
    USE_BIQUATERNION = True
    EPOCHS = 150
    BATCH_SIZE = 32
    MODEL_NAME = 'sportka_model'
    
    print(f"Configuration:")
    print(f"  Prediction Date: {PREDICTION_DATE}")
    print(f"  Use Biquaternion: {USE_BIQUATERNION}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Batch Size: {BATCH_SIZE}")
    
    # Step 1: Load data
    print_separator("Step 1: Loading Data")
    
    try:
        print("Downloading latest data from Sazka...")
        download_data_from_sazka()
        print("✓ Data downloaded successfully")
    except Exception as e:
        print(f"⚠ Warning: Could not download data: {e}")
        print("Attempting to use existing data...")
    
    try:
        dh = draw_history()
        print(f"✓ Loaded {len(dh.draws)} historical draws")
        print(f"  Date range: {dh.draws[0].date} to {dh.draws[-1].date}")
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        return 1
    
    # Step 2: Prepare training data
    print_separator("Step 2: Preparing Training Data")
    
    try:
        print("Creating training dataset...")
        if USE_BIQUATERNION:
            print("  - Applying biquaternion transformations")
            print("  - Applying theta orthogonalization")
        
        x_train, y_train = create_training_data_with_biquaternion(
            dh.draws,
            use_biquaternion=USE_BIQUATERNION
        )
        
        print(f"✓ Training data prepared")
        print(f"  Input shape: {x_train.shape}")
        print(f"  Output shape: {y_train.shape}")
        
        # Show sample transformation
        if USE_BIQUATERNION and len(dh.draws) > 0:
            sample_numbers = dh.draws[-1].first
            print(f"\n  Sample transformation:")
            print(f"    Original numbers: {sample_numbers}")
            transformed = apply_biquaternion_theta_transform(sample_numbers)
            print(f"    Transformed features: {len(transformed)} dimensions")
        
    except Exception as e:
        print(f"✗ Error preparing data: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Step 3: Build and train model
    print_separator("Step 3: Building and Training Model")
    
    try:
        # Determine input dimension
        input_dim = x_train.shape[1]
        print(f"Input dimension: {input_dim}")
        
        # Create predictor
        predictor = SportkaPredictor(
            input_dim=input_dim,
            hidden_layers=32,
            hidden_units=128,
            dropout_rate=0.3,
            model_dir='./models'
        )
        
        print("Building neural network...")
        predictor.build_model()
        print(f"✓ Model built")
        print(f"  Hidden layers: 32")
        print(f"  Hidden units: 128")
        print(f"  Dropout rate: 0.3")
        
        # Check if we should load existing model
        existing_models = predictor.list_saved_models()
        load_existing = False
        
        if existing_models and os.getenv('INTERACTIVE', '1') == '1':
            print(f"\n  Found {len(existing_models)} saved model(s)")
            print(f"  Latest: {os.path.basename(existing_models[0])}")
            
            response = input("\n  Load existing model? (y/n): ").lower()
            load_existing = (response == 'y')
        
        if load_existing:
            predictor.load_weights(existing_models[0])
            print("✓ Model loaded from disk")
            skip_training = True
        else:
            skip_training = False
        
        if not skip_training:
            print("\nStarting training...")
            print("  (This may take several minutes)")
            print()
            
            history = predictor.train(
                x_train, y_train,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                validation_split=0.2,
                verbose=1
            )
            
            print("\n✓ Training completed")
            print(f"  Final loss: {history.history['loss'][-1]:.4f}")
            print(f"  Final val_loss: {history.history['val_loss'][-1]:.4f}")
            
            # Save model
            print("\nSaving model...")
            weights_path = predictor.save_weights(MODEL_NAME)
            print(f"✓ Model saved to: {weights_path}")
        
    except Exception as e:
        print(f"✗ Error during training: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Step 4: Make predictions
    print_separator("Step 4: Making Predictions")
    
    try:
        predict_date = datetime.strptime(PREDICTION_DATE, '%d.%m.%Y').date()
        
        # Prepare prediction input
        x_predict_base = date_to_x(predict_date)
        x_hist1 = dh.draws[-1].y_train_1
        x_hist2 = dh.draws[-1].y_train_2
        
        if USE_BIQUATERNION:
            # Apply biquaternion transformation
            top_nums_1 = np.argsort(x_hist1)[-7:] + 1
            top_nums_2 = np.argsort(x_hist2)[-7:] + 1
            
            bq1 = apply_biquaternion_theta_transform(top_nums_1.tolist())
            bq2 = apply_biquaternion_theta_transform(top_nums_2.tolist())
            
            x_predict_full = np.concatenate([x_predict_base, x_hist1, x_hist2, bq1, bq2])
        else:
            x_predict_full = np.concatenate([x_predict_base, x_hist1, x_hist2])
        
        x_predict = np.array([x_predict_full])
        
        # Make prediction
        print(f"Predicting for date: {PREDICTION_DATE}")
        predictions = predictor.predict(x_predict)
        best_numbers = predictor.get_best_numbers(predictions, n=7)
        
        print("\n✓ Prediction completed")
        print("\nTop 7 Numbers:")
        print("-" * 40)
        for i, (num, prob) in enumerate(best_numbers, 1):
            bars = "█" * int(prob * 50)
            print(f"  {i}. Number {num:2d}  {prob:.4f}  {bars}")
        
        print("\n" + "-" * 40)
        print(f"Recommended combination: {[n for n, _ in best_numbers]}")
        print("-" * 40)
        
        # Alternative combinations
        print("\nAlternative combinations (top 12 numbers):")
        top_12 = predictor.get_best_numbers(predictions, n=12)
        top_12_numbers = [n for n, _ in top_12]
        
        import random
        random.seed(42)
        for i in range(3):
            combination = sorted(random.sample(top_12_numbers, 7))
            print(f"  {i+1}. {combination}")
        
    except Exception as e:
        print(f"✗ Error during prediction: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print_separator("Prediction Complete")
    print("To use the GUI interface, run:")
    print("  python -m sportka.gui")
    print()
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
