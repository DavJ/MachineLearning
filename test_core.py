#!/usr/bin/env python3
"""
Simple validation test for Sportka Predictor.
Tests core functionality without full data loading.
"""

import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print('=== SPORTKA PREDICTOR - CORE VALIDATION ===\n')

tests_passed = 0
tests_total = 4

# Test 1: Configuration
print('1. Configuration System...')
try:
    from sportka.config import Config
    config = Config('/tmp/test_config.json')
    config.set('training', 'default_epochs', 50)
    config.save()
    print('   ✓ PASS\n')
    tests_passed += 1
except Exception as e:
    print(f'   ✗ FAIL: {e}\n')

# Test 2: Biquaternion
print('2. Biquaternion Transformations...')
try:
    from sportka.biquaternion import apply_biquaternion_theta_transform
    numbers = [1, 7, 14, 21, 28, 35, 42]
    features = apply_biquaternion_theta_transform(numbers)
    assert features.shape == (24,), f"Expected (24,), got {features.shape}"
    print(f'   ✓ PASS (24-dimensional features)\n')
    tests_passed += 1
except Exception as e:
    print(f'   ✗ FAIL: {e}\n')

# Test 3: Neural Network
print('3. Neural Network...')
try:
    from sportka.neural_network import SportkaPredictor
    import numpy as np
    
    predictor = SportkaPredictor(
        input_dim=103,
        hidden_layers=2,
        hidden_units=32,
        model_dir='/tmp/test_models'
    )
    predictor.build_model()
    
    x_train = np.random.randn(50, 103)
    y_train = np.random.rand(50, 49)
    history = predictor.train(x_train, y_train, epochs=2, batch_size=16, verbose=0)
    
    x_test = np.random.randn(1, 103)
    pred = predictor.predict(x_test)
    best = predictor.get_best_numbers(pred, n=7)
    
    print(f'   ✓ PASS (top 3: {[n for n, _ in best[:3]]})\n')
    tests_passed += 1
except Exception as e:
    print(f'   ✗ FAIL: {e}\n')
    import traceback
    traceback.print_exc()

# Test 4: Model Persistence
print('4. Model Persistence...')
try:
    path = predictor.save_weights('validation_test')
    
    predictor2 = SportkaPredictor(model_dir='/tmp/test_models')
    predictor2.load_weights(path)
    pred2 = predictor2.predict(x_test)
    
    assert np.allclose(pred, pred2), "Loaded model predictions differ"
    print('   ✓ PASS\n')
    tests_passed += 1
except Exception as e:
    print(f'   ✗ FAIL: {e}\n')

# Summary
print('=' * 50)
print(f'RESULTS: {tests_passed}/{tests_total} tests passed')
if tests_passed == tests_total:
    print('✅ ALL CORE TESTS PASSED!')
    print('\nSystem is ready for use:')
    print('  - python sportka_launcher.py test')
    print('  - python sportka_launcher.py predict')
    print('  - python sportka_launcher.py gui')
    sys.exit(0)
else:
    print(f'⚠️  {tests_total - tests_passed} test(s) failed')
    sys.exit(1)
