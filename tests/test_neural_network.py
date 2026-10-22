"""Optional TensorFlow training and persistence contracts, on tiny real labels."""

import json

import numpy as np
import pytest


@pytest.fixture
def neural():
    tf = pytest.importorskip("tensorflow")
    pytest.importorskip("ephem")
    from sportka.neural_network import SportkaPredictor
    return tf, SportkaPredictor


def test_training_and_architecture_reload_keep_predictions(neural, tmp_path):
    tf, Predictor = neural
    rng = np.random.default_rng(1)
    x = rng.normal(size=(12, 103)).astype(np.float32)
    y = np.zeros((12, 49), dtype=np.float32)
    for row in y:
        row[rng.choice(49, 6, replace=False)] = 1
    predictor = Predictor(input_dim=103, hidden_layers=1, hidden_units=8,
                          dropout_rate=0, model_dir=str(tmp_path))
    callbacks = []
    history = predictor.train(x, y, epochs=1, batch_size=4,
                              validation_split=0.25, callbacks=callbacks, verbose=0)
    assert callbacks == []  # caller's list was not modified
    assert np.isfinite(history.history["val_loss"][-1])
    expected = predictor.predict(x[:1])
    path = predictor.save_weights("test")
    # Must rebuild an already-built instance with different dimensions.
    restored = Predictor(input_dim=151, hidden_layers=2, hidden_units=16,
                         model_dir=str(tmp_path))
    restored.build_model()
    restored.load_weights(path)
    np.testing.assert_allclose(restored.predict(x[:1]), expected, atol=1e-6)
    assert len(restored.get_best_numbers(expected)) == 6


def test_legacy_weights_require_retraining(neural, tmp_path):
    _, Predictor = neural
    weights = tmp_path / "old.weights.h5"
    (tmp_path / "old_metadata.json").write_text(json.dumps({"input_dim": 103}))
    with pytest.raises(ValueError, match="Legacy weights"):
        Predictor(model_dir=str(tmp_path)).load_weights(str(weights))


def test_training_features_match_prior_main_numbers(neural, tmp_path):
    from sportka.data import draw_history
    from sportka.neural_network import create_training_data_with_biquaternion
    from sportka.biquaternion import apply_biquaternion_theta_transform
    from tests.test_sportka_data import HEADER, ROW
    path = tmp_path / "history.csv"
    path.write_text(HEADER + ROW + ROW.replace("1. 1. 2025", "8. 1. 2025"))
    history = draw_history(path)
    x, y = create_training_data_with_biquaternion(history.draws)
    assert x.shape == (2, 151)
    assert y.shape == (2, 49)
    np.testing.assert_array_equal(y.sum(axis=1), 6)
    np.testing.assert_allclose(x[1, -48:-24],
                               apply_biquaternion_theta_transform([1, 2, 3, 4, 5, 6]))
    np.testing.assert_array_equal(x[0, -48:], 0)
