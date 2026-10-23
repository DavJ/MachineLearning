"""Causality and geometry tests for the fixed toroidal theta-grid comparison."""
import numpy as np

from sportka.experiments.theta_grid_benchmark import toroidal_neighborhood, grid_dataset


def test_torus_translation_equivariance():
    matrix = np.arange(49, dtype=np.float32).reshape(7, 7)
    shifted = np.roll(matrix, 2, axis=0)
    a = toroidal_neighborhood(matrix)
    b = toroidal_neighborhood(shifted)
    np.testing.assert_allclose(b.reshape(7, 7), np.roll(a.reshape(7, 7), 2, 0))


def test_theta_grid_future_causality():
    from sportka.features import generate_synthetic_data
    df = generate_synthetic_data(100, seed=11)
    X, Y = grid_dataset(df)
    assert X.shape == (48, 147) and Y.shape == (48, 49)
    changed = df.copy(deep=True)
    old = list(changed.at[99, "numbers"])
    changed.at[99, "numbers"] = [1, 2, 3, 4, 5, 6] if old != [1, 2, 3, 4, 5, 6] else [7, 8, 9, 10, 11, 12]
    changed_x, changed_y = grid_dataset(changed)
    np.testing.assert_array_equal(X[-1], changed_x[-1])
    assert not np.array_equal(Y[-1], changed_y[-1])


def test_grid_preserves_last_draw():
    from sportka.features import base_features, generate_synthetic_data
    df = generate_synthetic_data(110, seed=4)
    X, Y = grid_dataset(df)
    np.testing.assert_array_equal(X[:, :49], base_features(df)[51:-1])
    np.testing.assert_array_equal(Y, base_features(df)[52:])
    assert np.isfinite(X).all()
