"""Small deterministic tests for number theory and past-only features."""
import numpy as np
import pandas as pd
import pytest

from sportka.experiments.theta_mobius import (
    mobius, spectral, dataset, cache_key, train_or_load, training_signature,
)


def test_mobius_values():
    assert [mobius(n) for n in range(1, 11)] == [1, -1, -1, 0, -1, 1, -1, 0, 0, 1]


def test_spectral_is_finite():
    a = np.array([0., .125, .5, 1.])
    for kind in ("theta", "mobius", "mertens"):
        result = spectral(a, kind)
        assert result.shape == a.shape
        assert np.isfinite(result).all()


def test_no_future_features():
    seq = [list(range(1 + i % 43, 7 + i % 43)) for i in range(80)]
    df = pd.DataFrame({"numbers": seq})
    x1, y1 = dataset(df, "mobius")
    df2 = df.copy()
    modified = list(seq)
    modified[-1] = [1, 2, 3, 4, 5, 49]
    df2["numbers"] = modified
    x2, y2 = dataset(df2, "mobius")
    np.testing.assert_array_equal(x1[-1], x2[-1])
    assert not np.array_equal(y1[-1], y2[-1])


def test_checkpoint_signature():
    a = cache_key(b"data", 1, "theta", 20)
    assert a == cache_key(b"data", 1, "theta", 20)
    assert a != cache_key(b"more data", 1, "theta", 20)
    assert a != cache_key(b"data", 2, "theta", 20)
    assert a != cache_key(b"data", 1, "theta", 20, hidden=64)
    assert a != cache_key(b"data", 1, "theta", 20, max_iter=151)
    assert a != cache_key(b"data", 1, "theta", 20, seed=43)


def test_feature_variants_have_same_dimensions_targets_and_last_draw():
    from sportka.features import generate_synthetic_data
    df = generate_synthetic_data(100)
    raw, targets = dataset(df, "raw")
    assert raw.shape == (48, 147)
    for kind in ("theta", "mobius", "mertens"):
        transformed, y = dataset(df, kind)
        assert transformed.shape == raw.shape
        np.testing.assert_array_equal(y, targets)
        # Preserve the binary past draw; integer cosine transforms would erase it.
        np.testing.assert_array_equal(transformed[:, :49], raw[:, :49])
        assert not np.allclose(transformed[:, 49:], raw[:, 49:])


def test_short_history_has_consistent_empty_shapes():
    from sportka.features import generate_synthetic_data
    x, y = dataset(generate_synthetic_data(20), "raw")
    assert x.shape == (0, 147)
    assert y.shape == (0, 49)


@pytest.fixture
def training_arrays():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(20, 5)).astype(np.float32)
    y = np.zeros((20, 49), dtype=np.float32)
    for row in y:
        row[rng.choice(49, size=6, replace=False)] = 1
    return x, y


def test_cache_reuses_identical_model_without_fitting(training_arrays, tmp_path, monkeypatch):
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.neural_network import MLPClassifier
    x, y = training_arrays
    options = dict(key="identical", cache_dir=tmp_path, kind="theta", hidden=4, max_iter=2)
    with pytest.warns(ConvergenceWarning):
        model, reused = train_or_load(x, y, **options)
    assert not reused
    expected = model.predict_proba(x)
    def unexpected_fit(*args, **kwargs):
        raise AssertionError("A matching checkpoint must not train again")
    monkeypatch.setattr(MLPClassifier, "fit", unexpected_fit)
    restored, reused = train_or_load(x, y, **options)
    assert reused
    np.testing.assert_array_equal(restored.predict_proba(x), expected)


@pytest.mark.parametrize("change", ["features", "labels", "configuration"])
def test_reusing_wrong_cache_signature_fails(training_arrays, tmp_path, change):
    from sklearn.exceptions import ConvergenceWarning
    x, y = training_arrays
    options = dict(key="same-key", cache_dir=tmp_path, kind="theta", hidden=4, max_iter=2)
    with pytest.warns(ConvergenceWarning):
        train_or_load(x, y, **options)
    x, y = x.copy(), y.copy()
    if change == "features":
        x[0, 0] += 1
    elif change == "labels":
        y[0] = np.roll(y[0], 1)
    else:
        options["hidden"] = 8
    with pytest.raises(ValueError, match="signature mismatch"):
        train_or_load(x, y, **options)


def test_failing_checkpoint_write_keeps_existing_model(training_arrays, tmp_path, monkeypatch):
    import joblib
    from sklearn.exceptions import ConvergenceWarning
    x, y = training_arrays
    options = dict(key="existing", cache_dir=tmp_path, kind="raw", hidden=4, max_iter=2)
    with pytest.warns(ConvergenceWarning):
        train_or_load(x, y, **options)
    path = tmp_path / "raw-existing.joblib"
    before = path.read_bytes()
    def broken_dump(*args, **kwargs):
        raise OSError("Disk full")
    monkeypatch.setattr(joblib, "dump", broken_dump)
    with pytest.warns(ConvergenceWarning), pytest.raises(OSError, match="Disk full"):
        train_or_load(x, y, fresh=True, **options)
    assert path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]


def test_training_signature_includes_shape_and_values(training_arrays):
    x, y = training_arrays
    assert training_signature(x, y) == training_signature(x.copy(), y.copy())
    assert training_signature(x, y) != training_signature(x.reshape(5, 20), y)


def test_run_fits_training_only_and_reuses_all_four_checkpoints(tmp_path, monkeypatch):
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.neural_network import MLPClassifier
    from sportka.features import generate_synthetic_data, base_features
    from sportka.experiments import theta_mobius as experiment
    df = generate_synthetic_data(200, seed=19)
    source = tmp_path / "source.csv"
    source.write_text("source identity fixture")
    monkeypatch.setattr(experiment, "load_sportka_csv", lambda *args, **kwargs: df)
    observed = []
    original_fit = MLPClassifier.fit
    def recording_fit(self, x, y):
        observed.append((x.copy(), y.copy()))
        return original_fit(self, x, y)
    monkeypatch.setattr(MLPClassifier, "fit", recording_fit)
    with pytest.warns(ConvergenceWarning):
        first = experiment.run(source, cache_dir=tmp_path / "models", hidden=4, max_iter=2)
    assert first["split_samples"] == {"training": 103, "validation": 22, "test": 23}
    assert len(observed) == 4
    for x, y in observed:
        assert x.shape == (103, 147)
        np.testing.assert_array_equal(y, base_features(df)[52:155])
    def forbidden_fit(*args, **kwargs):
        raise AssertionError("Second run should reuse saved models")
    monkeypatch.setattr(MLPClassifier, "fit", forbidden_fit)
    second = experiment.run(source, cache_dir=tmp_path / "models", hidden=4, max_iter=2)
    for kind in experiment.KINDS:
        assert not first["models"][kind]["cache_reused"]
        assert second["models"][kind]["cache_reused"]
        assert first["models"][kind]["test"] == second["models"][kind]["test"]
        assert first["models"][kind]["validation"]["n"] == 22
        assert first["models"][kind]["test"]["n"] == 23
