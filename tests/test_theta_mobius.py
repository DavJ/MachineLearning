"""Small deterministic tests for number theory and past-only features."""
import numpy as np
import pandas as pd

from sportka.experiments.theta_mobius import mobius, spectral, dataset, cache_key


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
