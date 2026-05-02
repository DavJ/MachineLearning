"""
Model definitions for the Sportka UBT v2 pipeline.
===================================================
Two model variants for (C, 7, 7) multi-scale tensor input:

  UBTMLPV2
    Shallow MLP (sklearn) trained on flattened (C*49,) features.
    Architecture: 2 hidden layers (spec: max 2–3 layers).
    Loss: binary cross-entropy via sklearn MLPClassifier.
    Recommended for fast experimentation.

  UBTCNNv2
    Small 2-layer convolutional network with toroidal boundary conditions,
    followed by a 1-hidden-layer MLP head (sklearn).
    Convolutional filters are fixed random projections (random kitchen sinks),
    avoiding a deep-learning framework dependency while still enabling spatial
    feature extraction on the 7×7 grid.
    Architecture: Conv(C→nf1, 3×3) → ReLU → Conv(nf1→nf2, 3×3) → ReLU
                  → Flatten → MLP(nf2*49, hidden, 49).

Both models share the interface from sportka.models.BaseModel:
    fit(X_train, Y_train)           X: (n, C, 7, 7) or (n, d)
    predict_proba(X) → (n, 49)      probability per number

Input X should be (n, C, 7, 7) tensors produced by
sportka.multiscale_features.build_multiscale_tensors().
"""

from __future__ import annotations

import numpy as np
from typing import Tuple

from sportka.models import BaseModel

try:
    from sklearn.neural_network import MLPClassifier
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _flatten4d(X: np.ndarray) -> np.ndarray:
    """Flatten (n, C, 7, 7) or already-flat (n, d) to (n, d)."""
    X = np.asarray(X, dtype=np.float32)
    if X.ndim == 4:
        return X.reshape(len(X), -1)
    return X


def _toroidal_conv2d_batched(
    X_batch: np.ndarray,
    kernel: np.ndarray,
) -> np.ndarray:
    """
    Apply a single (kH, kW) kernel to a batch of (H, W) slices using
    toroidal boundary conditions.

    Implemented via np.roll; no external dependencies.

    Args:
        X_batch: (n, H, W) float32 array.
        kernel:  (kH, kW) float32 array; kH and kW must be odd.

    Returns:
        (n, H, W) float32 convolved array.
    """
    kH, kW = kernel.shape
    pH, pW = kH // 2, kW // 2
    out = np.zeros_like(X_batch, dtype=np.float32)
    for dr in range(-pH, pH + 1):
        for dc in range(-pW, pW + 1):
            w = float(kernel[dr + pH, dc + pW])
            if w != 0.0:
                shifted = np.roll(np.roll(X_batch, -dr, axis=1), -dc, axis=2)
                out += w * shifted
    return out


# ---------------------------------------------------------------------------
# UBTMLPV2 — shallow MLP on flattened multi-scale features
# ---------------------------------------------------------------------------

class UBTMLPV2(BaseModel):
    """
    Shallow MLP trained on flattened multi-scale theta tensors.

    Input:  (n, C, 7, 7)  →  flattened to  (n, C*49).
    Output: (n, 49) independent sigmoid probabilities per number.
    Architecture: 1–2 hidden layers (max 2 per spec).
    Loss: binary cross-entropy (sklearn MLPClassifier, one per number).
    """

    name = "ubt_mlp_v2"

    def __init__(
        self,
        hidden_layer_sizes: Tuple[int, ...] = (128, 64),
        max_iter: int = 300,
        alpha: float = 1e-3,
        random_state: int = 0,
    ):
        self._hidden = hidden_layer_sizes
        self._max_iter = max_iter
        self._alpha = alpha
        self._random_state = random_state
        self._models: list = []

    def fit(self, X_train: np.ndarray, Y_train: np.ndarray) -> "UBTMLPV2 | None":
        if not SKLEARN_AVAILABLE:
            print("⚠️ sklearn not available — skipping MLP model")
            return None

        Xf = _flatten4d(X_train)
        self._models = []
        for j in range(49):
            y_j = Y_train[:, j]
            if y_j.sum() > 0 and (1 - y_j).sum() > 0:
                clf = MLPClassifier(
                    hidden_layer_sizes=self._hidden,
                    max_iter=self._max_iter,
                    alpha=self._alpha,
                    random_state=self._random_state,
                )
                clf.fit(Xf, y_j)
            else:
                clf = None
            self._models.append(clf)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Xf = _flatten4d(X)
        n = len(Xf)
        out = np.full((n, 49), 7.0 / 49.0, dtype=np.float32)
        for j, clf in enumerate(self._models):
            if clf is not None:
                out[:, j] = clf.predict_proba(Xf)[:, 1].astype(np.float32)
        return out


# ---------------------------------------------------------------------------
# UBTCNNv2 — toroidal CNN + MLP head
# ---------------------------------------------------------------------------

class UBTCNNv2(BaseModel):
    """
    Small 2-layer CNN with toroidal boundary conditions + MLP head (sklearn).

    Architecture:
        Conv1: C → n_filters_1 feature maps, 3×3 toroidal, ReLU
        Conv2: n_filters_1 → n_filters_2 feature maps, 3×3 toroidal, ReLU
        Flatten: (n_filters_2 * 49,)
        MLP head: sklearn MLPClassifier with 1 hidden layer

    Convolutional filters are fixed random projections.  This avoids a
    deep-learning framework dependency while enabling spatial feature
    extraction on the 7×7 torus grid.

    Constraints satisfied: max 2–3 layers, no deep network.
    """

    name = "ubt_cnn_v2"

    def __init__(
        self,
        n_filters_1: int = 16,
        n_filters_2: int = 8,
        kernel_size: int = 3,
        hidden_mlp: int = 64,
        max_iter: int = 300,
        alpha: float = 1e-3,
        random_state: int = 0,
    ):
        self._nf1 = n_filters_1
        self._nf2 = n_filters_2
        self._ks = kernel_size
        self._hidden_mlp = hidden_mlp
        self._max_iter = max_iter
        self._alpha = alpha
        self._seed = random_state

        rng = np.random.default_rng(random_state)
        # Fixed random filters: (n_filters, kH, kW)
        self._W1 = rng.standard_normal(
            (n_filters_1, kernel_size, kernel_size)
        ).astype(np.float32)
        self._W2 = rng.standard_normal(
            (n_filters_2, kernel_size, kernel_size)
        ).astype(np.float32)

        self._models: list = []

    def _extract_features(self, X: np.ndarray) -> np.ndarray:
        """
        Extract CNN features from (n, C, 7, 7) input.

        Returns:
            (n, n_filters_2 * 49) float32 array.
        """
        X = np.asarray(X, dtype=np.float32)
        n, C = X.shape[0], X.shape[1]

        # Layer 1: (n, C, 7, 7) → (n, nf1, 7, 7)
        layer1 = np.zeros((n, self._nf1, 7, 7), dtype=np.float32)
        for f in range(self._nf1):
            for c in range(C):
                layer1[:, f] += _toroidal_conv2d_batched(X[:, c], self._W1[f])
        layer1 = np.maximum(layer1, 0.0)  # ReLU

        # Layer 2: (n, nf1, 7, 7) → (n, nf2, 7, 7)
        layer2 = np.zeros((n, self._nf2, 7, 7), dtype=np.float32)
        for f in range(self._nf2):
            for c in range(self._nf1):
                layer2[:, f] += _toroidal_conv2d_batched(layer1[:, c], self._W2[f])
        layer2 = np.maximum(layer2, 0.0)  # ReLU

        return layer2.reshape(n, -1)  # (n, nf2 * 49)

    def fit(self, X_train: np.ndarray, Y_train: np.ndarray) -> "UBTCNNv2 | None":
        if not SKLEARN_AVAILABLE:
            print("⚠️ sklearn not available — skipping MLP model")
            return None

        Xf = self._extract_features(X_train)
        self._models = []
        for j in range(49):
            y_j = Y_train[:, j]
            if y_j.sum() > 0 and (1 - y_j).sum() > 0:
                clf = MLPClassifier(
                    hidden_layer_sizes=(self._hidden_mlp,),
                    max_iter=self._max_iter,
                    alpha=self._alpha,
                    random_state=self._seed,
                )
                clf.fit(Xf, y_j)
            else:
                clf = None
            self._models.append(clf)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Xf = self._extract_features(X)
        n = len(Xf)
        out = np.full((n, 49), 7.0 / 49.0, dtype=np.float32)
        for j, clf in enumerate(self._models):
            if clf is not None:
                out[:, j] = clf.predict_proba(Xf)[:, 1].astype(np.float32)
        return out
