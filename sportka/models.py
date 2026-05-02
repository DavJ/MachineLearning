"""
Model definitions for the Sportka UBT/theta experiment.

Three families:
  - Baseline models (no learning): RandomPredictor, GlobalFreqPredictor,
    RollingFreqPredictor
  - ML models: LogisticRegressionModel, MLPModel
  - UBT model: UBTModel (MLP with the full feature set)

All models share the same interface:
    fit(X_train, Y_train)
    predict_proba(X)  -> ndarray (n_samples, 49)  probabilities per number
"""

from __future__ import annotations

import numpy as np
from typing import Optional


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class BaseModel:
    """Shared interface for all models."""

    name: str = "base"

    def fit(self, X_train: np.ndarray, Y_train: np.ndarray) -> "BaseModel":
        """Fit the model. Y_train is (n, 49) binary matrix."""
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probability estimate (n_samples, 49)."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Baseline models
# ---------------------------------------------------------------------------

class RandomPredictor(BaseModel):
    """
    Null baseline: draw 7 numbers uniformly at random from 1–49.
    Returns uniform probability 7/49 = 1/7 ≈ 0.143 for every number.
    """

    name = "random_uniform"

    def fit(self, X_train: np.ndarray, Y_train: np.ndarray) -> "RandomPredictor":
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        n = len(X)
        return np.full((n, 49), 7.0 / 49.0, dtype=np.float32)


class GlobalFreqPredictor(BaseModel):
    """
    Predicts proportional to the historical frequency of each number
    in the training set.
    """

    name = "global_frequency"

    def __init__(self):
        self._freq: Optional[np.ndarray] = None

    def fit(self, X_train: np.ndarray, Y_train: np.ndarray) -> "GlobalFreqPredictor":
        freq = Y_train.sum(axis=0).astype(np.float64)
        total = freq.sum()
        self._freq = (freq / total).astype(np.float32) if total > 0 else np.full(49, 1.0 / 49.0, dtype=np.float32)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        n = len(X)
        return np.tile(self._freq, (n, 1))


class RollingFreqPredictor(BaseModel):
    """
    Uses the last `window` draws' frequencies as the probability estimate.
    Requires that X already contains a rolling-frequency feature block
    (first 49 columns are assumed to be the rolling-window-1 frequencies).

    For simplicity this predictor takes the rolling-freq-1 slice from X
    when X contains winding features, otherwise falls back to global freq.
    """

    name = "rolling_frequency"

    def __init__(self, window_feature_offset: int = 49 + 13):
        """
        window_feature_offset: index of the first rolling-window-1 feature
        in the combined feature matrix (after base + time features = 49+13=62).
        """
        self._offset = window_feature_offset
        self._global_freq: Optional[np.ndarray] = None

    def fit(self, X_train: np.ndarray, Y_train: np.ndarray) -> "RollingFreqPredictor":
        freq = Y_train.sum(axis=0).astype(np.float64)
        total = freq.sum()
        self._global_freq = (freq / total).astype(np.float32) if total > 0 else np.full(49, 1.0 / 49.0, dtype=np.float32)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        n_cols = X.shape[1]
        if n_cols > self._offset + 49:
            probs = X[:, self._offset: self._offset + 49].astype(np.float32)
            # Normalise rows
            row_sum = probs.sum(axis=1, keepdims=True)
            row_sum = np.where(row_sum == 0, 1.0, row_sum)
            return probs / row_sum
        # Fallback
        return np.tile(self._global_freq, (len(X), 1))


# ---------------------------------------------------------------------------
# ML models (scikit-learn wrappers)
# ---------------------------------------------------------------------------

class LogisticRegressionModel(BaseModel):
    """
    Independent logistic regression for each of the 49 numbers.
    Treats each number as an independent binary classification problem.
    """

    name = "logistic_regression"

    def __init__(self, C: float = 0.1, max_iter: int = 200, solver: str = "saga"):
        from sklearn.linear_model import LogisticRegression

        self._C = C
        self._max_iter = max_iter
        self._solver = solver
        self._models: list = []

    def fit(self, X_train: np.ndarray, Y_train: np.ndarray) -> "LogisticRegressionModel":
        from sklearn.linear_model import LogisticRegression

        self._models = []
        for j in range(49):
            clf = LogisticRegression(C=self._C, max_iter=self._max_iter, solver=self._solver)
            y_j = Y_train[:, j]
            # Only fit if both classes are present
            if y_j.sum() > 0 and (1 - y_j).sum() > 0:
                clf.fit(X_train, y_j)
            else:
                clf = None
            self._models.append(clf)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        n = len(X)
        out = np.full((n, 49), 7.0 / 49.0, dtype=np.float32)
        for j, clf in enumerate(self._models):
            if clf is not None:
                out[:, j] = clf.predict_proba(X)[:, 1].astype(np.float32)
        return out


class MLPModel(BaseModel):
    """
    Small multi-layer perceptron using scikit-learn's MLPClassifier,
    trained independently per number (binary outputs).

    hidden_layer_sizes controls the architecture (1–2 layers).
    """

    name = "mlp"

    def __init__(
        self,
        hidden_layer_sizes=(64,),
        max_iter: int = 200,
        alpha: float = 1e-3,
        random_state: int = 0,
    ):
        self._hidden = hidden_layer_sizes
        self._max_iter = max_iter
        self._alpha = alpha
        self._random_state = random_state
        self._models: list = []

    def fit(self, X_train: np.ndarray, Y_train: np.ndarray) -> "MLPModel":
        from sklearn.neural_network import MLPClassifier

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
                clf.fit(X_train, y_j)
            else:
                clf = None
            self._models.append(clf)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        n = len(X)
        out = np.full((n, 49), 7.0 / 49.0, dtype=np.float32)
        for j, clf in enumerate(self._models):
            if clf is not None:
                out[:, j] = clf.predict_proba(X)[:, 1].astype(np.float32)
        return out


# ---------------------------------------------------------------------------
# UBT model (full feature set MLP)
# ---------------------------------------------------------------------------

class UBTModel(MLPModel):
    """
    UBT model: MLP trained on the full combined feature set
    (base + time + winding + torus + theta).

    This is an MLPModel variant with slightly larger capacity.
    """

    name = "ubt_mlp"

    def __init__(
        self,
        hidden_layer_sizes=(128, 64),
        max_iter: int = 300,
        alpha: float = 1e-3,
        random_state: int = 0,
    ):
        super().__init__(
            hidden_layer_sizes=hidden_layer_sizes,
            max_iter=max_iter,
            alpha=alpha,
            random_state=random_state,
        )
        self.name = "ubt_mlp"


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

def get_all_models():
    """
    Return a dict of {name: model_instance} for all experiment models.
    """
    return {
        RandomPredictor.name: RandomPredictor(),
        GlobalFreqPredictor.name: GlobalFreqPredictor(),
        RollingFreqPredictor.name: RollingFreqPredictor(),
        LogisticRegressionModel.name: LogisticRegressionModel(),
        "mlp_base": MLPModel(hidden_layer_sizes=(64,), max_iter=200),
        UBTModel.name: UBTModel(),
    }
