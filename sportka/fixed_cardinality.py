"""Condition independent Bernoulli scores on a fixed number of selected balls.

This defines a model over complete sets; it does not establish calibration or
a predictive edge. It cannot represent arbitrary interactions between balls.
The normalizer and marginals use exact dynamic programming, not Monte Carlo.
"""

from __future__ import annotations

from math import comb, log
from numbers import Integral

import numpy as np

from sportka.rules import MAIN_COUNT


VERSION = "conditional-bernoulli-v1"
PROBABILITY_CLIP = 1e-9


class FixedCardinality:
    """Batch of positive, normalized distributions over ``count``-element sets.

    ``scores`` is a nonempty (rows, pool) matrix in [0, 1]. After a fixed
    float64 clip to [1e-9, 1-1e-9], let w[j] = p[j] / (1-p[j]). A set S has
    probability product(w[S]) / sum_{|T|=count} product(w[T]). The original
    scores are generally different from the resulting inclusion marginals.

    No parameters are fitted. A common odds multiplier cancels, and top-score
    sets remain maximum-probability sets, except ties introduced by clipping.
    General pool/count geometry also permits exhaustive small-lottery checks.
    """

    def __init__(self, scores: np.ndarray, count: int = MAIN_COUNT):
        p = np.asarray(scores, dtype=np.float64)
        if p.ndim != 2 or not p.shape[0] or not p.shape[1]:
            raise ValueError("Expected a nonempty (rows, pool) score matrix")
        if not np.all(np.isfinite(p)) or not np.all((p >= 0) & (p <= 1)):
            raise ValueError("Scores must be finite and in [0, 1]")
        if isinstance(count, bool) or not isinstance(count, Integral) or not 0 <= count <= p.shape[1]:
            raise ValueError("count must be an integer between zero and pool size")
        self.count = int(count)
        self.shape = p.shape
        p = np.clip(p, PROBABILITY_CLIP, 1 - PROBABILITY_CLIP)
        self._log_weights = np.log(p) - np.log1p(-p)
        # A row-wise common odds scale has no effect on a fixed-size law.
        self._log_weights -= self._log_weights.mean(axis=1, keepdims=True)
        n, pool = self.shape
        self._prefix = np.full((n, pool + 1, self.count + 1), -np.inf)
        self._prefix[:, :, 0] = 0.0
        for j in range(pool):
            self._prefix[:, j + 1, 1:] = np.logaddexp(
                self._prefix[:, j, 1:],
                self._log_weights[:, j, None] + self._prefix[:, j, :-1],
            )
        self._log_normalizer = self._prefix[:, -1, self.count]

    def log_prob(self, targets: np.ndarray) -> np.ndarray:
        """Return one complete-set log probability per row, in natural logs."""
        y = np.asarray(targets)
        if y.shape != self.shape or not np.all((y == 0) | (y == 1)):
            raise ValueError("Targets must be matching binary rows")
        if not np.all(y.sum(axis=1) == self.count):
            raise ValueError("Each target must contain exactly count selected balls")
        return np.sum(y * self._log_weights, axis=1) - self._log_normalizer

    def marginals(self) -> np.ndarray:
        """Return implied inclusion probabilities, with every row summing to count."""
        n, pool = self.shape
        if self.count == 0:
            return np.zeros(self.shape, dtype=np.float64)
        # Keep only the suffix at the current position; prefix is already saved.
        suffix = np.full((n, self.count + 1), -np.inf)
        suffix[:, 0] = 0.0
        result = np.empty(self.shape, dtype=np.float64)
        for j in range(pool - 1, -1, -1):
            without_j = np.logaddexp.reduce(
                self._prefix[:, j, :self.count] + suffix[:, self.count - 1::-1], axis=1
            )
            result[:, j] = np.exp(self._log_weights[:, j] + without_j - self._log_normalizer)
            updated = suffix.copy()
            updated[:, 1:] = np.logaddexp(suffix[:, 1:], self._log_weights[:, j, None] + suffix[:, :-1])
            suffix = updated
        # Remove only roundoff excursions at the probability boundaries.
        return np.clip(result, 0.0, 1.0)


def conditional_set_metrics(targets: np.ndarray, scores: np.ndarray,
                            count: int = MAIN_COUNT) -> dict[str, float]:
    """Score the explicitly conditioned law, separately from raw-score metrics.

    Set NLL and gain are nats per whole draw. Conditional BCE/Brier are means
    per ball. Positive gain against uniform is descriptive, not a significance
    test, profit estimate, or validation of the original sigmoid outputs.
    """
    law = FixedCardinality(scores, count)
    log_probability = law.log_prob(targets)
    marginal = law.marginals()
    clipped = np.clip(marginal, PROBABILITY_CLIP, 1 - PROBABILITY_CLIP)
    y = np.asarray(targets, dtype=np.float64)
    nll = -float(log_probability.mean())
    return {
        "conditional_set_nll": nll,
        "conditional_set_gain_vs_uniform": log(comb(law.shape[1], law.count)) - nll,
        "conditional_bce": float(-(y * np.log(clipped) + (1 - y) * np.log1p(-clipped)).mean()),
        "conditional_brier": float(np.mean((y - marginal) ** 2)),
    }
