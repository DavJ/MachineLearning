# Sportka technical notes

## Rules and target semantics

Sportka uses six main numbers from 49 in each of two draws, with an additional
bonus ball for each draw. A column contains six numbers shared by both draws.
The bonus distinguishes five-main-match prize tiers and is never a seventh main
match. Source: [Allwyn](https://www.allwyn.cz/loterie/sportka/jak-hrat), checked
2026-10-22. Šance/Superjackpot are separate and are not modelled.

`rules.py` validates combinations, classifies main/bonus prize tiers and computes
exact per-column tier probabilities. `tickets.py` samples integer combination
ranks uniformly without replacement and un-ranks them to valid columns. This
avoids duplicate jackpot outcomes and unbounded rejection loops. Under a fair
draw, every valid six-number column has the same jackpot probability.

## Modules

| Module | Purpose |
| --- | --- |
| `data.py` | Local CSV validation, separate bonus columns, legacy history objects |
| `learn.py` | Lightweight compatibility imports, no network or training side effects |
| `cli.py` | Offline tickets, odds, limits and reproducible baseline backtests |
| `features.py` | Base, time, rolling, exponential, torus and theta feature families |
| `evaluation.py` | Causal temporal splits, proper scores, hit rates and controls |
| `fixed_cardinality.py` | Exact conditional Bernoulli six-set likelihoods and implied inclusion marginals |
| `models.py`, `model.py` | Frequency baselines and experimental sklearn models |
| `neural_network.py` | Optional TensorFlow first-draw model and metadata-aware weights |
| `gui.py` | Optional Tkinter workflow with six-number output |
| `config.py` | Independent deep-copied configuration defaults |

## Causal feature contract

Feature row `i` may use draws `0..i` and predicts the main numbers of draw `i+1`.
Rolling and exponential histories include the current known draw. The cumulative
frequency reference at row `i` is computed only from that prefix, in the same
per-draw inclusion-probability units. Appending future draws cannot alter earlier
features. Exponential history uses a linear-time recurrence rather than rescanning
all earlier draws for every row.

The time-index scale is fitted on training data and reused for validation/test.
`temporal_feature_split` constructs causal features once and splits by **target**
position. It retains prior history at boundaries and includes every held-out
target. The v2 tensor and null-test runners share this split contract.
`build_walk_forward_features` remains available for custom sequential builders.

Labels are binary vectors with six ones. A fair marginal baseline is `6/49` per
number, so inclusion probabilities sum to six, not one or seven. Learned sigmoid
scores are not constrained to sum to six and must not be labelled calibrated
probabilities without additional validation. The probability assertion checks
finite values, bounds and every row's total rather than an average over rows.
BCE clips in float64 so exact zero/one float32 predictions remain finite.

The shared evaluator also [conditions independent scores on exactly six
balls](../docs/sportka_exact_six.md), with a fixed 1e-9 boundary clip. It reports
complete-set NLL/gain and the resulting marginal BCE/Brier separately from raw
scores. This specifies a normalized model; it does not establish empirical
calibration or infer arbitrary ball interactions from marginal probabilities.

## Testing and interpretation

```bash
python -m pytest -q
python sportka_launcher.py backtest --synthetic 1000 --seed 42
```

Regression tests cover CSV bonus separation, invalid rows, independent histories,
import side effects, exact combinatorial odds, duplicate coverage, decimal limits,
future-data mutation, split boundaries, constant labels, CLI errors and optional
TensorFlow training/weight round trips. CI runs core tests on Python 3.10/3.12
and neural tests on Python 3.12. Neural tests skip when TensorFlow is absent.

The default CLI evaluates uniform, global-frequency and rolling-frequency
baselines. `--include-ml` adds a fixed logistic model. Reports include proper
probability scores and main-hit rates, but not returns. The larger experiment
runners remain research tools and can be expensive. Their null verdict is an
exploratory interval screen; it cannot prove significance or equivalence.
Historical reports generated with the seven-ball target need rerunning.

Biquaternion encodings and theta rotations are feature transforms, not evidence
of a forecasting advantage. An orthogonal rotation preserves Euclidean geometry;
it does not in general decorrelate arbitrary data. Tests of feature correctness
and software execution do not establish profitable lottery prediction.
