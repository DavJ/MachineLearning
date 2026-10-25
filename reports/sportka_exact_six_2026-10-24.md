# Exact six-set scoring — 24 October 2026

**All three existing nonuniform baselines have worse complete-set test NLL than
uniform in both games. No predictive edge is established.** This is a
descriptive re-scoring of previously studied history, without a significance
test, model selection or changes to the registered prospective forecasts.

## Source and fixed protocol

- Reused the frozen [official Allwyn export](https://www.allwyn.cz/loterie/historie-cisel?game=sportka)
  already validated for the 23 October prospective issue. This run did not
  download another snapshot.
- 3,822 dates per game, 1994-01-09 through 2026-10-23.
- CSV SHA-256: `ff89e477192390abfb0ae81892ae824c1eaeefa6e86dc90204949863358f2cbd`.
- Existing chronological split: 2,674 training targets (1994-01-16–2019-02-24),
  573 validation targets (2019-02-27–2023-02-24), 574 test targets
  (2023-02-26–2026-10-23). The first overall draw supplies history only.
- Unchanged feature groups: base, time, winding. Fit uses training targets
  only; each held-out prediction can use earlier observed draws.
- Fixed existing models: uniform, training frequency, first rolling-frequency
  block (window=1), and logistic regression (`C=0.1`, `max_iter=200`,
  `solver=saga`, `random_state=0`). No hyperparameters were searched.
- Python 3.12.14, NumPy 2.3.5, pandas 2.2.3, scikit-learn 1.8.0.

The [JSON companion](sportka_exact_six_2026-10-24.json) contains all original
and conditional validation/test metrics, settings, implementation file hashes,
source provenance and independent arithmetic checks.

## Whole-set test results

The versioned adapter clips independent input scores to `[1e-9, 1-1e-9]`,
then conditions their joint Bernoulli law on exactly six selected balls.
NLL is the mean negative log probability of the actual complete six-set,
in nats per draw; lower is better. Gain is uniform NLL minus model NLL;
positive would be better. Uniform gain is zero up to roundoff.

| Existing model | Game 1 NLL | Game 2 NLL | Game 1 gain | Game 2 gain |
| --- | ---: | ---: | ---: | ---: |
| Uniform | 16.453411 | 16.453411 | 0 | 0 |
| Training frequency | 16.469400 | 16.470577 | −0.015989 | −0.017166 |
| Rolling frequency, window=1 | 216.402884 | 217.991427 | −199.949473 | −201.538016 |
| Logistic regression | 16.997498 | 17.110275 | −0.544086 | −0.656864 |

Every nonuniform model also had negative validation gain. The existing
window=1 baseline returns zero/one scores that repeat the preceding draw;
it assigns extremely little probability to other sets after the fixed clip.
Its very large loss reflects that overconfidence. This run retains that
baseline unchanged so its behavior is explicit and comparable. It is different
from the registered prospective last-52-frequency recipe.

## What changes in evaluation

`compute_all_metrics` now adds exact conditional set NLL/gain and conditional
marginal BCE/Brier, retaining the original raw metrics and ticket-hit policy.
The CLI records the adapter version and semantics. All three legacy experiment
report tables display the set scores. Neural experiments using the same
evaluator inherit the additional scores, but no neural network was fitted for
this dated comparison.

Dynamic programming computes the normalizer and implied marginals without
sampling or enumerating all Sportka combinations. This specifies one coherent
law from a vector of scores; it does not recover arbitrary ball interactions
or make those scores empirically calibrated. Existing models with their own
joint laws should keep their original likelihoods. See the
[method and API](../docs/sportka_exact_six.md).

## Verification and interpretation

- 160 local tests passed, including 41 new mathematical/validation cases;
  three optional TensorFlow tests skipped in the local environment and four
  existing plotting warnings remained. The PR's neural CI job also exercises
  the TensorFlow installation.
- Independent Bernoulli-product enumeration verifies every 8-ball outcome
  at counts 0, 1, 3, 7 and 8: normalization, implied marginals and MAP sets.
- A closed-form one-favored-ball 6-of-49 law checks the full Sportka geometry.
  Tests cover common odds scaling, permutations, batch isolation, input
  mutation, float32 boundaries, rejected inputs, CLI and all report runners.
- Independent CSV parsing confirms all 574 test targets in both games. A
  separate probability-domain Bernoulli-sum recurrence reproduces all six
  baseline/game NLL values within `5.7e-14` nats/draw, without production
  log-space/odds helpers. Logistic metrics are covered by the shared scorer
  tests rather than an independent refit.
- `compileall` and `git diff --check` passed. The prospective ledger is
  byte-identical to master: target 2026-10-25 pending, zero scored dates.

These historical comparisons have no corrected p-values and do not establish
jackpot odds, payout returns, calibration or general impossibility of a signal.
Fresh, genuinely advance-published results remain the next independent evidence.
No model is promoted into the existing prospective protocol.

## Reproduce

Use the exact CSV hash above and the recorded library versions:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python sportka_launcher.py backtest \
  --csv /path/to/sportka.csv --game 1 --include-ml
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python sportka_launcher.py backtest \
  --csv /path/to/sportka.csv --game 2 --include-ml
python -m pytest -q
```
