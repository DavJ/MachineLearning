# Sportka repair summary — 22 October 2026

The repair focuses on valid Sportka rules, reproducible evaluation and usable
offline commands. No winning or profitability guarantee is made.

## Corrected behavior

- Six main balls and a separate bonus replace the experiment's seven-ball targets.
- Valid six-number columns, uniform sampling without duplicate combinations,
  exact jackpot coverage and main/bonus tier probabilities.
- Explicit user-supplied cost/budget validation with decimal arithmetic.
- Direct public Allwyn CSV download without Selenium, validated before atomic replacement.
- Strict local CSV loading with chronological ordering and actionable errors.
- Importing the legacy history module no longer downloads data or trains models;
  histories belong to individual instances.
- Prefix-only winding features and retained history at all temporal split edges.
- Linear-time exponential frequency accumulation and consistent frequency units.
- Rolling baseline handles a feature block ending at the final column.
- Constant training labels retain their observed zero/one probability.
- Float64 BCE clipping prevents NaNs for float32 zero/one scores.
- Default pytest collection no longer executes/exits the legacy smoke script.
- Configuration objects no longer mutate shared nested defaults.
- Optional neural training uses binary six-ball labels and chronological batches;
  old target-incompatible weights require retraining. Reload rebuilds architecture.
- TensorFlow device visibility is left unchanged; GUI callback imports and stop
  behavior are corrected, with Tk variable reads performed on the main thread.
- Missing real CSV paths fail instead of silently using synthetic draws.
- Null screens cannot label significantly worse real-data performance as signal.
- Core and optional neural dependencies are separated, with GitHub Actions tests.

## Verification scope

At this revision, `python -m pytest -q` passed all 72 tests, including TensorFlow
training/persistence and all experiment-runner reports. The documented CLI smoke
commands passed, and both draws from the official 3,821-date export were evaluated
with the fixed baselines and logistic regression. See the dated baseline report. Synthetic backtests test execution and null behavior only. Actual
predictive performance requires an official local CSV and untouched future draws.
The direct Allwyn CSV endpoint was exercised on the official export. GUI
event-loop behavior still requires a desktop environment; headless tests do not
verify the complete desktop interface. The finance examples are outside this repair's scope.
