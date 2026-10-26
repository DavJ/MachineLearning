# Sportka / Allwyn usage

## Install and check

Use Python 3.10 or newer in a virtual environment:

```bash
python -m venv .venv
# Activate the environment for your shell, then:
python -m pip install -r requirements-sportka.txt pytest
python sportka_launcher.py test
python -m pytest -q
```

TensorFlow, ephem, ReportLab and Tkinter are optional. The installation
check reports them separately. To train the neural model or use the GUI, install
`requirements-sportka-neural.txt` and make Tkinter available through your system.

## Generate valid tickets

```bash
python sportka_launcher.py tickets
python sportka_launcher.py predict --lines 2 --json
python sportka_launcher.py tickets --lines 2 --seed 42 --json
```

`predict` is an alias for the offline uniform ticket generator. Each column contains
six distinct integers from 1 to 49. Columns are sampled without replacement from
the entire combination space, so there are no duplicate jackpot outcomes.
The default is one column. A seed is useful for repeatable tests; omit it for
operating-system randomness. Requests are limited to 10,000 columns per call.
There is no claimed prediction advantage for these tickets.

## Odds and limits

```bash
python sportka_launcher.py odds --lines 1 --draws 1 --json
python sportka_launcher.py odds --lines 1 --draws 2 --json
```

For `m` distinct columns, one draw's exact jackpot probability is
`m / C(49,6)`. Across `d` independent draws using the same columns it is
`1 - (1 - m/C(49,6))**d`. The default `--draws 2` represents Sportka's two main
draws. These formulas concern six main matches and exclude Šance, Superjackpot,
jackpot sharing and monetary returns. Duplicate columns are counted once by the
programmatic coverage function. Lower-tier events across multiple columns can
be correlated; their portfolio probability is not computed by this tool.

To enforce a cost limit, supply **both** `--budget` and `--cost-per-line`.
The price is your own total price per column for the intended schedule, in the
same currency as the budget. Extras must be included in that price or budgeted
separately. The code rejects an over-budget request instead of changing the count.
It never purchases a ticket. Prices are not inferred from old documentation.

## Historical evaluation

Download the official semicolon CSV export explicitly, or supply your existing file:

```bash
python sportka_launcher.py download --directory ./data
```

The public export is validated before atomically replacing `sportka.csv`; a failed
request or malformed response preserves any existing file. No browser is needed.

```bash
python sportka_launcher.py backtest --csv /path/to/sportka.csv --game 1
python sportka_launcher.py backtest --csv /path/to/sportka.csv --game 2
python sportka_launcher.py backtest --csv /path/to/sportka.csv --include-ml
```

The supported export layout has a header followed by:

| Positions (zero-based) | Content |
| --- | --- |
| 0 | Date, DD. MM. YYYY or YYYY-MM-DD |
| 1–3 | Year, week, weekday metadata |
| 4–9 | First draw's six main numbers |
| 10 | First draw's bonus |
| 11–16 | Second draw's six main numbers, if present |
| 17 | Second draw's bonus, if present |

The loader sorts dates ascending, accepts UTF-8 (including BOM) or Windows-1250,
and rejects duplicate dates, repeated or out-of-range balls, invalid dates and
malformed rows. It reports the offending CSV line instead of silently dropping
it. First-draw evaluation can use an export without the second draw; the legacy
neural model requires both. Schema changes require an explicit parser update.
Missing files fail; real-data requests never silently become synthetic tests.

Evaluation uses a 70%/15%/15% chronological split. The first overall draw lacks a
preceding history and is omitted as a target. All validation and test targets
retain prior history across split boundaries. Model fitting uses training data
only; validation and test predictions use actual previously observed draws.
The backtest is one-step-ahead, not a prediction of all future draws at once.
No hyperparameters or betting strategy are selected from the test set.

JSON includes source type and CSV SHA-256, sample counts, BCE, Brier score,
recall, average main hits, and rates of 3+, 4+, 5+ and 6 main matches for the
selected six-number column. Equal scores use a fixed numerical tie ordering.
Prize amounts and profit are not estimated. A 5+ main-hit rate does not
distinguish the separate 5+bonus tier.

The same report includes `conditional_set_nll` and
`conditional_set_gain_vs_uniform`: log scores for the complete observed six-set
after conditioning independent model outputs on six balls. Conditional marginal
BCE/Brier use the resulting inclusion probabilities; raw BCE/Brier are retained.
The JSON names the fixed clipping rule and score version. See
[the method and limits](docs/sportka_exact_six.md). Positive historical gain is
descriptive and has no significance or profitability guarantee.

Use synthetic data explicitly for diagnostics:

```bash
python sportka_launcher.py backtest --synthetic 1000 --seed 42
```

The result is labelled `synthetic_null`; it is not evidence of real-world success.
Theoretical random average main hits per six-number column are `36/49 ≈ 0.7347`.
A chance fluctuation in a holdout is not a reproducible predictive edge.

## Optional neural experiment

```bash
python sportka_launcher.py train --csv /path/to/sportka.csv --date 01.11.2026 --epochs 10
python sportka_launcher.py gui
```

Choose a target date after every loaded draw. With no date, the training command
uses tomorrow on the machine running it. It reads an existing local CSV and does
not download automatically. Biquaternion features can be disabled with
`--no-biquaternion`. New weights include target-semantics metadata; legacy or
unidentified weights are rejected so they cannot silently produce invalid output.
The model uses binary six-main-number labels, sigmoid outputs, binary
cross-entropy and chronological validation. These corrections do not establish
predictive ability. Its first-draw scores have not been calibrated as ticket odds.

The GUI reads a selected CSV directly, supports model training and selection of
six numbers, and exports results/PDFs. Its stop button requests a stop after the
current training batch. The public CSV downloader is a separate explicit
action and needs no browser. If its format or URL changes, load the official CSV
manually and update the parser. There is no automated betting integration.

## Theta / Möbius / Mertens research comparison

```bash
python sportka_launcher.py download --directory ./data
python -m sportka.experiments.theta_mobius --csv data/sportka.csv --game 1
python -m sportka.experiments.theta_mobius --csv data/sportka.csv --game 2
```

The four variants use the same 147 feature positions, target dates, fixed MLP
settings and chronological training/validation/test split after a 52-draw warmup.
The last draw remains raw in every variant; the 16/52-draw frequency channels
are raw or receive a finite theta-like cosine transform. Möbius variants use
mu(n) coefficients; Mertens variants use their partial sums. This experiment is
not the separate adaptive Theta Grid architecture and does not derive a lottery
signal from UBT. JSON reports both validation and test scores and iteration limits.

Models are saved automatically under `models/sportka_research`. An identical
source/configuration reuses the saved model, avoiding repeated fitting. The key
includes the CSV hash, draw, feature settings, network parameters and dependency
versions. The checkpoint also verifies the exact training arrays before reuse.
Use only a local checkpoint directory you trust, because joblib files execute
Python code when loaded. To retrain explicitly:

```bash
python -m sportka.experiments.theta_mobius --csv data/sportka.csv --game 1 --fresh
```

`--hidden`, `--max-iter` and `--seed` change the configuration and create a new
checkpoint identity. Changed history is retrained for the fixed chronological
protocol; cached models do not incrementally resume optimization on new data.
Choosing a winning variant from these test scores would be retrospective model
selection. A forecasting claim needs a prespecified future evaluation.

To measure recovery of known artificial signals with this same frozen neural
configuration and validation-only selection:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m sportka.experiments.neural_recovery \
  --csv data/sportka.csv --trials 16 --workers 2
```

The [method](docs/sportka_neural_recovery.md) records weak/strong number, pair and
temporal controls, whole-set likelihoods, saved models and limits. Historical
results remain exploratory, and the registered prospective protocol is unchanged.
