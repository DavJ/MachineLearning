# Machine Learning Projects

Experiments in lottery modelling and financial data analysis.

## Sportka / Allwyn

Sportka tools now generate valid **six-number** columns, keep the bonus ball
separate, calculate exact jackpot odds, and evaluate research models on
chronological held-out draws. The default ticket generator is uniform and
avoids duplicate combinations. It works offline without TensorFlow or a browser.

**No model in this repository has demonstrated a reproducible lottery edge.**
Fair independent draws cannot be predicted from historical results. More features,
more training, biquaternions, and theta transforms do not establish predictability.
Winning every time or guaranteeing the largest prize is not achievable here.

```bash
# Python 3.10+
python -m pip install -r requirements-sportka.txt pytest
python -m pytest -q

# One valid six-number column; default odds cover both main draws
python sportka_launcher.py predict
python sportka_launcher.py odds --lines 1 --draws 1 --json

# Explicit synthetic diagnostic; this does not use real lottery results
python sportka_launcher.py backtest --synthetic 1000 --seed 42

# Explicitly download the public CSV, then evaluate either draw
python sportka_launcher.py download --directory ./data

# Evaluate an existing Allwyn/Sazka CSV, first or second draw
python sportka_launcher.py backtest --csv /path/to/sportka.csv --game 1
python sportka_launcher.py backtest --csv /path/to/sportka.csv --game 2
```

For one column in one main draw, matching all six numbers has probability
`1 / C(49,6) = 1 / 13,983,816`. Distinct columns cover distinct jackpot outcomes;
duplicate columns do not increase the probability of hitting a jackpot. Covering
more outcomes also costs more. A six-number jackpot is different from the
Superjackpot, which has additional eligibility conditions involving Šance.

[Allwyn's rules and prize tiers](https://www.allwyn.cz/loterie/sportka/jak-hrat)
and [explanation of the two draws](https://www.allwyn.cz/allwyn-svet/blog/jak-se-urcuje-vyhra-ve-sportce)
were checked on 22 October 2026. Prices and prize amounts are not hardcoded.

## Optional neural experiment and GUI

```bash
python -m pip install -r requirements-sportka-neural.txt
python sportka_launcher.py train --csv /path/to/sportka.csv --date 01.11.2026 --epochs 10
python sportka_launcher.py gui
```

The neural model predicts the first main draw using earlier results from both
draws. Its outputs are experimental scores, not demonstrated winning probabilities.
Existing weights using seven-ball or normalised targets must be retrained.
TensorFlow uses the available device configuration; Apple GPU support requires a
compatible backend installation. Tkinter is needed for the GUI. No command buys
tickets, connects to a gambling account, or increases a budget automatically.

## Documentation

- [Usage](USAGE.md): commands, CSV format, budget validation and error handling
- [Technical notes](sportka/README.md): rules, causal evaluation and limitations
- [Exact six-set scoring](docs/sportka_exact_six.md): normalized set likelihoods for existing model outputs
- [Repair summary](IMPLEMENTATION_SUMMARY.md): bugs fixed and verification scope
- [Real-data baseline check](reports/sportka_baseline_2026-10-22.md): both draws, 3,821 dates
- [Historical reports](reports/README.md): archived experiments requiring reruns

## Signal audit

[Audit and sensitivity controls](docs/sportka_signal_audit.md) test single-number
frequencies, all pairs and temporal ordering against appropriate lottery and
permutation references, with explicit search correction. These are historical
diagnostics and synthetic detector checks, not a demonstrated betting edge.

[Prospective checks](docs/sportka_prospective.md) seal three baseline forecasts
for both main draws before a future date and score only new results. A fixed
mixture of sequential tests accounts for the registered models and repeated
inspection. Forecasts and protocol must be published before their target day;
the [first scored date](reports/sportka_prospective_2026-10-25.md) provides no
prospective evidence of an advantage.

## Finance

The `finance/` directory contains separate Binance, Alpha Vantage and Kalman filter
experiments. These are not part of the Sportka validation. `requirements.txt`
contains the broader legacy dependencies; the smaller Sportka requirements are
recommended for this project.
