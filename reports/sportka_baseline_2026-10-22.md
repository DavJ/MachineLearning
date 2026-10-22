# Corrected Sportka baseline check — 22 October 2026

The software now evaluates six main numbers separately from the bonus. In this
fixed comparison, logistic regression had worse held-out BCE and Brier scores
than uniform inclusion probabilities in both draws. No tested method establishes
a reproducible prediction edge, and this check does not estimate monetary profit.

- Source: [official Allwyn export](https://www.allwyn.cz/loterie/historie-cisel?game=sportka).
- 3,821 dates, 1994-01-09 through 2026-10-21, with two separately evaluated draws.
- CSV SHA-256: `631ecac60a8884a96a5331675972edc336f07e826c72eea207f66edc8442d5d6`.
- Per draw: 2,673 training pairs, 573 validation targets, 574 test targets.
- Fixed chronological 70%/15%/15% split; past observed results remain available
  at split boundaries. No model is selected or tuned from the test results.

| Draw | Model | Test BCE (lower better) | Test Brier (lower better) | Mean main hits / 6 | Six matches |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | random_uniform | 0.371776 | 0.107455 | 0.7631 | 0 |
| 1 | global_frequency | 0.372101 | 0.107523 | 0.7666 | 0 |
| 1 | rolling_frequency | 4.417859 | 0.213184 | 0.7770 | 0 |
| 1 | logistic_regression | 0.382568 | 0.109900 | 0.7422 | 0 |
| 2 | random_uniform | 0.371776 | 0.107455 | 0.7578 | 0 |
| 2 | global_frequency | 0.372124 | 0.107529 | 0.7439 | 0 |
| 2 | rolling_frequency | 4.448805 | 0.214677 | 0.7404 | 0 |
| 2 | logistic_regression | 0.384847 | 0.110400 | 0.7160 | 0 |

These are point estimates, not corrected significance tests. Small differences
in hit averages or lower-tier rates must not be interpreted as an edge. The
rolling baseline uses the previous draw as a 0/1 score vector and therefore
incurs a large probability-score penalty when the next draw differs. It is a
diagnostic comparison, not the default ticket generator.

The exact fair-draw expectation is 36/49 ≈ 0.7347 main hits for a fixed
six-number column, with jackpot probability 1/13,983,816 per draw. Zero
jackpot hits in these samples is unsurprising and is not by itself evidence
for or against predictability. Uniform top-k ties use a fixed numeric ordering;
the standalone ticket generator samples distinct combinations using OS randomness.

To rerun on the same export (a future download may contain additional dates):

```bash
python sportka_launcher.py backtest --csv /path/to/sportka.csv --game 1 --include-ml
python sportka_launcher.py backtest --csv /path/to/sportka.csv --game 2 --include-ml
```

Full metrics, provenance and dependency versions: [JSON](sportka_baseline_2026-10-22.json).

Verification at this revision: 72 pytest tests passed, including tiny TensorFlow
training and weight round trips, plus runner/report integration and live public
CSV download/validation. GUI desktop interaction was not exercised.
