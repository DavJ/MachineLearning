# Sportka reports

Reports predating the 22 October 2026 repair are historical. They used a seven-ball
main target and/or earlier split handling and must not be treated as evidence of
a valid six-main-ball Sportka prediction advantage. Their plots are also archived.

Generate a fresh source-labelled baseline readout with:

```bash
python sportka_launcher.py backtest --csv /path/to/sportka.csv --game 1
python sportka_launcher.py backtest --synthetic 1000 --seed 42
```

The second command is a synthetic null diagnostic, not a real lottery backtest.

## Current check

[Corrected real-data baseline check, 22 October 2026](sportka_baseline_2026-10-22.md)
includes both draws, the source hash and all metrics. It demonstrates execution
and reports held-out point estimates; it does not establish a betting edge.
