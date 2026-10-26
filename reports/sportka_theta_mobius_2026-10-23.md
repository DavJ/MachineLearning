# Sportka neural theta / Möbius / Mertens check — 23 October 2026

The uniform baseline had lower test BCE and Brier scores than all four neural
variants in both draws. This configured experiment does not demonstrate a
prediction advantage. All eight neural fits reached their 150-iteration limit
without convergence, so this is also an optimization-limited comparison.

## Data and protocol

- [Official Allwyn CSV](https://www.allwyn.cz/loterie/historie-cisel?game=sportka), freshly downloaded and validated.
- 3,821 dates through 2026-10-21; source SHA-256 `631ecac60a8884a96a5331675972edc336f07e826c72eea207f66edc8442d5d6`.
- After 52 warmup draws: 2,638 training, 565 validation and 566 test targets per draw.
- Test dates: 2023-03-15 through 2026-10-21. Earlier work has already examined this historical period; this is not prospective validation.
- Same 147 feature positions and target rows for every variant. The previous draw stays raw, while 16/52-draw frequency channels are raw or transformed.
- Fixed one-hidden-layer MLP: 32 units, alpha=0.1, max_iter=150, seed=42. No variant is selected for betting.

## Held-out point estimates

| Draw | Feature variant | Test BCE ↓ | Test Brier ↓ | Main hits / six | Iteration limit reached |
| --- | --- | ---: | ---: | ---: | --- |
| 1 | raw | 0.378055 | 0.108846 | 0.7208 | yes |
| 1 | theta | 0.382610 | 0.109794 | 0.7951 | yes |
| 1 | mobius | 0.378074 | 0.108799 | 0.7739 | yes |
| 1 | mertens | 0.379589 | 0.109150 | 0.7491 | yes |
| 1 | uniform | 0.371776 | 0.107455 | 0.7650 | n/a |
| 2 | raw | 0.379724 | 0.109266 | 0.7420 | yes |
| 2 | theta | 0.384859 | 0.110465 | 0.7208 | yes |
| 2 | mobius | 0.379599 | 0.109175 | 0.6979 | yes |
| 2 | mertens | 0.381512 | 0.109637 | 0.7332 | yes |
| 2 | uniform | 0.371776 | 0.107455 | 0.7544 | n/a |

Point estimates and occasional higher hit averages do not establish significance,
profit or a jackpot advantage. Comparing four variants on two already examined
historical test periods creates selection opportunities; a forecasting claim
requires a prespecified future evaluation. These are finite damped cosine
features with Möbius or Mertens weights, not a derived UBT prediction or the
separate adaptive Theta Grid architecture.

## Persistence and verification

All eight models were saved locally. A second complete execution reused all
eight checkpoints without fitting, and validation/test metrics matched exactly.
Checkpoint identities now include network parameters and dependency versions;
the stored training-array signature prevents accidental reuse for other data.
Changed history/configuration triggers a new fit rather than incremental resume.
Use only trusted local joblib files.

85 pytest tests passed, including actual checkpoint round trips, training-only
fit boundaries, cached-run reuse and failed-write preservation. `compileall`
and `git diff --check` passed.

Full validation/test metrics, configuration, versions and checkpoint identities:
[JSON results](sportka_theta_mobius_2026-10-23.json).

```bash
python -m sportka.experiments.theta_mobius --csv data/sportka.csv --game 1
python -m sportka.experiments.theta_mobius --csv data/sportka.csv --game 2
```
