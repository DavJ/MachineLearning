# First prospective forecast — 23 October 2026

**Pending target: 2026-10-25. Scored dates: 0. No prospective evidence yet.**
The registered forecasts are research controls, not demonstrated winning picks.

## What is locked

- Protocol: `sportka-prospective-v1`, three baseline recipes for both main draws.
- Issued at: `2026-10-23T20:13:17.114604+00:00` (22:13 in Prague).
- Target: 25 October 2026, a Sunday. The protocol requires publication before
  the beginning of the target Prague calendar day; same-day issue is prohibited.
- [Official Allwyn export](https://www.allwyn.cz/loterie/historie-cisel?game=sportka):
  3,822 dates per game, 1994-01-09 through 2026-10-23, freshly downloaded and validated.
- CSV SHA-256: `ff89e477192390abfb0ae81892ae824c1eaeefa6e86dc90204949863358f2cbd`.
- Semantic history prefix SHA-256:
  `0dfc35cbdc68989eb354a166de919ec588bfb588ed41b86eefb268920e49b7e0`.
- Forecast event SHA-256:
  `912cddc298f8021c948cc569b1d01e2b9a427c71d5f5430e90700996478b6a09`.
- Predictor module SHA-256:
  `5144aaafb0937bfc07fdce9e737a99263fd610fcb85c186731841520c7ecd8a5`.

| Fixed recipe | Game 1 main-number set | Game 2 main-number set |
| --- | --- | --- |
| All-history frequency | 6, 18, 20, 29, 30, 47 | 12, 13, 23, 36, 40, 44 |
| Last-52 frequency | 9, 11, 13, 27, 29, 42 | 1, 5, 20, 22, 40, 42 |
| Previous draw | 17, 21, 30, 38, 41, 43 | 3, 11, 15, 30, 41, 47 |

The [machine-readable ledger](sportka_prospective_ledger.json) is the authoritative
record. Future results are appended; this dated report describes the initial
state. Check the forecast commit's external publication time and later history:
a local timestamp or hash chain alone is not evidence that the forecast existed
before results were known. The official
[schedule](https://www.allwyn.cz/loterie/sportka/jak-hrat) was checked on 23 October;
the CLI itself does not validate draw scheduling.

## What will count as evidence

The mean of 42 persistent game/model/tilt test processes starts at **1.0**.
The current anytime p-value bound is **1.0**. There are no observed hits to
report. The threshold is 20 at the predeclared 0.05 level for this single ledger,
conditional on the fair-draw null, valid source data and genuine advance
publication. All future scored results stay in the calculation, including
losses. Multiple alternative ledgers or restarts are not covered.

These tests concern excess mean main-ball hits. They neither evaluate all
possible joint structure nor establish jackpot probability, payouts or ROI.
The record remains `prediction_edge: NOT_ESTABLISHED` even if the statistical
threshold is crossed. [Method, proof, limitations and commands](../docs/sportka_prospective.md).

## Historical theta-grid context

Independently inspected the completed
[fixed-grid CI benchmark](https://github.com/DavJ/MachineLearning/actions/runs/39968666172)
at PR #17 head `6718c2aad4aadc9ea7ac2953b3cdbd4b79d20561`.
It uses the older 3,821-date snapshot through 2026-10-21, the same source hash
as the [historical audit](sportka_signal_audit_2026-10-23.md), and 566 test targets
from 2023-03-15. This is fixed toroidal feature extraction plus MLP, not an
adaptive Theta Grid architecture.

| Game | Fixed-grid test BCE | Uniform test BCE | Fixed-grid test Brier | Uniform test Brier |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0.37753909 | 0.37177618 | 0.10872371 | 0.10745522 |
| 2 | 0.37797201 | 0.37177618 | 0.10883277 | 0.10745522 |

Lower is better for both scores. Both grid fits reached the configured
150-iteration limit. These particular optimization-limited fits do not establish
a forecasting advantage; they do not rule out every theta model. They are not
promoted into the first prospective protocol.

## Verification

- 122 local pytest tests passed, including 28 new prospective tests and the
  existing neural training/persistence tests; 22 existing dependency warnings.
- Exact small-lottery enumeration checked the hit law and two-round adaptive
  null expectation of every registered evidence factor.
- Calibration checked even with perfectly dependent game/model outcomes.
- Synthetic perfect-hit and subsequent loss controls checked threshold crossing
  and retention of the running bound; large-evidence logs remained finite.
- Future-day deadlines, bonus exclusion, pending locks, no backfilling, source
  revisions, hash/recipe changes, missing results, concurrent writers and failed
  atomic replacement were checked.
- The first real issue command completed against the fresh official export.
- No betting or scheduled automation was created.
