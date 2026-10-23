# Prospective Sportka checks

Historical hypotheses have already been examined repeatedly. A held-out score on
that history is not a fresh prospective result. This protocol seals forecasts
before a **future calendar day in Europe/Prague**, then scores only that exact
date when an explicit Allwyn CSV download contains its two complete main draws.
It neither places bets nor establishes a profitable model.

## Fixed first protocol

`sportka-prospective-v1` registers three transparent, deterministic baselines:

| Recipe | Forecast for each main draw |
| --- | --- |
| `frequency_all` | Six most frequent numbers in all available source rows |
| `frequency_52` | Six most frequent numbers in the last 52 source rows |
| `previous_draw` | The six main numbers from the last source row |

Frequency ties use ascending number order. Both games are sealed together.
The CSV must contain at least 52 complete dates. Bonuses never count as main
hits. These recipes were chosen as simple controls, not validated winners.
Neural/theta models from PR #17 are **not** included in this first protocol.

There is one pending date at a time. It cannot be replaced, cancelled or skipped
by these commands. A source can include unforecast intermediate dates, but those
are never backfilled into the score. The forecast stays based on its original
locked source prefix. Targets must increase, and issuing on the target day is
deliberately disallowed. The caller checks the official schedule; the program
does not guess regular or exceptional draw dates. A missing target among later
results is an error requiring investigation.

## Exact sequential evidence

For any six-number set fixed before a fair six-of-49 draw, the number of main
hits has the hypergeometric law

\[
 p_0(h)=\frac{\binom6h\binom{43}{6-h}}{\binom{49}6},\qquad
 h=0,\ldots,6,\quad \mathbb E_0H=36/49.
\]

For each registered model and game, maintain seven processes with fixed tilts
`0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0`. At each scored date multiply by

\[
 f_\lambda(H)=\frac{\exp(\lambda H)}
 {\sum_{h=0}^6p_0(h)\exp(\lambda h)}.
\]

Under the null of uniform draws **conditional on all prior information**, each
factor has conditional expectation one, even though the next forecast may use
past draws. Its persistent product starts at one and is a nonnegative test
martingale. Use the arithmetic mean of all **42** products (3 models × 2 games
× 7 tilts). This remains a test martingale, without assuming that models or the
two games on the same date are independent. Both games update in one date batch;
their products are never multiplied together.

For this single predeclared ledger, Ville's inequality gives

\[
 \Pr_0\{\sup_t E_t\ge20\}\le0.05.
\]

The report therefore uses the *mixture*, not the best model, tilt or date. Its
anytime p-value bound is `min(1, 1 / running_max(E))`. Inspecting repeatedly does
not require a new fixed sample-size test; losing observations remain in the
products. Passing the threshold is a discrepancy with this conditional fair-draw
null, not proof of a causal mechanism or profitable betting. `prediction_edge`
always remains `NOT_ESTABLISHED`. Logarithms avoid overflow; display values
outside the safe exponential range are `null` with their finite logs retained.

This positive-tilt test measures excess *mean main-ball hits*. It is not a
complete distributional test or a direct test of jackpots, payouts, ROI, or every
joint-number signal. The finite tilt grid and weak forecasting recipes limit
power; failure to cross the threshold cannot exclude all predictive signals.
No fixed sample count guarantees discovery. Adding a new model, restarting after
losses, running many alternative ledgers, or selectively publishing successes
would need additional predeclared search control. This ledger does not correct
the project's entire previous or future research history.

References: [SciPy's hypergeometric definition](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.hypergeom.html);
Shafer, Shen, Vereshchagin and Vovk (2011),
[Test Martingales, Bayes Factors and p-Values](https://arxiv.org/abs/0912.4269),
*Statistical Science* 26(1), 84–101, DOI `10.1214/10-STS347`.

## Commands and integrity

```bash
# Explicitly refresh and validate the official CSV.
python sportka_launcher.py download --directory data

# Example first target. On later dates choose a scheduled date after today.
python -m sportka.experiments.prospective issue --csv data/sportka.csv \
  --ledger reports/sportka_prospective_ledger.json --target-date 2026-10-25

# PUBLISH THE FORECAST COMMIT BEFORE ITS TARGET DAY.
# After results become available, refresh the CSV and resolve the pending date.
python sportka_launcher.py download --directory data
python -m sportka.experiments.prospective score --csv data/sportka.csv \
  --ledger reports/sportka_prospective_ledger.json

# Read-only, offline status; this does not fetch or verify new results.
python -m sportka.experiments.prospective status \
  --ledger reports/sportka_prospective_ledger.json
```

Each forecast records the UTC issue time, full-file CSV SHA-256, semantic history
prefix hash, source cutoff/count, predictor module hash, all six forecasts, and
previous event hash. Later commands recompute the old recipes and prefix hashes
against the new CSV and reject revisions, truncation or changed scored outcomes.
Formatting and within-draw number-order changes do not change the semantic hash.
A missing result leaves the ledger untouched; an actual result is appended once.
Atomic replacement preserves the previous file if writing fails. An exclusive
lock rejects concurrent writers; inspect a stale lock after a crashed process
before removing it manually.

**A local hash chain cannot prove creation time or prevent wholesale rewriting,
truncation, deletion, a forged system clock, or a fabricated source.** An external
version-history record is essential: verify that each complete forecast and
protocol were published before the target Prague calendar day and that every
subsequent result is retained. The CLI always labels external publication as
requiring verification; it does not treat a local timestamp as an attestation.
Use trusted hosting activity/receipt times, not user-editable Git author or
committer dates, to check advance publication.
The numerical threshold is conditional on this prospective timing and valid
data, not a substitute for checking them.

[First sealed forecast and current initial status](../reports/sportka_prospective_2026-10-23.md).
