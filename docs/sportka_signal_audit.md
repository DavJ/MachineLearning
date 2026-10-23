# Sportka signal audit and sensitivity controls

This research tool checks specified departures from a fair 6/49 model. It does
not choose tickets, estimate profits or establish a prospective prediction edge.
It evaluates the six main balls; the bonus is excluded.

```bash
python sportka_launcher.py download --directory data
OPENBLAS_NUM_THREADS=1 python -m sportka.experiments.signal_audit \
  --csv data/sportka.csv --resamples 1999 --trials 64 --progress \
  > reports/local_signal_audit.json
```

The default audits both draws, four fixed lags and all three families below.
It also runs synthetic sensitivity controls. The controls use separate generated
arrays and never modify the real CSV. They measure the **audit detectors**, not
neural-network training or ticket profitability. Full sensitivity calibration
can take several minutes. `--no-power` runs only the historical audit.

## Prespecified families

| Family, per draw | Coordinates | Reference distribution |
| --- | ---: | --- |
| Single-number frequencies | 49 | Independent uniform six-element subsets of 1..49 |
| Within-draw pair frequencies | 1,176 | The same without-replacement lottery simulator |
| Ordering: repeat counts by number and total overlap at lags 1, 2, 5, 10 | 200 | Random permutations of the complete observed draw multiset |

Lags count successive dates in the export, not calendar days. The two Sportka
draws are analysed separately; this version does not test cross-game dependence,
every historical subperiod, machinery metadata or all possible nonlinear patterns.

For an individual number, the uniform inclusion probability is `6/49`. For a
specified unordered pair it is `6*5/(49*48)`. Across `n` independent uniform draws,
each coordinate's count has expectation `n*p` and variance `n*p*(1-p)`. The
absolute count deviation is scaled by this standard deviation. Correlations
between the 49 numbers and 1,176 pairs are retained by simulating whole valid
draws; there is no independent-balls or naive multinomial approximation.

For ordering tests, let `c[j]` be a number's total count. Conditional on the draw
multiset, its expected repeat count at lag `l` under row exchangeability is
`(n-l)*c[j]*(c[j]-1)/(n*(n-1))`. Aggregate overlap is the sum over numbers.
Absolute deviations are divided by `sqrt(max(expected_count, 1))` only as a
scaling convention. **No normal or chi-square distribution is assumed for these
temporal scores.** The permutations calibrate their complete maximum statistic.

Permutations preserve every number and within-draw pair frequency. An ordering
rejection can nevertheless reflect regime changes or data artifacts rather than
a useful causal forecasting relationship. A within-draw pair rejection may be
explained by marginal bias and does not alone demonstrate a pair interaction.

## Search correction and interpretation

Each simulated/permuted sample contributes the largest absolute scaled
deviation in its family. The observed maximum is compared with these maxima:

`p = (1 + count(reference_max >= observed_max)) / (resamples + 1)`.

Ties count as exceedances, including a conservative floating-point tolerance.
P-values never become zero. The top coordinates are localisations of a global
model discrepancy, not conditional discoveries after explaining other biases.

The family-level p-value is then Bonferroni-adjusted across all requested
draw/family combinations: `min(1, family_count*p)`. With both draws there are six
families, covering all 2,850 coordinates inside those groups. Their dependence
does not invalidate the Bonferroni bound. With 1,999 replicates the smallest
overall adjusted p-value is `6/2000 = 0.003`.

The configured level is 0.05. Correction covers **this invocation's specified
families and lags**, not every earlier or future analysis, adaptive model search,
or data-driven selection of historical periods. The historical export has already
been inspected in earlier project work, so the audit remains exploratory.
Rejection identifies a historical discrepancy requiring explanation and new-data
replication. Non-rejection does not prove randomness or exclude weak signals.

Method references:

- [NIST: Bonferroni's method](https://www.itl.nist.gov/div898/handbook/prc/section4/prc473.htm)
- [SciPy: Monte Carlo testing and finite-precision ties](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.monte_carlo_test.html)
- Phipson & Smyth (2010), cited in the SciPy reference: randomly sampled permutation p-values should not be zero.

## Planted alternatives and sensitivity

Each level runs 64 independent synthetic histories of the same length as the
real data. The tested family's rejection uses the same maximum-statistic and
across-family correction as the real audit. By default the temporal controls use
the same 1,999 permutations per trial as the actual ordering test. An explicit
`--power-permutations` override changes that Monte Carlo resolution and must be
accounted for when interpreting the sensitivity estimate.

| Artificial alternative | Preset mixture/copy probabilities | Meaning |
| --- | --- | --- |
| Number | 0, 0.01, 0.025, 0.05 | Force number 1 into that fraction of rows, replacing one ball when needed |
| Pair | 0, 0.025, 0.05, 0.10 | Mix in a law with coupled numbers 1/2 and unchanged individual marginals |
| Temporal | 0, 0.005, 0.01, 0.025 | Copy the preceding six-number set with that probability; otherwise use a fresh uniform draw |

For number bias at strength `s`, number 1's inclusion probability is
`6/49 + s*(1-6/49)`. Strength is **not** an additive percentage-point bias.

In the coupled-pair component, numbers 1/2 occur together with probability
`p=6/49`, with four other balls chosen uniformly from the remaining 47. Otherwise
both are absent and six balls are chosen from the remaining 47. Numbers 1/2 each
have marginal `p`; every other number has marginal
`(4*p+6*(1-p))/47 = 6/49`. Thus the mixture changes joint structure while every
single-number marginal remains fair. Its specified pair probability is
`30/(49*48) + s*(6/49-30/(49*48))`.

The temporal control has uniform stationary one-draw marginals and within-draw
pair probabilities, but a dependence on the preceding draw. It is an artificial
detector check, not a claimed mechanism in Allwyn's physical lottery.

Output includes detections/trials and approximate Wilson 95% intervals. For the
frequency/pair controls, intervals condition on the shared simulated calibration
and do not include its Monte Carlo uncertainty. At zero strength the reported
rate estimates that family's test size, not the probability of any false
discovery across all families. A few dozen trials do not certify a false-positive
rate or power against untested alternatives.

## Reproducibility and persistence

The JSON records the CSV SHA-256, dates, sample size, seed, lags, alpha, Monte
Carlo counts, top count deviations and all sensitivity levels. The uniform-null
calibration is reused for matching sample size, replicate count, seed, NumPy
version and audit version. It is saved atomically under `models/sportka_audit` as
an NPZ loaded with `allow_pickle=False`. This is a simulated reference, not a
trained predictor. Conditional ordering references are rebuilt for each history.

Missing/invalid files fail rather than becoming synthetic data. Inputs with
too few resamples to reach the corrected significance threshold are rejected.
Progress goes to stderr and JSON to stdout; no account access or betting occurs.

## Recorded run

[23 October 2026: both draws and full sensitivity results](../reports/sportka_signal_audit_2026-10-23.md).
