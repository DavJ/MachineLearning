# Sportka signal audit — 23 October 2026

No corrected historical deviation was detected in the specified single-number,
within-draw pair or row-ordering tests. No prediction advantage is established.
This does not prove randomness or exclude weak, different or time-varying signals.

## Data and fixed audit

- [Official Allwyn CSV](https://www.allwyn.cz/loterie/historie-cisel?game=sportka), freshly downloaded and validated.
- 3,821 chronological dates per draw, 1994-01-09 through 2026-10-21; six main balls only.
- CSV SHA-256: `631ecac60a8884a96a5331675972edc336f07e826c72eea207f66edc8442d5d6`.
- Each draw: 49 single numbers, all 1,176 pairs, and 200 repeat/overlap statistics across lags 1, 2, 5, 10.
- 1,999 uniform Monte Carlo samples for the frequency/pair maxima and 1,999 row permutations per real ordering test; seed 42.
- Maximum-statistic adjustment within each family, then Bonferroni across six draw/family combinations, at alpha=0.05.
- Uniform subset sampling retains the without-replacement lottery structure. Ordering permutations preserve the complete observed draw multiset.
- All historical periods have already been available to project research. This is exploratory, not prospective validation.

## Real-data results

| Draw | Family | Coordinates | Within-family max-statistic p | Across-six adjusted p | Rejected at 0.05 |
| --- | --- | ---: | ---: | ---: | --- |
| 1 | numbers | 49 | 0.4315 | 1.0000 | no |
| 1 | pairs | 1176 | 0.5665 | 1.0000 | no |
| 1 | ordering | 200 | 0.4795 | 1.0000 | no |
| 2 | numbers | 49 | 0.5265 | 1.0000 | no |
| 2 | pairs | 1176 | 0.8670 | 1.0000 | no |
| 2 | ordering | 200 | 0.3135 | 1.0000 | no |

The first p-value already accounts for searching all coordinates within that
family. The second also accounts for all six requested families. Top coordinate
counts are retained in the JSON for inspection; they are not recommendations
of numbers to play. Even a rejection would identify a historical discrepancy,
not its mechanism or profitability. Pair anomalies can reflect marginal bias;
ordering anomalies can reflect regime changes or data artifacts.

## Synthetic sensitivity, same history length

Each row below uses 64 independent synthetic histories of length 3,821. The
**specified family** is tested with the same search correction as the real audit.
Ordering controls use 1,999 conditional permutations per trial, matching the
real test. These are detector controls, not neural-network training experiments.

| Control family | Mixture/copy strength | Interpretable planted effect | Family detections / 64 | Rate | Approx. Wilson 95% interval |
| --- | ---: | --- | ---: | ---: | --- |
| number | 0.000 | Number 1 inclusion = 12.245% (fair: 12.245%) | 0/64 | 0.0% | 0.0%–5.7% |
| number | 0.010 | Number 1 inclusion = 13.122% (fair: 12.245%) | 5/64 | 7.8% | 3.4%–17.0% |
| number | 0.025 | Number 1 inclusion = 14.439% (fair: 12.245%) | 40/64 | 62.5% | 50.3%–73.3% |
| number | 0.050 | Number 1 inclusion = 16.633% (fair: 12.245%) | 64/64 | 100.0% | 94.3%–100.0% |
| pair | 0.000 | Pair 1/2 inclusion = 1.276% (fair: 1.276%); individual marginals unchanged | 0/64 | 0.0% | 0.0%–5.7% |
| pair | 0.025 | Pair 1/2 inclusion = 1.550% (fair: 1.276%); individual marginals unchanged | 0/64 | 0.0% | 0.0%–5.7% |
| pair | 0.050 | Pair 1/2 inclusion = 1.824% (fair: 1.276%); individual marginals unchanged | 2/64 | 3.1% | 0.9%–10.7% |
| pair | 0.100 | Pair 1/2 inclusion = 2.372% (fair: 1.276%); individual marginals unchanged | 50/64 | 78.1% | 66.6%–86.5% |
| temporal | 0.000 | Copy previous full set with probability 0.0% | 1/64 | 1.6% | 0.3%–8.3% |
| temporal | 0.005 | Copy previous full set with probability 0.5% | 7/64 | 10.9% | 5.4%–20.9% |
| temporal | 0.010 | Copy previous full set with probability 1.0% | 39/64 | 60.9% | 48.7%–71.9% |
| temporal | 0.025 | Copy previous full set with probability 2.5% | 64/64 | 100.0% | 94.3%–100.0% |

For example, shifting one number from 12.245% inclusion to 13.122% was detected
by the frequency family in only 5/64 trials. Raising it to 16.633% was detected
in 64/64 trials. The negative real-data result therefore cannot rule out weak
biases. The coupled-pair controls also confirm that joint structure can be
detectable even when all individual-number marginals remain exactly 6/49.

The intervals condition on the shared uniform calibration and do not include
its Monte Carlo uncertainty. Zero-strength rows estimate the size of that single
corrected family, not the chance of any false discovery across all families.
These artificial alternatives do not represent every possible lottery mechanism.

## Verification and reproduction

- 94 local pytest tests passed in this independent audit branch, including 22 new audit tests.
- Existing TensorFlow training/persistence and experiment-runner tests passed.
- Actual count formulas were checked against explicit enumeration; temporal expectations were checked across every ordering of a four-row example.
- Strong known marginal, pair-only and temporal alternatives were detected in deterministic regression checks.
- Reusing the uniform calibration reproduced every real-data family result exactly; ordering permutations were rebuilt deterministically.
- `compileall` and `git diff --check` passed.

```bash
python -m sportka.experiments.signal_audit --csv /path/to/sportka.csv \
  --resamples 1999 --trials 64 --progress
```

Exact reproduction requires the same source snapshot; future Allwyn downloads
may add or revise rows. [Method and assumptions](../docs/sportka_signal_audit.md).
[Full JSON results](sportka_signal_audit_2026-10-23.json).

No test of cross-game dependence, every subperiod, machinery metadata or all
nonlinear patterns is included. Correction covers this invocation, not the
project's entire history of hypotheses. No betting or purchase integration is added.
