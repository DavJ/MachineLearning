# Joint-set prediction and signal-recovery controls

This experiment checks whether a small fitted forecasting pipeline can recover
known synthetic structure, and explores the same fixed pipeline on Allwyn data.
It is separate from the [historical audit](sportka_signal_audit.md) and does not
change the [registered prospective ledger](sportka_prospective.md).

## Why score a whole combination?

Marginal BCE and Brier correctly assess individual-ball probabilities. They do
not uniquely assess the joint law of a complete six-number set. For example,
the coupled-pair control below has exactly uniform individual marginals at every
mixture strength. Its marginal probability scores therefore equal the uniform
model's scores, even when the full combination distribution is different.
Expected main-ball hits for any fixed six-number set also remain `36/49` in
this IID control. An average-hit test does not describe every possible form of
predictability or the upper tail of match counts.

The new primary score is the mean held-out log-likelihood gain

\[
 \Delta=\frac1m\sum_{t=1}^m\log\frac{Q(S_t\mid\text{past})}{U(S_t)},
 \qquad U(S)=1/\binom{49}6.
\]

Positive gain gives lower joint negative log loss than uniform on those targets.
The code also reports marginal BCE, Brier and mean main-ball hits, without using
them to select the winning family. Joint log score is a proper scoring rule for
the full outcome distribution; that fact does not make a finite positive score
statistically conclusive or establish profitable betting. See Gneiting and
Raftery (2007), [Strictly Proper Scoring Rules, Prediction, and Estimation](https://sites.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf),
*JASA* 102(477), 359–378, DOI `10.1198/016214506000001437`.

## Four coherent candidate laws

All nonuniform candidates use `Q=(1-s)U+sA`, with `0 <= s <= 0.95` so every
valid six-set retains positive probability. Number and pair identities are
chosen by maximum likelihood on training targets only.

| Candidate | Alternative law A | Training search |
| --- | --- | --- |
| Uniform | Uniform six-of-49 subsets | None |
| Number | Uniform conditional on including ball b | All 49 b, positive strength |
| Coupled pair | Include both b/c with probability 6/49 and choose four of the other 47; otherwise include neither and choose six of the other 47 | All 1,176 pairs, positive strength |
| Exact copy | Point mass on the immediately previous six-set | Copy probability |

These are simple, intentionally matched control families, **not neural or
adaptive Theta Grid models**. They are not a universal search over mechanisms.
For the pair alternative, b and c each have inclusion probability `6/49`; any
other ball has probability

\[
 (6/49)(4/47)+(43/49)(6/47)=6/49.
\]

The joint likelihood ratios `A/U` are `49/6` or zero for a number; `48/5`, zero,
or `8/7` for both, exactly one, or neither pair member; and `C(49,6)` or zero
for an exact copy. These identities allow exact, finite scoring without treating
49 ball labels as independent. Pair profile likelihoods are concave and fitted
with vectorized bounded bisection; the other strengths have closed-form MLEs.
Zero strength is uniform. Uniform wins exact validation ties. Any reported
maximum-probability set is a deterministic diagnostic, not a recommended bet.

## Chronology, selection and calibration

For N chronological source rows, all candidates share N−1 targets. Row t may use
row t−1, with no future features. Fit only the first `floor(0.70*(N−1))`
targets. Select among the four fixed candidates on the following targets through
`floor(0.85*(N−1))`. Evaluate the selected, frozen fit on the final targets once.
Validation rows are never added to training. Scores for the other test models
are diagnostic; no test-score winner is substituted for the validation choice.

Each uniform Monte Carlo history independently reruns **all training identity
searches, strength estimation, validation selection and test scoring**. The
reference statistic is the selected candidate's mean test gain. A conservative
rank p-value `(1 + count(reference >= observed))/(B+1)` includes ties and cannot
be zero. Bonferroni also covers all requested real games; a rejection additionally
requires positive test gain. Adequate Monte Carlo resolution is checked before
loading data. Complete reference values and selection counts are saved in JSON.

This calibration covers the fixed current pipeline, not the researcher's earlier
or later choice of model classes, scores, strengths, periods or experiments.
All real historical rows have already been examined by project research, so a
rejection would be exploratory and require fresh replication.

## Recovery simulations

At the same source length, use 64 independent histories per default level:
uniform; positive single-number strengths `0.01, 0.05, 0.15`; coupled-pair
strengths `0.05, 0.10, 0.25`; exact-copy probabilities `0.005, 0.01, 0.025`.
The existing audit generators supply exactly six distinct main balls per row.
Separate random streams isolate null calibration from all control histories.

Each trial trains, selects and scores the full fitted pipeline. A separately
labelled oracle knows the artificial law and its true parameters; it is never
a fitted or selectable candidate. Oracle scores clarify what signal exists
even when training cannot identify it. The uniform row estimates rejection for
one pipeline using the same across-game correction, not an empirical two-game
familywise error rate. Wilson intervals condition on the shared calibration and
do not include its Monte Carlo uncertainty. Exact-copy controls are a particularly
easy rare-outcome alternative and must not be generalized to arbitrary weak
dependencies. Negative results do not exclude weaker or different signals.

## Reproduction and scope

```bash
python sportka_launcher.py download --directory data
OPENBLAS_NUM_THREADS=1 python -m sportka.experiments.joint_signal_recovery \
  --csv data/sportka.csv --resamples 999 --trials 64 --progress
```

`--game 1|2|both`, `--seed` and `--no-power` are explicit controls. JSON includes
source SHA-256, NumPy version, splits/dates, all fitted parameters, validation
scores, selected test scores, null reference and recovery summaries. Changed
inputs/configuration describe a different experiment. There is no synthetic
fallback when a CSV is missing, no checkpoint promotion and no new dependency.

The original registered forecasts and alpha budget remain unchanged. No future
joint model is silently inserted into that ledger. This experiment never sets
`prediction_edge` to established; neither likelihood gains nor model-implied
set probabilities are measured jackpot, Superjackpot, payout or ROI advantages.

[Recorded run and verification](../reports/sportka_joint_recovery_2026-10-24.md).
