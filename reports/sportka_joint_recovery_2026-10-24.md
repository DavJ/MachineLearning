# Joint-set signal recovery — 24 October 2026

**No corrected predictive gain was detected on the real Allwyn history.**
The fitted pipeline can exploit several known synthetic alternatives, including
a coupled-pair law whose individual-ball probability scores remain uniform.

## Source and fixed procedure

- Official source: https://www.allwyn.cz/loterie/historie-cisel?game=sportka.
- 3,822 dates per game, 1994-01-09 through 2026-10-23; six main balls only.
- CSV SHA-256: `ff89e477192390abfb0ae81892ae824c1eaeefa6e86dc90204949863358f2cbd`.
- Shared targets: 3,821; training 2,674, validation 573, test 574.
- Train: 1994-01-16–2019-02-24; validation: 2019-02-27–2023-02-24;
  test: 2023-02-26–2026-10-23. These already-examined historical dates are exploratory.
- Four coherent set laws: uniform, one positively biased number, coupled pair,
  exact previous-set copy. Search all 49 numbers and 1,176 pairs on training only.
- Select the family using validation joint log likelihood; freeze the training fit.
- 999 independent uniform histories repeat the complete fit/search/selection/test
  procedure; Bonferroni also covers both real games. Seed 42, NumPy 2.3.5.
- No test-score winner is substituted for the validation-selected candidate.

## Real-data selected models

| Game | Validation-selected family | Test joint gain, nats/draw | Pipeline p | Across-game adjusted p | Rejected at 0.05 |
| --- | --- | ---: | ---: | ---: | --- |
| 1 | number | +0.000798093 | 0.067 | 0.134 | no |
| 2 | number | -0.001403671 | 0.798 | 1.000 | no |

Positive joint gain is better than uniform on those particular targets. The
small game-1 gain is not significant after calibration; game 2 is worse. Uniform
joint negative log loss is 16.453411219 in both games. The selected losses are
16.452613126 and 16.454814889 respectively. Neither result establishes a lottery edge.
All fitted parameters and unselected model diagnostics are retained in the JSON.

Uniform-reference validation selected uniform in 567/999 simulations, number
in 241/999 and pair in 191/999; copy was never selected. A nonuniform validation
winner can therefore occur under a fair lottery. Its full test distribution,
rather than the mere fact of selection, is the reference for the reported p-value.

## Synthetic recovery at the same history length

Each row uses 64 independent histories and the same training, validation and
test sizes. Detection means a positive selected test gain with corrected
p <= 0.05. Correct structure selection additionally requires the true number
or pair identity, where applicable. The uniform row is a one-game false-positive
control using the two-game correction, not a direct two-game familywise estimate.

| Artificial law | Strength | Interpretable effect | Detections / 64 | Rate | Conditional Wilson 95% interval | Correct structure selected / 64 |
| --- | ---: | --- | ---: | ---: | --- | ---: |
| uniform | 0.000 | Fair independent six-sets | 2/64 | 3.1% | 0.9%–10.7% | 38/64 |
| number | 0.010 | Ball-1 inclusion 13.122% (fair 12.245%) | 4/64 | 6.2% | 2.5%–15.0% | 11/64 |
| number | 0.050 | Ball-1 inclusion 16.633% (fair 12.245%) | 51/64 | 79.7% | 68.3%–87.7% | 59/64 |
| number | 0.150 | Ball-1 inclusion 25.408% (fair 12.245%) | 64/64 | 100.0% | 94.3%–100.0% | 64/64 |
| pair | 0.050 | Pair-1/2 inclusion 1.824%; all single marginals fair | 2/64 | 3.1% | 0.9%–10.7% | 8/64 |
| pair | 0.100 | Pair-1/2 inclusion 2.372%; all single marginals fair | 31/64 | 48.4% | 36.6%–60.4% | 46/64 |
| pair | 0.250 | Pair-1/2 inclusion 4.018%; all single marginals fair | 64/64 | 100.0% | 94.3%–100.0% | 64/64 |
| temporal | 0.005 | Copy previous whole set with probability 0.5% | 55/64 | 85.9% | 75.4%–92.4% | 61/64 |
| temporal | 0.010 | Copy previous whole set with probability 1.0% | 64/64 | 100.0% | 94.3%–100.0% | 64/64 |
| temporal | 0.025 | Copy previous whole set with probability 2.5% | 64/64 | 100.0% | 94.3%–100.0% | 64/64 |

The pair-strength 0.25 control was detected in 64/64 trials, while its oracle
marginal BCE gain is **exactly zero**. The oracle mean joint gain is 0.026341164
nats/draw. This illustrates why a correct individual-ball forecast can still miss
joint structure. The oracle knows the artificial law and its parameters; it is
never a fitted or selectable candidate. Strong-control recovery does not imply
equal power for real, weak or differently structured signals.

At number strength 0.01, the pipeline detected only 4/64 histories; at pair
strength 0.05, only 2/64. These negative results do not exclude all weak effects.
Exact-copy controls are especially easy rare-outcome alternatives, so their
high detection rates must not be generalized to arbitrary temporal dependence.
All intervals condition on the shared calibration and exclude its Monte Carlo
uncertainty. No single sample size guarantees detection.

## Verification and scope

- **150 local pytest tests passed**, including 28 new joint-recovery tests and
  existing TensorFlow training/persistence checks; 22 existing dependency warnings.
- Independently enumerated every 3-of-8 outcome to verify normalization,
  marginals and maximum-probability sets for all four laws.
- Independently checked pair MLE against a scalar likelihood grid.
- Mutating validation or test targets cannot change training parameters;
  mutating test targets also leaves validation selection unchanged.
- Independent CSV counts and scalar log-likelihood arithmetic reproduced both
  real primary gains and corrected Monte Carlo p-values.
- Missing-source, invalid-prior, finite-support, tie-breaking and whole-pipeline
  resampling contracts passed.
- The prospective ledger is byte-identical to master, with the original pending
  2026-10-25 forecasts. Its registered models and alpha budget are unchanged.

These are simple probabilistic mixture models, not neural or adaptive Theta Grid
models. Correction covers this fixed pipeline and the requested games; it does
not correct the entire project research history. No jackpot, Superjackpot, payout
or profitability advantage has been demonstrated. No betting or scheduling is added.

[Full JSON, including all 999 reference statistics](sportka_joint_recovery_2026-10-24.json).
[Laws, assumptions and reproduction command](../docs/sportka_joint_recovery.md).
Exact reproduction requires the same input snapshot, configuration and compatible
numerical environment; future downloads can add or revise rows.
