# Neural signal recovery — 26 October 2026

**The frozen neural pipeline recovers some known alternatives but misses others.
Both historical Sportka scores improve slightly over uniform after validation
shrinkage; neither reaches the reference threshold, and the already-explored
history cannot supply confirmatory evidence. No lottery edge is established.**

## Source and frozen procedure

Reused the official Allwyn snapshot retrieved on 25 October at 21:26 Prague time:
3,823 dates per game, through 2026-10-25, SHA-256
`8c90255d3b6b78efa86c0f7d106f9140dbae752bf37c1119e7769b5c20885315`.
No new download was needed for this controlled comparison.

The four existing feature variants are raw, theta, Möbius and Mertens, with
147 past-only inputs and a 52-draw warmup. Reused hidden=32, Adam, alpha=0.1,
max_iter=150, seed=42. The chronological split is 2,639 training / 566 validation /
566 test targets; test dates are 2023-03-19 through 2026-10-25. Models fit training
rows only. Validation chooses the feature family and complete-set mixture
weight from `0, .01, .03, .1, .3, 1`; uniform wins ties. The choice is frozen
before the selected model is asked for test predictions.

Each neural output defines a normalized conditional Bernoulli six-set law Q,
with fixed 1e-9 boundary clipping. Its selected mixture is `(1-eta) U + eta Q`,
where U is uniform over all six-sets. This is not a new independent-probability
calibration claim. See the [method](../docs/sportka_neural_recovery.md).

## Historical results

| Game | Validation choice | Neural mixture weight | Test gain, nats/draw | Terminal E |
| --- | --- | ---: | ---: | ---: |
| 1 | Mertens | .03 | +0.001677508 | 2.584339 |
| 2 | Raw | .03 | +0.000472907 | 1.306910 |

Uniform NLL is 16.453411219 nats/draw. Selected NLLs are 16.451733711 and
16.452938312. Neither E value reaches 40. A selected weight of .03 means that
97% of the probability law remains uniform; the improvement should not be
described as a strong neural forecasting result. All eight fits reached the
150-iteration limit, so these remain optimization-limited configurations.

For genuinely fresh test draws under the conditional uniform null, normalization
gives E0[E]=1 and hence P(E>=40)<=.025 per game, or <=.05 for either of two games
by a union bound. The historical figures above remain **exploratory** because
this history and many other model ideas have already been examined. These
values are not confirmatory p-values or a correction for the project's history.

## Known artificial controls

Every row below uses 16 independent histories of 3,823 draws and repeats
validation selection and test scoring with four models fitted only on each
history's training prefix; matching cached fits are reused. The oracle knows
the planted mechanism and its parameters, but never takes part in fitting or
selection. Crossings refer to terminal E>=40.

| Known mechanism | Strength | Fitted-pipeline crossings | Wilson 95% interval | Oracle crossings |
| --- | ---: | ---: | --- | ---: |
| Fair uniform | 0 | 0/16 | 0–19.4% | 0/16 |
| Forced-number mixture | .01 | 0/16 | 0–19.4% | 0/16 |
| Forced-number mixture | .05 | 0/16 | 0–19.4% | 8/16 |
| Forced-number mixture | .15 | 11/16 | 44.4–85.8% | 16/16 |
| Coupled pair, fair individual marginals | .25 | 0/16 | 0–19.4% | 15/16 |
| Copy entire previous set | .10 | 16/16 | 80.6–100% | 16/16 |
| Copy entire previous set | .25 | 16/16 | 80.6–100% | 16/16 |

The forced-number levels imply inclusion probabilities of 13.12%, 16.63% and
25.41%, versus the fair 12.24%. The strongest number control is a substantial
effect, not a weak lottery-sized promise. The temporal controls repeat a whole
previous outcome in 10% or 25% of draws and are unusually explicit signals.
Raw features won validation in all 32 temporal trials; the spectral transforms
did not provide an observed advantage for these copying mechanisms.

The pair control retains exactly fair single-ball marginals. The conditional
independent output family cannot generally represent this positive coupling.
Its failure contrasts with a knowledgeable joint-law oracle, and demonstrates
why a negative neural result cannot certify randomness. The oracle has advance
knowledge of the artificial law; its rates are a diagnostic benchmark, not a
fair competitor trained on the same limited data.

All 448 control fits also reached 150 iterations. Recovery rates therefore
describe this fixed fitting budget and selector. Sixteen trials give wide
uncertainty; 16/16 does not establish universal 100% power, and 0/16 fair
crossings do not by themselves validate a two-game false-positive rate.

## Verification and scope

- 199 local tests passed; three optional TensorFlow tests skipped; eight warnings
  (four existing plotting warnings and four expected iteration-limit warnings).
- Twenty-two new recovery cases cover coherent mixture normalization, adaptive
  two-step null expectation by exhaustive small-lottery enumeration, training
  isolation from test mutations, tie behavior, oracle category arithmetic,
  finite log evidence, invalid inputs, independent trial records and real
  checkpoint round trips without refitting.
- A new truncated-checkpoint case verifies explicit failure and successful
  deliberate rebuilding. One unreadable locally generated checkpoint from an
  interrupted run was quarantined and deterministically rebuilt; model settings
  and the experimental protocol were unchanged. New files are flushed before
  atomic checkpoint replacement.
- Independent probability-domain Bernoulli-sum normalization and ordinary
  mixture arithmetic reproduced both real mean gains within `1e-17` nats/draw.
- Model configuration, dependency versions, implementation hashes, selection
  diagnostics and every one of the 112 synthetic trials are in the
  [JSON companion](sportka_neural_recovery_2026-10-26.json).

This experiment uses the same saved-model machinery as PR #17 and reuses the
shared exact-six scorer. It is not the separate adaptive Theta Grid system.
The already published prospective forecast and the
[first result in PR #21](https://github.com/DavJ/MachineLearning/pull/21) are not
modified, restarted, or replaced by these historical choices. This work does
not select betting tickets or establish jackpot odds or profitability.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m sportka.experiments.neural_recovery \
  --csv /path/to/the/recorded/sportka.csv --trials 16 --workers 2
python -m pytest -q
```
