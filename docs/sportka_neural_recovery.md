# Can the neural pipeline recover a known signal?

`sportka.experiments.neural_recovery` checks the actual raw/theta/Möbius/Mertens
MLP pipeline on artificial histories with known structure. This separates the
ability of the configured model and validation procedure to recover these
particular alternatives from the unresolved question of a real Sportka signal.
It is not a proof that every possible signal would be detected.

## Frozen fit, validation and test

The experiment reuses `theta_mobius.dataset`, `model_config` and `train_or_load`.
All four variants have 147 past-only features, a 52-draw warmup, hidden size 32,
Adam, alpha 0.1, 150 maximum iterations and model seed 42. These settings are
not searched. Targets after the warmup are split chronologically 70/15/15.
Only training rows are fitted. Iteration-limit warnings are material diagnostics,
not evidence of convergence.

Each sigmoid output vector specifies a positive complete-six-set conditional
Bernoulli law Q using the [exact-six adapter](sportka_exact_six.md). The boundary
clip is fixed at 1e-9. To limit overconfidence, validation chooses from
`Q_eta = (1-eta) U + eta Q`, with eta in `0, .01, .03, .1, .3, 1` and U uniform
over all six-number sets. This is a mixture of complete-set distributions,
not an arithmetic rescaling of independent inclusion scores.

Uniform is the initial choice. The family and weight with the greatest
validation mean log gain are frozen; differences within 1e-12 retain the earlier
choice, so uniform wins ties. Only after this choice is fixed does the code ask
the selected model for test predictions. There is no validation refit or test
selection. Each test prediction uses only earlier observed draws.

## Evidence and its scope

For each test target, calculate `g[t] = log(Q_eta(actual[t]) / U(actual[t]))`.
The terminal likelihood ratio is `E = exp(sum(g))`; report its log even when
the exponential is outside the safe numerical range. Mean gain is nats per
whole draw; positive is descriptively better than uniform.

Under genuinely fresh draws uniform **conditional on all prior information**,
each factor has conditional mean one because Q_eta is normalized. This remains
true when past observations determine features and when earlier validation
data determine the family and weight. Iterated conditional expectation gives
`E0[E] = 1`; Markov's inequality gives `P0(E >= 40) <= .025` for one game.
A union bound covers either of two games at .05, without assuming independence
between games. Family/weight selection requires no extra test-data search
penalty because it precedes all test outcomes under this fresh-data premise.
The numerical tests also enumerate an adaptive two-step small-lottery example.

The actual historical Sportka data have already been explored repeatedly.
Their reported evidence values are therefore **exploratory**, not confirmatory
p-values. This calculation does not cover choosing this experiment after earlier
research or changing its protocol after seeing results. It does not promote a
model, estimate returns, write a prospective ledger or restart an alpha budget.
The already sealed prospective protocol remains separate.

## Known artificial alternatives

The existing audit's generators always produce six distinct main numbers:

| Kind | Fixed strength levels | Mechanism |
| --- | --- | --- |
| Uniform | 0 | Independent uniform six-sets; false-crossing control |
| Number | .01, .05, .15 | Mix uniform with a set forced to contain ball 1 |
| Pair | .25 | Balls 1/2 coupled, with every individual marginal still 6/49 |
| Temporal | .1, .25 | Mix a fresh uniform set with copying the entire previous set |

Every level uses 16 independent histories by default, each as long as the source
CSV. Each reruns the whole four-model fit, validation selection and test score.
Independent streams use `SeedSequence([seed, n, 901, level, trial])`. Family
selection counts, all individual trial results, crossings of 40 and Wilson 95%
intervals are retained. These finite intervals concern repeated matched
artificial histories with this frozen pipeline. A one-game fair-control rate is
not an empirical two-game family-wise error estimate. Exact-copy alternatives
are special rare-event signals and do not represent all temporal structure.

A separately labelled oracle knows the planted mechanism and parameters and
computes its exact complete-set law. It never trains, selects or changes the
neural model. Comparing oracle and fitted recovery exposes estimation or model
limitations. In particular fair-marginal pair dependence cannot generally be
represented by conditional independent sigmoid outputs. A static pair signal
need not be recovered even when an oracle can score it.

## Run and resume

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python -m sportka.experiments.neural_recovery --csv /path/to/sportka.csv \
  --cache-dir models/neural_recovery --trials 16 --workers 2
```

Use only a checkpoint directory you control: joblib loads executable Python
objects. Matching source rows, feature family, configuration, libraries and
training arrays reuse the saved model; no new fitting is needed for matching
checkpoints after an interrupted run. The source identity includes the full
main-ball matrix and this experiment's version. Changed inputs create another
identity; this is not incremental optimization on changed data.
Unreadable checkpoints fail with an explicit rebuild instruction and are not
silently trusted. `--fresh` rebuilds all requested models. New checkpoint files
are flushed before their atomic replacement; this is not a guarantee against
external storage loss.

`--no-power` runs just the two historical games. `--seed` changes only synthetic
histories, not the fixed neural initialization. `--workers` accepts 1–4 workers;
results are recorded in stable level/trial order regardless of completion order.
Missing or invalid real files fail without a synthetic fallback.

[Dated experiment and limitations](../reports/sportka_neural_recovery_2026-10-26.md).
