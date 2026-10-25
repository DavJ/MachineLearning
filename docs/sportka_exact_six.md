# Exact six-set scores for existing models

`sportka.fixed_cardinality` turns a row of independent Bernoulli scores into
an explicitly specified distribution over complete six-number sets. The shared
`compute_all_metrics` path now reports its whole-set log score and implied
inclusion scores alongside the original raw BCE/Brier. This also applies to
neural and theta experiments that call that evaluator. It does not retrain or
select a model, alter ticket generation, or modify the prospective protocol.

## The specified distribution

First clip each raw score `p[j]` in float64 to `[1e-9, 1-1e-9]`. This fixed,
versioned boundary rule gives every valid set positive probability, even when
a fitted model returns exact zeros or ones. It is part of the scoring model,
not a parameter tuned on held-out results.

Let `w[j] = p[j] / (1-p[j])`. For a six-element set S,

```text
Q(S) = product(w[j] for j in S) / e_6(w)
e_k(w) = sum(product(w[j] for j in T) for all k-element sets T)
```

This follows by conditioning independent Bernoulli variables on their sum
being six: the common product of `(1-p[j])` cancels. A row-wise common odds
multiplier also cancels. Consequently any equal-score row, even all 0.9,
induces the uniform six-set law. This is generally different from multiplying
probabilities by `6 / sum(p)`.

The prefix recurrence is `e_k(first j) = e_k(first j-1) + w[j] *
e_(k-1)(first j-1)`, initialized by `e_0 = 1`. The implementation uses log space
and row-centered log odds for stability. A backward suffix pass computes
`P(j in S) = w[j] * e_5(w without j) / e_6(w)`. These marginals sum to six and
generally differ from the input scores. Runtime is O(rows * pool * count),
without enumerating the 13,983,816 Sportka sets or using random samples.

For background on conditional Bernoulli models, see Sean X. Chen (2000),
[General Properties and Estimation of Conditional Bernoulli Models](https://doi.org/10.1006/jmva.1999.1872).
The recurrence above and exhaustive tests specify this implementation.

## Metrics and use

| Field | Definition | Units / direction |
| --- | --- | --- |
| `bce`, `brier` | Original raw-score marginal metrics | Mean per number; lower is better |
| `conditional_set_nll` | Mean `-log Q(actual six-set)` | Nats per complete draw; lower is better |
| `conditional_set_gain_vs_uniform` | `log C(49,6)` minus set NLL | Nats per complete draw; higher is better |
| `conditional_bce`, `conditional_brier` | Marginal scores using the implied conditional inclusion probabilities | Mean per number; lower is better |

The uniform complete-set NLL is `log C(49,6) = 16.45341121889615`, and its
gain against itself is zero up to floating-point roundoff. Whole-set NLL is
not on the same scale as BCE, which averages 49 separate binary scores.
Hit and ticket-rate metrics still use the original scores and tie policy.
Conditioning preserves the top-six order except ties introduced by clipping.

```bash
python sportka_launcher.py backtest --csv /path/to/sportka.csv --game 1 --include-ml
python sportka_launcher.py backtest --csv /path/to/sportka.csv --game 2 --include-ml
```

The JSON records version `conditional-bernoulli-v1`, the boundary clip and
metric semantics. The v1/v2/null experiment Markdown tables also display
conditional set NLL and gain. The legacy null runner's exploratory decision
rule is unchanged; the extra descriptive columns are not new hypothesis tests.

For a model that already produces a valid complete-set law, score that law
directly. Do not recover a different law from its marginals using this adapter.
In particular the separate joint-recovery experiment's coupled-pair mixture
must retain its own likelihood. Conditional independent scores cannot express
arbitrary pair interactions; fair individual marginals alone do not determine
the true six-set law.

```python
from sportka.fixed_cardinality import FixedCardinality

law = FixedCardinality(scores)  # nonempty (rows, 49) matrix in [0, 1]
per_draw_log_probability = law.log_prob(binary_six_main_targets)
conditional_inclusion_probabilities = law.marginals()
```

Positive descriptive gain on previously studied history is not a corrected
significance result, verified calibration, an increased jackpot probability,
or an estimate of profit. Those claims require their own evidence. No model
is added to the already registered prospective forecasts.

## Verification

Tests enumerate every outcome of a 3-of-8 lottery, also covering empty/full
sets and other counts. Independent Bernoulli-product arithmetic checks exact
normalization, marginals and maximum-probability sets. Additional checks cover
a closed-form one-favored-ball 6-of-49 law, common odds scaling, row/column
permutations, batch independence, input mutation, malformed arrays, and float32
zeros/ones. CLI and all three report runners exercise the shared integration.

The [dated real-data check](../reports/sportka_exact_six_2026-10-24.md) records
the source snapshot, fixed model settings and both validation/test scores.
