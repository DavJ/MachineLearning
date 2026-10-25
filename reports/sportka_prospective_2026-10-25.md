# First prospective result — 25 October 2026

**One advance-published date scored. Best ticket comparison: 2/6 main hits.
No registered combination reached three main hits in either draw. No prospective
predictive advantage is established.**

## Official results and source

The [official Allwyn CSV](https://www.allwyn.cz/loterie/historie-cisel?game=sportka)
was retrieved on 25 October at 21:26:52 Prague time (HTTP date 20:26:52 UTC).
It contains 3,823 complete dates through 2026-10-25. CSV SHA-256:
`8c90255d3b6b78efa86c0f7d106f9140dbae752bf37c1119e7769b5c20885315`.

| Draw | Six main balls | Bonus, excluded from main-hit counts |
| --- | --- | --- |
| 1 | 3, 9, 14, 23, 25, 45 | 8 |
| 2 | 11, 27, 33, 35, 43, 45 | 4 |

## Every sealed set checked against both draws

These are the original sets from the
[23 October issue](sportka_prospective_2026-10-23.md); none was changed after
the result. A six-number ticket can be compared with both draws.

| Recipe / originally assigned draw | Sealed numbers | Draw 1 matches | Draw 2 matches |
| --- | --- | --- | --- |
| All history / 1 | 6, 18, 20, 29, 30, 47 | 0 | 0 |
| All history / 2 | 12, 13, 23, 36, 40, 44 | 1: **23** | 0 |
| Last 52 / 1 | 9, 11, 13, 27, 29, 42 | 1: **9** | 2: **11, 27** |
| Last 52 / 2 | 1, 5, 20, 22, 40, 42 | 0 | 0 |
| Previous draw / 1 | 17, 21, 30, 38, 41, 43 | 0 | 1: **43** |
| Previous draw / 2 | 3, 11, 15, 30, 41, 47 | 1: **3** | 1: **11** |

The best cross-draw comparison is the last-52 set originally assigned to draw 1,
which matches 11 and 27 in draw 2. These descriptive comparisons do not add
statistical experts or change the registered decision rule.

## Registered sequential evaluation

The existing protocol evaluates each forecast against its originally assigned
draw. Its main-hit counts are all-history **0 / 0**, last-52 **1 / 0**,
previous-draw **0 / 1** (draw 1 / draw 2).

- Scored dates: **1**; no pending forecast remains.
- Mean of the 42 registered persistent evidence processes: **0.8757266078490182**.
- Running maximum evidence: **1.0**, retaining the initial value.
- Anytime p-value bound: **1.0**; rejection threshold: **20**.
- Verdict: `NO_PROSPECTIVE_EVIDENCE`; `prediction_edge: NOT_ESTABLISHED`.

One date is insufficient to assess long-run predictive ability. This report
does not estimate payouts or profit. The ledger retains this result under the
original recipes, tilt grid and alpha budget; no next forecast was issued.

## Integrity and independent checks

GitHub reports [PR #18](https://github.com/DavJ/MachineLearning/pull/18) merged at
**2026-10-23 20:23:26 UTC** into commit
`1fcbabec14aa22bf042e5e13d072656496e1f814`. Its
[merged ledger](https://github.com/DavJ/MachineLearning/blob/1fcbabec14aa22bf042e5e13d072656496e1f814/reports/sportka_prospective_ledger.json)
has blob SHA `944a210ae8ff5563dbb67b4576696c0dbcb1ab13` and exactly matches the
original protocol and forecast event retained here. This hosting merge event
precedes the target-day deadline, 2026-10-25 00:00 Prague time
(2026-10-24 22:00 UTC, before that night's clock change). Publication timing was
checked using the GitHub event, rather than editable Git author timestamps.

The scoring command validated the original 3,822-date source prefix and recipes
against the fresh export, preserved the original forecast and protocol, and
appended exactly one result. Forecast event hash:
`912cddc298f8021c948cc569b1d01e2b9a427c71d5f5430e90700996478b6a09`.
Result event hash:
`10954ebe8f419b31f80d9a5a067b7a9931027b08703743a94f7b48b9e3582b7b`.

Independent stdlib CSV parsing and set intersections reproduced all main-hit
counts and bonus values. A separate exact hypergeometric calculation and
arithmetic mean of the 42 factors gave E = **0.8757266078490185**, within
`2.3e-16` of the production result. All six sets were also checked against both
draws, establishing the descriptive maximum of two main hits.

The CLI continues to require external-publication verification by design; this
report records that independent check for this issue. The
[JSON companion](sportka_prospective_2026-10-25.json) contains the provenance,
publication check, forecasts, actual outcomes, both types of comparisons and
complete sequential status. The [ledger](sportka_prospective_ledger.json) is
the append-only record.
