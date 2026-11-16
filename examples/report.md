# Synthetic knowledge-tracing experiment

Generated toy descriptors and simulated learners only. These results do not measure human learning.

Seed: 7; learners: 12; attempts per learner: 24.

| Model | Role | Log loss | AUC | Brier | ECE |
| --- | --- | --- | --- | --- | --- |
| oracle | synthetic_reference | 0.6441 | 0.6857 | 0.2262 | 0.1659 |
| base_rate | baseline | 0.6955 | 0.5000 | 0.2512 | 0.0370 |
| item_only | baseline | 0.7043 | 0.5286 | 0.2554 | 0.1979 |
| bkt | baseline | 0.7333 | 0.5270 | 0.2663 | 0.1495 |
| elo_irt | baseline | 0.7013 | 0.5506 | 0.2539 | 0.1231 |
| pfa | baseline | 0.7184 | 0.5475 | 0.2611 | 0.1827 |

Log loss and Brier are losses; lower is better. AUC measures ranking. ECE depends on binning and sample size; compare it cautiously.

The JSON report includes learner-level percentile intervals and paired log-loss differences against BaseRate. The reported bootstrap fraction is not a p-value.

## Limits

- Synthetic outcomes do not establish human learning effectiveness.
- Oracle reads simulator probabilities and is not a deployable predictor.
- A finite sample need not rank the oracle first.
- Fixed catalog and same learners; not a cold-start or unseen-learner evaluation.
- Bootstrap fractions are descriptive, not null-hypothesis-test p-values.
- Undefined AUC values and intervals are represented as null.
- No game policy, runtime benchmark, or production recommendation is produced.
