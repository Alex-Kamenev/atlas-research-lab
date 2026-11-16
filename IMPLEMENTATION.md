# Implementation and evaluation notes

This is a scoped extraction of an existing personal research harness. It is not
a paper-reproduction package. Model families are credited in the README; the
specific variants below are determined by this implementation.

| Model | Implementation in this lab |
| --- | --- |
| BaseRate | Training-set success frequency, shared across held-out predictions. |
| Oracle | Returns the synthetic generator's true probability for each attempt. It deliberately has privileged simulator information and is a reference, not a deployable predictor. Finite sampled outcomes can rank another model ahead of it. |
| ItemOnly | L2-penalized logistic regression on standardized item attributes and hint fraction, with an unpenalized intercept. It has no learner state. |
| BKT | Four-probability two-state BKT fitted per concept by bounded likelihood optimization with three initial starts; a pooled fit is the fallback. It updates per-learner concept belief and has no forgetting term. The slip/guess constraints and optimizer settings are implementation choices. |
| EloIRT | A feature/hint logistic fit plus concept-specific online ability updates. Step, decay and discrimination are chosen on training sequences. This is an educational Elo-style hybrid, not a reproduction of one published Elo/IRT variant. |
| PFA | A PFA-inspired logistic baseline with concept indicators, item/hint covariates and **log1p** counts of preceding successes and failures. Count coefficients are shared, unlike the per-skill raw-count coefficients in the original PFA formulation. |

The generic model implementations, replay and scoring calculations are retained
from the original lab. Category names in the cohort simulation are changed to
`pattern`, `sequence` and `allocation`. The neutral item generator and nonlinear
synthetic difficulty function are new extraction work. No descriptor was copied
from the original game's item bank.

## Protocol

The default fixture has 180 invented item descriptors and nine invented concepts.
Attributes are bounded numbers in three neutral surfaces. The simulator generates
binary outcomes with continuous concept knowledge, practice, forgetting, transfer,
occasional insights, hints, surface-specific slip/guess and fatigue. These are
assumptions for a toy experiment, not estimated facts about real learners.

Each learner's chronological prefix is training data; its tail is held out.
Replay observes the prefix, then predicts each held-out attempt before observing
that attempt's outcome. Models fit on the pooled training prefixes. This is
within-learner prediction with a known item catalog, not a globally synchronized
deployment, unseen-learner, or unseen-item evaluation. Item standardization uses
the known descriptor catalog and never held-out outcome labels.

Point metrics are ROC AUC, log loss, Brier score, binned calibration error and
thresholded accuracy. AUC is undefined when only one outcome class is present;
the JSON report represents undefined values as `null`. Small-sample ECE is noisy
and depends on bins. A learner-block percentile bootstrap resamples complete
learners rather than treating their attempts as independent.

Paired comparisons use the same resampled learner blocks for a model and BaseRate.
Differences are model log loss minus BaseRate log loss; a negative difference is
lower loss in that sample. `bootstrap_fraction_left_lower` is the fraction of
bootstrap differences below zero. It is **not** a null-hypothesis-test p-value,
and the harness does not declare a winning model or export a deployment policy.

## Reproducibility and limits

A single CLI seed is split into separate item, difficulty, cohort and bootstrap
streams. Reports record the settings and NumPy/SciPy versions. Identical settings
and software versions are expected to reproduce the aggregate report; different
versions can change optimization and random-number behavior. Wall-clock timings,
machine configuration, fitted weights and raw interaction records are not emitted.

Optimization uses the original prototype fitting routines. Passing tests or finite
predictions are not a proof of global optimizer convergence, statistical validity
on human data, or robustness across all generator settings. Low-level research
APIs expect appropriately aligned, nonempty arrays/records. The CLI bounds its
experiment size and validates generated model probabilities before scoring.

This release includes no real learner records, employer data, game assets, game
mastery/pacing policy, game-derived item bank, game conformance vectors, Godot
runtime or export hooks, neural model or GPU requirement. The MIT grant covers
this standalone extraction; it does not license the excluded game or other local
projects. Third-party dependencies retain their own licenses and are not vendored.
