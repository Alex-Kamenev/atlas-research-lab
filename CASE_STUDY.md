# ATLAS research lab

Learner-model comparisons depend on replay order, evaluation splits and resampling choices as well as the model itself. This standalone harness makes those choices explicit for model-selection experiments with synthetic learners. It adapts the generic research core of a personal ATLAS lab into a focused Python package.

The scoped core contains BaseRate, Oracle, ItemOnly, BKT, EloIRT and PFA, together with synthetic generation, replay and metrics. Evaluation predicts before observing each outcome, uses temporal holdout and resamples learners for bootstrap analysis. New neutral fixtures use pattern, sequence and allocation surfaces. AI-assisted isolation and a new CLI make the adaptation runnable without the game's content or policies; game conformance and neural/GPU work are outside this release.

**Evidence:** The standalone extraction passed **77 tests** with `python -m pytest -q`. A generated 12-learner, 24-attempt-per-learner run produced aggregate JSON/Markdown reports. An offline-built wheel reproduced the source report after installation. These checks are separate from earlier tests of the original game/lab.

**Status and limits:** Independent, AI-assisted research prototype released in November 2026. The [public MIT repository](https://github.com/Alex-Kamenev/atlas-research-lab) passed its GitHub Actions checks on Python 3.10, 3.12 and 3.14. Its learners are simulated. The experiments are not peer-reviewed evidence of human learning effectiveness, and their results do not establish a game policy or a generally superior learner model.
