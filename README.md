# ATLAS research lab

A standalone Python research harness for prototype learner-model selection experiments on synthetic learners. It adapts the generic research core of a personal ATLAS lab: synthetic cohort generation, model replay and evaluation metrics.

The original generic algorithms are preserved in the scoped adaptation. Isolation, a new neutral synthetic fixture and the command-line entry point were prepared with AI assistance for this release. See [Origin](ORIGIN.md), the [case study](CASE_STUDY.md), and the [portfolio walkthrough](https://alex-kamenev.github.io/work/atlas/).

## Scope

The initial release contains six models, including the synthetic reference baseline:

| Group | Models |
| --- | --- |
| Baseline/reference models | `BaseRate`, `Oracle`, `ItemOnly` |
| Classic learner-tracing models | `BKT`, `EloIRT`, `PFA` |

The harness uses predict-before-observe replay, temporal holdout evaluation and learner-level bootstrap analysis. The new fixture uses neutral `pattern`, `sequence` and `allocation` surfaces. Its learners and outcomes are simulated.

The oracle is a synthetic reference, not a deployable learner predictor. This release does not include game-specific mastery/pacing policies, game-derived item banks, game conformance vectors, game exports, neural models or GPU features. It does not export a game policy or establish a winning model for a game.

## Run

The Python package is `atlas_research_lab`. Requires Python 3.10 or newer. Create a virtual environment, then install from the repository checkout:

```console
python -m venv .venv
# Activate .venv in your shell, then:
python -m pip install ".[test]"
python -m pytest -q
python -m atlas_research_lab --output ./results
```

An explicitly sized synthetic experiment:

```console
python -m atlas_research_lab --output ./results/small-run --learners 20 --attempts 40 --seed 7 --bootstrap 100
```

| Argument | Purpose |
| --- | --- |
| `--output` | Destination directory for experiment outputs |
| `--learners` | Synthetic learner count |
| `--attempts` | Simulated attempt count |
| `--seed` | Seed for the synthetic experiment |
| `--bootstrap` | Learner-level bootstrap resample count |

`--output` is required. The command creates `report.json` and `report.md` and refuses to overwrite either file. The CLI has no data-import or game-export option. Learners/attempts are bounded to at most 100,000 generated interactions; bootstrap rounds range from 20 to 10,000. Use a new directory for each run.

Reports contain aggregate metrics and settings, not raw learner records, fitted model weights, machine details or wall-clock timings. An undefined metric is `null` in JSON.

Use explicit experiment settings when comparing runs. Results depend on the synthetic generator and fixture assumptions; they are not evidence of human learning outcomes.

## Verification

`python -m pytest -q` passed **77 tests** on Python 3.12 with NumPy 2.5.2 and SciPy 1.18.1. A seeded synthetic CLI run completed, an offline wheel build/install succeeded, and the installed wheel reproduced the source aggregate report exactly. GitHub Actions [passed tests, a synthetic CLI smoke run and wheel builds on Python 3.10, 3.12 and 3.14](https://github.com/Alex-Kamenev/atlas-research-lab/actions/runs/43778253819). Earlier tests of the original personal lab are separate evidence.

These are prototype model-selection experiments. The harness and its results are not peer-reviewed research, a human learning-effectiveness study or a production recommendation.

## Algorithm lineage

The classic model families have established research origins:

- BKT: Corbett and Anderson, [*Knowledge tracing: Modeling the acquisition of procedural knowledge*](https://link.springer.com/article/10.1007/BF01099821), publisher date 1994.
- PFA: Pavlik, Cen and Koedinger, [*Performance Factors Analysis – A New Alternative to Knowledge Tracing*](https://pact.cs.cmu.edu/pubs/AIED%202009%20final%20Pavlik%20Cen%20Keodinger%20corrected.pdf), 2009.
- Educational Elo/IRT background: Pelánek, [*Applications of the Elo Rating System in Adaptive Educational Systems*](https://www.fi.muni.cz/~xpelanek/publications/CAE-elo.pdf), 2016.

These references identify algorithm lineage, not a claim that this harness reproduces each paper's equations, fitting procedure or empirical results. See [Implementation notes](IMPLEMENTATION.md) for the variants used here.

## Example output

[`examples/report.json`](examples/report.json) and [`examples/report.md`](examples/report.md) are a fresh **synthetic smoke example**, generated with `--learners 12 --attempts 24 --seed 7 --bootstrap 40`. They are not results from the private game or human learners. The sample is included to make the output format inspectable, not to recommend a model.

## License

[MIT](LICENSE), copyright 2026 Aleksandar Kamenev.
