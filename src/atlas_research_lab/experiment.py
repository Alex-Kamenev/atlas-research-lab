"""Aggregate, synthetic experiments over the original research baselines."""

from __future__ import annotations

import math
from importlib.metadata import PackageNotFoundError, version

import numpy as np

from . import __version__, metrics
from .cohort import Cohort, CohortConfig, temporal_split
from .items import build_feature_space, generate_bank, true_difficulty
from .models import classic_models, replay


def _integer(value: int, name: str, lower: int, upper: int) -> int:
    if type(value) is not int or not lower <= value <= upper:
        raise ValueError(f"{name} must be an integer from {lower} to {upper}")
    return value


def _finite_json(value):
    """Represent undefined metrics as null, never nonstandard JSON NaN/Infinity."""
    if isinstance(value, dict):
        return {key: _finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite_json(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def run_experiment(*, learners: int = 32, attempts: int = 40,
                   seed: int = 20261115, bootstrap: int = 200) -> dict:
    """Compare six baselines on generated interactions from one fixed seed.

    The held-out tail belongs to the same simulated learners as the training
    prefix. It tests within-learner forecasting, not generalization to new people
    or unseen items. Model rankings depend on this one synthetic data generator.
    """
    _integer(learners, "learners", 2, 5000)
    _integer(attempts, "attempts", 4, 1000)
    _integer(seed, "seed", 0, 2**32 - 1)
    _integer(bootstrap, "bootstrap", 20, 10000)
    if learners * attempts > 100_000:
        raise ValueError("experiment exceeds the 100000-interaction limit")
    streams = np.random.SeedSequence(seed).spawn(4)
    bank_seed = int(streams[0].generate_state(1)[0])
    cohort_seed = int(streams[2].generate_state(1)[0])
    bootstrap_seed = int(streams[3].generate_state(1)[0])
    bank = generate_bank(seed=bank_seed)
    space = build_feature_space(bank)
    difficulty = true_difficulty(bank, space, np.random.default_rng(streams[1]))
    interactions = Cohort(bank, difficulty,
                          CohortConfig(learners=learners, attempts_per_learner=attempts),
                          seed=cohort_seed).generate()
    train, test = temporal_split(interactions, holdout=0.25)
    if not train or not test:
        raise ValueError("experiment must contain training and held-out observations")
    truth = np.asarray([float(row.correct) for row in test], dtype=np.float64)
    learner_ids = np.asarray([row.learner for row in test], dtype=np.int64)
    model_results = []
    predictions: dict[str, np.ndarray] = {}
    for model in classic_models():
        model.fit(train, space)
        predicted = replay(model, train, test)
        if (predicted.shape != truth.shape or not np.all(np.isfinite(predicted))
                or np.any((predicted < 0) | (predicted > 1))):
            raise ValueError("model produced invalid probabilities")
        predictions[model.name] = predicted
        point = metrics.score(truth, predicted)
        _, loss_low, loss_high = metrics.bootstrap_metric(
            truth, predicted, learner_ids, metrics.log_loss,
            rounds=bootstrap, seed=bootstrap_seed)
        _, auc_low, auc_high = metrics.bootstrap_metric(
            truth, predicted, learner_ids, metrics.auc,
            rounds=bootstrap, seed=bootstrap_seed)
        model_results.append({
            "name": model.name,
            "role": "synthetic_reference" if model.name == "oracle" else "baseline",
            "scores": point.as_dict(),
            "learner_bootstrap_95_percent": {
                "log_loss": [loss_low, loss_high], "auc": [auc_low, auc_high],
            },
        })
    for result in model_results:
        name = result["name"]
        if name in {"oracle", "base_rate"}:
            continue
        paired = metrics.paired_bootstrap(
            truth, predictions[name], predictions["base_rate"], learner_ids,
            metrics.log_loss, rounds=bootstrap, seed=bootstrap_seed)
        result["paired_log_loss_vs_base_rate"] = {
            "difference": paired["difference"],
            "interval_95_percent": [paired["low"], paired["high"]],
            "bootstrap_fraction_left_lower": paired["p_left_lower"],
        }
    packages = {}
    for name in ("numpy", "scipy"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = "unknown"
    return _finite_json({
        "schema": "atlas.research_report/1",
        "package_version": __version__,
        "dependencies": packages,
        "settings": {"seed": seed, "learners": learners, "attempts_per_learner": attempts,
                     "bootstrap_rounds": bootstrap, "holdout_fraction": 0.25},
        "data": {"source": "generated-neutral-fixture-v1", "synthetic": True,
                 "items": len(bank), "surfaces": bank.surfaces(),
                 "training_observations": len(train), "test_observations": len(test)},
        "protocol": "Per-learner chronological prefix/tail; predict before observing each held-out outcome",
        "limitations": [
            "Synthetic outcomes do not establish human learning effectiveness.",
            "Oracle reads simulator probabilities and is not a deployable predictor.",
            "A finite sample need not rank the oracle first.",
            "Fixed catalog and same learners; not a cold-start or unseen-learner evaluation.",
            "Bootstrap fractions are descriptive, not null-hypothesis-test p-values.",
            "Undefined AUC values and intervals are represented as null.",
            "No game policy, runtime benchmark, or production recommendation is produced.",
        ],
        "models": model_results,
    })


def render_markdown(report: dict) -> str:
    """Render the aggregate metrics, with the synthetic-data boundary visible."""
    def number(value) -> str:
        return "undefined" if value is None else f"{value:.4f}"

    rows = ["# Synthetic knowledge-tracing experiment", "",
            "Generated toy descriptors and simulated learners only. These results do not measure human learning.",
            "", f"Seed: {report['settings']['seed']}; learners: {report['settings']['learners']}; "
            f"attempts per learner: {report['settings']['attempts_per_learner']}.", "",
            "| Model | Role | Log loss | AUC | Brier | ECE |", "| --- | --- | --- | --- | --- | --- |"]
    for model in report["models"]:
        scores = model["scores"]
        rows.append(f"| {model['name']} | {model['role']} | {number(scores['log_loss'])} | "
                    f"{number(scores['auc'])} | {number(scores['brier'])} | {number(scores['ece'])} |")
    rows.extend(["", "Log loss and Brier are losses; lower is better. AUC measures ranking. "
                 "ECE depends on binning and sample size; compare it cautiously.", "",
                 "The JSON report includes learner-level percentile intervals and paired log-loss "
                 "differences against BaseRate. The reported bootstrap fraction is not a p-value.",
                 "", "## Limits", ""])
    rows.extend(f"- {limit}" for limit in report["limitations"])
    return "\n".join(rows) + "\n"
