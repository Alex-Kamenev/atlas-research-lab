"""Prediction metrics and learner-cluster bootstrap summaries.

AUC measures ranking, log loss and Brier score evaluate probabilities, and
expected calibration error summarizes bin-level confidence gaps. Accuracy is
also included. These quantities describe synthetic predictive performance,
not learning effectiveness or human outcomes.

Bootstrap routines resample whole learners to preserve dependence between
attempts from the same simulated learner. The original scoring and numerical
behavior are retained, including score() clipping predictions into [0, 1].
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EPSILON = 1e-12


@dataclass(frozen=True)
class Scores:
    auc: float
    log_loss: float
    brier: float
    ece: float
    accuracy: float
    n: int

    def as_dict(self) -> dict[str, float]:
        return {
            "auc": self.auc, "log_loss": self.log_loss, "brier": self.brier,
            "ece": self.ece, "accuracy": self.accuracy, "n": float(self.n),
        }


def auc(truth: np.ndarray, predicted: np.ndarray) -> float:
    """Area under the ROC curve, via the rank-sum identity.

    Ties get averaged ranks, which matters here: a baseline that predicts one
    constant would otherwise score 1.0 or 0.0 depending on sort order instead of
    the correct 0.5.
    """
    positives = int(truth.sum())
    negatives = int(len(truth) - positives)
    if positives == 0 or negatives == 0:
        return float("nan")

    order = np.argsort(predicted, kind="mergesort")
    ranks = np.empty(len(predicted), dtype=np.float64)
    sorted_scores = predicted[order]

    i = 0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        average = (i + j) / 2.0 + 1.0
        ranks[order[i:j + 1]] = average
        i = j + 1

    rank_sum = float(ranks[truth == 1].sum())
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def log_loss(truth: np.ndarray, predicted: np.ndarray) -> float:
    clipped = np.clip(predicted, EPSILON, 1.0 - EPSILON)
    return float(-np.mean(truth * np.log(clipped) + (1 - truth) * np.log(1 - clipped)))


def brier(truth: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean((predicted - truth) ** 2))


def expected_calibration_error(truth: np.ndarray, predicted: np.ndarray,
                               bins: int = 15) -> float:
    """Weighted average gap between predicted confidence and observed frequency."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        in_bin = (predicted >= lower) & (predicted < upper if upper < 1.0 else predicted <= upper)
        count = int(in_bin.sum())
        if count == 0:
            continue
        total += (count / len(truth)) * abs(float(predicted[in_bin].mean())
                                            - float(truth[in_bin].mean()))
    return float(total)


def score(truth: np.ndarray, predicted: np.ndarray) -> Scores:
    truth = np.asarray(truth, dtype=np.float64)
    predicted = np.clip(np.asarray(predicted, dtype=np.float64), 0.0, 1.0)
    return Scores(
        auc=auc(truth, predicted),
        log_loss=log_loss(truth, predicted),
        brier=brier(truth, predicted),
        ece=expected_calibration_error(truth, predicted),
        accuracy=float(np.mean((predicted >= 0.5) == (truth == 1))),
        n=len(truth),
    )


def reliability_curve(truth: np.ndarray, predicted: np.ndarray,
                      bins: int = 10) -> list[tuple[float, float, int]]:
    """(mean predicted, observed rate, count) per bin, for a calibration plot."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows: list[tuple[float, float, int]] = []
    for lower, upper in zip(edges[:-1], edges[1:]):
        in_bin = (predicted >= lower) & (predicted < upper if upper < 1.0 else predicted <= upper)
        count = int(in_bin.sum())
        if count == 0:
            continue
        rows.append((float(predicted[in_bin].mean()), float(truth[in_bin].mean()), count))
    return rows


# --- uncertainty ------------------------------------------------------------

def bootstrap_metric(truth: np.ndarray, predicted: np.ndarray, learners: np.ndarray,
                     metric, rounds: int = 2000, seed: int = 11,
                     confidence: float = 0.95) -> tuple[float, float, float]:
    """Point estimate and percentile interval from whole-learner resampling.
    
    Rows from one learner may be correlated. Resampling individual rows would
    treat those dependent attempts as independent observations."""
    rng = np.random.default_rng(seed)
    unique = np.unique(learners)
    by_learner = {learner: np.flatnonzero(learners == learner) for learner in unique}

    point = float(metric(truth, predicted))
    draws = np.empty(rounds, dtype=np.float64)
    for i in range(rounds):
        picked = rng.choice(unique, size=len(unique), replace=True)
        rows = np.concatenate([by_learner[learner] for learner in picked])
        draws[i] = metric(truth[rows], predicted[rows])

    draws = draws[~np.isnan(draws)]
    if draws.size == 0:
        return point, float("nan"), float("nan")
    tail = (1.0 - confidence) / 2.0
    return point, float(np.quantile(draws, tail)), float(np.quantile(draws, 1.0 - tail))


def paired_bootstrap(truth: np.ndarray, left: np.ndarray, right: np.ndarray,
                     learners: np.ndarray, metric, rounds: int = 2000,
                     seed: int = 13, confidence: float = 0.95) -> dict[str, float]:
    """Compare two predictors on the same resampled learner clusters.
    
    Returns left-minus-right metric differences, a percentile interval, and
    the fraction of bootstrap differences below zero. For a loss, negative
    values favor left. p_left_lower is this descriptive bootstrap fraction;
    it is not a separately calibrated hypothesis-test p-value."""
    rng = np.random.default_rng(seed)
    unique = np.unique(learners)
    by_learner = {learner: np.flatnonzero(learners == learner) for learner in unique}

    observed = float(metric(truth, left) - metric(truth, right))
    differences = np.empty(rounds, dtype=np.float64)
    for i in range(rounds):
        picked = rng.choice(unique, size=len(unique), replace=True)
        rows = np.concatenate([by_learner[learner] for learner in picked])
        differences[i] = metric(truth[rows], left[rows]) - metric(truth[rows], right[rows])

    differences = differences[~np.isnan(differences)]
    tail = (1.0 - confidence) / 2.0
    return {
        "difference": observed,
        "low": float(np.quantile(differences, tail)) if differences.size else float("nan"),
        "high": float(np.quantile(differences, 1.0 - tail)) if differences.size else float("nan"),
        "p_left_lower": float(np.mean(differences < 0.0)) if differences.size else float("nan"),
    }
