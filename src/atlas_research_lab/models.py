"""Generic knowledge-tracing baselines and sequential evaluation.

Five fitted or constant baselines share the Tracer interface. Oracle is a
simulation-only reference that reads latent ground truth, not a deployable
predictor. Test rows are predicted before their outcomes are observed.

These original research routines were isolated with AI assistance. The lab
compares behavior on explicitly synthetic data; it does not validate learning
effectiveness or outcomes for people.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from .cohort import Interaction
from .items import FeatureSpace

EPSILON = 1e-9


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


class Tracer:
    """Base class. Subclasses fit on training data, then run online."""

    name: str = "tracer"

    def fit(self, train: list[Interaction], space: FeatureSpace) -> None:
        raise NotImplementedError

    def start_learner(self, learner: int) -> None:
        """Reset per-learner state before replaying a sequence."""

    def predict_one(self, row: Interaction) -> float:
        """Probability this attempt is correct, using only what has been observed."""
        raise NotImplementedError

    def observe(self, row: Interaction) -> None:
        """Fold in the outcome after the prediction has been made."""

    def belief_for(self, row: Interaction) -> float | None:
        """The model's latent concept belief, if available.
        
        This differs from the probability that the current attempt succeeds.
        Models without such a latent state return None. A returned value is not
        evidence of validated knowledge or human mastery."""
        return None


def replay(model: Tracer, train: list[Interaction], test: list[Interaction]) -> np.ndarray:
    """Predictions for every test row, produced strictly in sequence.

    The learner's training rows are replayed first so the model arrives at the
    holdout with the state it would really have had, then each test row is
    predicted before being observed.
    """
    by_learner_train: dict[int, list[Interaction]] = {}
    for row in train:
        by_learner_train.setdefault(row.learner, []).append(row)
    by_learner_test: dict[int, list[Interaction]] = {}
    for row in test:
        by_learner_test.setdefault(row.learner, []).append(row)

    predictions: dict[tuple[int, int], float] = {}
    for learner in sorted(by_learner_test):
        model.start_learner(learner)
        for row in sorted(by_learner_train.get(learner, ()), key=lambda r: r.order):
            model.observe(row)
        for row in sorted(by_learner_test[learner], key=lambda r: r.order):
            predictions[(learner, row.order)] = float(model.predict_one(row))
            model.observe(row)

    return np.array([predictions[(row.learner, row.order)] for row in test], dtype=np.float64)


class BaseRate(Tracer):
    """Predict the training set's overall success rate for every attempt.
    
    This constant baseline has AUC 0.5 when both outcome classes are present.
    Its loss provides a reference for the more complex models."""

    name = "base_rate"

    def __init__(self) -> None:
        self.rate = 0.5

    def fit(self, train: list[Interaction], space: FeatureSpace) -> None:
        self.rate = float(np.mean([row.correct for row in train])) if train else 0.5

    def predict_one(self, row: Interaction) -> float:
        return self.rate


class Oracle(Tracer):
    """Read the synthetic generator's probability for the current attempt.
    
    This uses latent information unavailable to the other models. It provides
    the Bayes-optimal expected-loss reference for this simulator, rather than
    a deployable predictor or a universal finite-sample score bound."""

    name = "oracle"

    def fit(self, train: list[Interaction], space: FeatureSpace) -> None:
        pass

    def predict_one(self, row: Interaction) -> float:
        return float(row.true_probability)


class ItemOnly(Tracer):
    """Logistic regression on item features and hint fraction.
    
    There is no learner state. This baseline helps separate item-feature
    signal from improvements due to sequential learner modeling."""

    name = "item_only"

    def __init__(self, l2: float = 1.0) -> None:
        self.l2 = l2
        self.weights: np.ndarray | None = None
        self.space: FeatureSpace | None = None

    def fit(self, train: list[Interaction], space: FeatureSpace) -> None:
        self.space = space
        X = np.column_stack([
            np.ones(len(train)),
            np.array([space.row(row.item_id) for row in train]),
            np.array([[row.hint_fraction] for row in train]),
        ])
        y = np.array([1.0 if row.correct else 0.0 for row in train])
        self.weights = _fit_logistic(X, y, self.l2)

    def predict_one(self, row: Interaction) -> float:
        assert self.weights is not None and self.space is not None
        x = np.concatenate(([1.0], self.space.row(row.item_id), [row.hint_fraction]))
        return float(sigmoid(float(x @ self.weights)))


@dataclass
class BktParams:
    prior: float = 0.25
    learn: float = 0.12
    slip: float = 0.10
    guess: float = 0.20


class BKT(Tracer):
    """Two-state Bayesian Knowledge Tracing, fitted per concept.
    
    The four probabilities are fitted by bounded likelihood maximization.
    This baseline has no forgetting or partially known latent state, while
    the synthetic cohort contains both continuous knowledge and forgetting."""

    name = "bkt"

    def __init__(self) -> None:
        self.params: dict[str, BktParams] = {}
        self.default = BktParams()
        self._belief: dict[str, float] = {}

    def fit(self, train: list[Interaction], space: FeatureSpace) -> None:
        by_concept: dict[str, list[Interaction]] = {}
        for row in train:
            by_concept.setdefault(row.concept, []).append(row)

        for concept, rows in by_concept.items():
            sequences: dict[int, list[bool]] = {}
            for row in sorted(rows, key=lambda r: (r.learner, r.order)):
                sequences.setdefault(row.learner, []).append(row.correct)
            self.params[concept] = _fit_bkt(list(sequences.values()))

        all_sequences: dict[int, list[bool]] = {}
        for row in sorted(train, key=lambda r: (r.learner, r.order)):
            all_sequences.setdefault(row.learner, []).append(row.correct)
        self.default = _fit_bkt(list(all_sequences.values()))

    def start_learner(self, learner: int) -> None:
        self._belief = {}

    def _params_for(self, concept: str) -> BktParams:
        return self.params.get(concept, self.default)

    def predict_one(self, row: Interaction) -> float:
        p = self._params_for(row.concept)
        known = self._belief.get(row.concept, p.prior)
        return known * (1.0 - p.slip) + (1.0 - known) * p.guess

    def belief_for(self, row: Interaction) -> float | None:
        return self._belief.get(row.concept, self._params_for(row.concept).prior)

    def observe(self, row: Interaction) -> None:
        p = self._params_for(row.concept)
        known = self._belief.get(row.concept, p.prior)
        if row.correct:
            numerator = known * (1.0 - p.slip)
            denominator = numerator + (1.0 - known) * p.guess
        else:
            numerator = known * p.slip
            denominator = numerator + (1.0 - known) * (1.0 - p.guess)
        posterior = numerator / denominator if denominator > EPSILON else known
        self._belief[row.concept] = posterior + (1.0 - posterior) * p.learn


def _fit_bkt(sequences: list[list[bool]]) -> BktParams:
    """Maximise the sequence likelihood over the four BKT probabilities.

    Slip and guess are capped at 0.4 and 0.45. Without that bound the optimiser
    happily finds the degenerate solution where guess is so high that "not
    knowing" predicts correctness better than "knowing" -- a perfect fit with the
    states swapped, and a model that reports mastery for a learner who has none.
    """
    if not sequences:
        return BktParams()

    def negative_log_likelihood(theta: np.ndarray) -> float:
        prior, learn, slip, guess = _unpack_bkt(theta)
        total = 0.0
        for sequence in sequences:
            known = prior
            for correct in sequence:
                if correct:
                    p = known * (1.0 - slip) + (1.0 - known) * guess
                    total -= math.log(max(p, EPSILON))
                    numerator = known * (1.0 - slip)
                    denominator = numerator + (1.0 - known) * guess
                else:
                    p = known * slip + (1.0 - known) * (1.0 - guess)
                    total -= math.log(max(p, EPSILON))
                    numerator = known * slip
                    denominator = numerator + (1.0 - known) * (1.0 - guess)
                posterior = numerator / denominator if denominator > EPSILON else known
                known = posterior + (1.0 - posterior) * learn
        return total

    best: BktParams | None = None
    best_score = math.inf
    # A few starts, because the likelihood surface has local optima.
    for start in ((0.2, 0.1, 0.1, 0.2), (0.5, 0.2, 0.05, 0.1), (0.1, 0.05, 0.2, 0.3)):
        result = minimize(
            negative_log_likelihood,
            x0=np.array(start),
            bounds=[(0.01, 0.95), (0.001, 0.6), (0.01, 0.40), (0.01, 0.45)],
            method="L-BFGS-B",
        )
        if result.fun < best_score:
            best_score = float(result.fun)
            best = BktParams(*_unpack_bkt(result.x))
    return best or BktParams()


def _unpack_bkt(theta: np.ndarray) -> tuple[float, float, float, float]:
    prior = float(np.clip(theta[0], 0.01, 0.95))
    learn = float(np.clip(theta[1], 0.001, 0.6))
    slip = float(np.clip(theta[2], 0.01, 0.40))
    guess = float(np.clip(theta[3], 0.01, 0.45))
    return prior, learn, slip, guess


class EloIRT(Tracer):
    """Online continuous ability updates against feature-based item logits.
    
    A logistic fit supplies the item-feature and hint terms. An Elo-style
    update then adjusts concept-specific ability, with step size decaying
    as that concept is practiced."""

    name = "elo_irt"

    def __init__(self, l2: float = 1.0) -> None:
        self.l2 = l2
        self.space: FeatureSpace | None = None
        self.difficulty_weights: np.ndarray | None = None
        self.discrimination = 1.0
        self.step = 0.6
        self.decay = 0.55
        self.hint_effect = 0.0
        self._ability: dict[str, float] = {}
        self._seen: dict[str, int] = {}

    def fit(self, train: list[Interaction], space: FeatureSpace) -> None:
        self.space = space
        X = np.column_stack([
            np.ones(len(train)),
            np.array([space.row(row.item_id) for row in train]),
            np.array([[row.hint_fraction] for row in train]),
        ])
        y = np.array([1.0 if row.correct else 0.0 for row in train])
        weights = _fit_logistic(X, y, self.l2)
        self.difficulty_weights = weights
        self.hint_effect = float(weights[-1])

        # Tune the online update on the training sequences themselves.
        best = (math.inf, self.step, self.decay, self.discrimination)
        for step in (0.25, 0.4, 0.6, 0.9):
            for decay in (0.3, 0.55, 0.8):
                for discrimination in (0.6, 1.0, 1.5):
                    loss = self._sequence_loss(train, step, decay, discrimination)
                    if loss < best[0]:
                        best = (loss, step, decay, discrimination)
        _, self.step, self.decay, self.discrimination = best

    def _base_logit(self, row: Interaction) -> float:
        assert self.space is not None and self.difficulty_weights is not None
        x = np.concatenate(([1.0], self.space.row(row.item_id), [row.hint_fraction]))
        return float(x @ self.difficulty_weights)

    def _sequence_loss(self, rows: list[Interaction], step: float, decay: float,
                       discrimination: float) -> float:
        by_learner: dict[int, list[Interaction]] = {}
        for row in rows:
            by_learner.setdefault(row.learner, []).append(row)
        total = 0.0
        count = 0
        for sequence in by_learner.values():
            ability: dict[str, float] = {}
            seen: dict[str, int] = {}
            for row in sorted(sequence, key=lambda r: r.order):
                theta = ability.get(row.concept, 0.0)
                p = float(sigmoid(self._base_logit(row) + discrimination * theta))
                total -= math.log(max(p if row.correct else 1.0 - p, EPSILON))
                count += 1
                n = seen.get(row.concept, 0)
                rate = step / (1.0 + decay * n)
                ability[row.concept] = theta + rate * ((1.0 if row.correct else 0.0) - p)
                seen[row.concept] = n + 1
        return total / max(1, count)

    def start_learner(self, learner: int) -> None:
        self._ability = {}
        self._seen = {}

    def predict_one(self, row: Interaction) -> float:
        theta = self._ability.get(row.concept, 0.0)
        return float(sigmoid(self._base_logit(row) + self.discrimination * theta))

    def belief_for(self, row: Interaction) -> float | None:
        # Ability is a logit. This transform places it on a probability scale;
        # it does not establish a calibrated interpretation of latent knowledge.
        return float(sigmoid(self._ability.get(row.concept, 0.0)))

    def observe(self, row: Interaction) -> None:
        theta = self._ability.get(row.concept, 0.0)
        p = self.predict_one(row)
        n = self._seen.get(row.concept, 0)
        rate = self.step / (1.0 + self.decay * n)
        self._ability[row.concept] = theta + rate * ((1.0 if row.correct else 0.0) - p)
        self._seen[row.concept] = n + 1


class PFA(Tracer):
    """Performance Factors Analysis using prior successes and failures.
    
    The logit combines item features, hint fraction, a concept indicator,
    and log1p counts of preceding successes and failures. No latent belief
    is exposed through belief_for."""

    name = "pfa"

    def __init__(self, l2: float = 1.0) -> None:
        self.l2 = l2
        self.space: FeatureSpace | None = None
        self.weights: np.ndarray | None = None
        self.concepts: list[str] = []
        self._successes: dict[str, int] = {}
        self._failures: dict[str, int] = {}

    def _design(self, row: Interaction, successes: int, failures: int) -> np.ndarray:
        assert self.space is not None
        concept_indicator = np.zeros(len(self.concepts))
        if row.concept in self.concepts:
            concept_indicator[self.concepts.index(row.concept)] = 1.0
        # Counts enter through log1p: the difference between the first and second
        # attempt matters far more than between the twentieth and twenty-first.
        return np.concatenate((
            [1.0],
            self.space.row(row.item_id),
            [row.hint_fraction],
            concept_indicator,
            [math.log1p(successes), math.log1p(failures)],
        ))

    def fit(self, train: list[Interaction], space: FeatureSpace) -> None:
        self.space = space
        self.concepts = sorted({row.concept for row in train})

        rows: list[np.ndarray] = []
        targets: list[float] = []
        successes: dict[tuple[int, str], int] = {}
        failures: dict[tuple[int, str], int] = {}
        for row in sorted(train, key=lambda r: (r.learner, r.order)):
            key = (row.learner, row.concept)
            rows.append(self._design(row, successes.get(key, 0), failures.get(key, 0)))
            targets.append(1.0 if row.correct else 0.0)
            if row.correct:
                successes[key] = successes.get(key, 0) + 1
            else:
                failures[key] = failures.get(key, 0) + 1

        self.weights = _fit_logistic(np.array(rows), np.array(targets), self.l2)

    def start_learner(self, learner: int) -> None:
        self._successes = {}
        self._failures = {}

    def predict_one(self, row: Interaction) -> float:
        assert self.weights is not None
        x = self._design(row, self._successes.get(row.concept, 0),
                         self._failures.get(row.concept, 0))
        return float(sigmoid(float(x @ self.weights)))

    def observe(self, row: Interaction) -> None:
        if row.correct:
            self._successes[row.concept] = self._successes.get(row.concept, 0) + 1
        else:
            self._failures[row.concept] = self._failures.get(row.concept, 0) + 1


def _fit_logistic(X: np.ndarray, y: np.ndarray, l2: float) -> np.ndarray:
    """Fit L2-penalized logistic regression with an analytic gradient.
    
    The original NumPy/SciPy implementation leaves the intercept unpenalized."""
    n, d = X.shape

    def objective(w: np.ndarray) -> tuple[float, np.ndarray]:
        z = X @ w
        p = sigmoid(z)
        p = np.clip(p, EPSILON, 1.0 - EPSILON)
        loss = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        penalty = np.copy(w)
        penalty[0] = 0.0
        loss += 0.5 * l2 * float(penalty @ penalty) / n
        gradient = X.T @ (p - y) / n + l2 * penalty / n
        return float(loss), gradient

    result = minimize(objective, x0=np.zeros(d), jac=True, method="L-BFGS-B")
    return np.asarray(result.x, dtype=np.float64)


def classic_models() -> list[Tracer]:
    """The five generic baselines, plus the simulation-only Oracle reference."""
    return [Oracle(), BaseRate(), ItemOnly(), BKT(), EloIRT(), PFA()]
