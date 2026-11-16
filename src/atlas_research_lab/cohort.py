"""Synthetic learners for controlled knowledge-tracing experiments.

These are simulated learners, not people. Their outcomes do not establish
learning effectiveness, cognitive mechanisms, or real-world model performance.
The process deliberately differs from the fitted baselines: knowledge is
continuous per concept, practice partially transfers across shared surfaces,
forgetting depends on elapsed time, occasional insights create jumps, and
surface-specific slip/guess rates and within-session fatigue affect outcomes.

Defaults and numerical mechanics are retained from the original research
simulation. Only the three surface categories are renamed for the neutral
synthetic item bank; no original application assets or datasets are used.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .items import Item, ItemBank

SECONDS_PER_DAY = 86400.0


@dataclass(frozen=True)
class Interaction:
    """One attempt in a simulated learner's interaction sequence."""

    learner: int
    item_id: str
    concept: str
    surface: str
    correct: bool
    at: float                 # seconds on the simulated learner's clock
    hint_fraction: float
    order: int                # index within this learner's sequence

    # Simulator ground truth. Available only to the Oracle, which exists to
    # bound what any model could achieve; every candidate is checked by
    # `test_no_model_reads_ground_truth` never to touch these two fields.
    difficulty: float = 0.0
    true_probability: float = 0.5


@dataclass
class CohortConfig:
    learners: int = 400
    attempts_per_learner: int = 120
    session_length: int = 14              # attempts before a break
    break_days: tuple[float, float] = (0.4, 3.5)
    within_session_seconds: tuple[float, float] = (40.0, 260.0)

    ability_sd: float = 1.0
    learning_rate: tuple[float, float] = (0.06, 0.26)
    forget_half_life_days: tuple[float, float] = (2.0, 30.0)
    transfer: float = 0.28                # share of learning that spills to neighbours
    insight_chance: float = 0.02          # probability of a discrete jump per attempt
    insight_size: tuple[float, float] = (0.5, 1.4)
    fatigue: float = 0.10                 # accuracy lost by the end of a long session
    discrimination: float = 1.25

    # Synthetic surface categories have different slip and guess defaults.
    slip_by_surface: dict[str, float] = field(default_factory=lambda: {
        "pattern": 0.07, "sequence": 0.12, "allocation": 0.09,
    })
    guess_by_surface: dict[str, float] = field(default_factory=lambda: {
        "pattern": 0.16, "sequence": 0.06, "allocation": 0.12,
    })

    hint_chance: float = 0.22
    hint_help: float = 0.45               # how much a hint raises the odds


class Cohort:
    """Generates interaction sequences from the process described above."""

    def __init__(self, bank: ItemBank, difficulty: dict[str, float],
                 config: CohortConfig | None = None, seed: int = 20260825) -> None:
        self.bank = bank
        self.difficulty = difficulty
        self.config = config or CohortConfig()
        self.rng = np.random.default_rng(seed)
        self.concepts = bank.concepts()
        self._items_by_concept = {
            concept: [item for item in bank.items if concept in item.concepts]
            for concept in self.concepts
        }
        # A simple neighbourhood: concepts sharing a surface reinforce each other.
        self._neighbours = {
            concept: [
                other for other in self.concepts
                if other != concept and self._share_surface(concept, other)
            ]
            for concept in self.concepts
        }

    def _share_surface(self, left: str, right: str) -> bool:
        left_surfaces = {item.surface for item in self._items_by_concept[left]}
        right_surfaces = {item.surface for item in self._items_by_concept[right]}
        return bool(left_surfaces & right_surfaces)

    def generate(self) -> list[Interaction]:
        interactions: list[Interaction] = []
        for learner in range(self.config.learners):
            interactions.extend(self._one_learner(learner))
        return interactions

    def _one_learner(self, learner: int) -> list[Interaction]:
        cfg = self.config
        rng = self.rng

        ability = float(rng.normal(0.0, cfg.ability_sd))
        learning_rate = float(rng.uniform(*cfg.learning_rate))
        half_life = float(rng.uniform(*cfg.forget_half_life_days))

        # Continuous per-concept knowledge, starting near the learner's ability.
        knowledge = {c: ability + float(rng.normal(0.0, 0.4)) for c in self.concepts}
        last_practised = {c: 0.0 for c in self.concepts}

        clock = 0.0
        produced: list[Interaction] = []

        for order in range(cfg.attempts_per_learner):
            position_in_session = order % cfg.session_length
            if position_in_session == 0 and order > 0:
                clock += float(rng.uniform(*cfg.break_days)) * SECONDS_PER_DAY
            clock += float(rng.uniform(*cfg.within_session_seconds))

            concept = self._choose_concept(knowledge, rng)
            item = self._choose_item(concept, knowledge[concept], rng)
            if item is None:
                continue

            # Forgetting, applied to the concept about to be exercised.
            elapsed_days = max(0.0, (clock - last_practised[concept]) / SECONDS_PER_DAY)
            retained = 0.5 ** (elapsed_days / half_life)
            effective = knowledge[concept] * retained + ability * (1.0 - retained) * 0.35

            # Fatigue over a long session.
            effective -= cfg.fatigue * (position_in_session / max(1, cfg.session_length))

            hinted = bool(rng.random() < cfg.hint_chance)
            hint_fraction = float(rng.uniform(0.25, 1.0)) if hinted else 0.0

            b = self.difficulty[item.id]
            slip = cfg.slip_by_surface.get(item.surface, 0.08)
            guess = cfg.guess_by_surface.get(item.surface, 0.12)
            logit = cfg.discrimination * (effective - b) + cfg.hint_help * hint_fraction
            core = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, logit))))
            probability = guess + (1.0 - guess - slip) * core

            correct = bool(rng.random() < probability)

            produced.append(Interaction(
                learner=learner, item_id=item.id, concept=concept, surface=item.surface,
                correct=correct, at=clock, hint_fraction=hint_fraction, order=order,
                difficulty=b, true_probability=probability,
            ))

            self._learn(knowledge, concept, correct, hint_fraction, learning_rate, rng)
            last_practised[concept] = clock

        return produced

    def _learn(self, knowledge: dict[str, float], concept: str, correct: bool,
               hint_fraction: float, learning_rate: float, rng: np.random.Generator) -> None:
        cfg = self.config
        # The simulation assigns a smaller practice gain after a failure.
        gain = learning_rate * (1.0 if correct else 0.45)
        gain *= (1.0 - 0.4 * hint_fraction)
        if rng.random() < cfg.insight_chance:
            gain += float(rng.uniform(*cfg.insight_size))

        knowledge[concept] += gain
        for neighbour in self._neighbours[concept]:
            knowledge[neighbour] += gain * cfg.transfer / max(1, len(self._neighbours[concept]))

    def _choose_concept(self, knowledge: dict[str, float], rng: np.random.Generator) -> str:
        """Sample concepts with greater weight on lower current knowledge."""
        weights = np.array([math.exp(-knowledge[c]) for c in self.concepts])
        weights = weights / weights.sum()
        return str(rng.choice(self.concepts, p=weights))

    def _choose_item(self, concept: str, level: float, rng: np.random.Generator) -> Item | None:
        """Pick an item near the learner's level, as adaptive selection would."""
        candidates = self._items_by_concept.get(concept, [])
        if not candidates:
            return None
        sample = candidates if len(candidates) <= 24 else [
            candidates[i] for i in rng.choice(len(candidates), size=24, replace=False)
        ]
        # Softmax over closeness to the learner's level, with enough spread that
        # the log is not degenerate.
        gaps = np.array([abs(self.difficulty[item.id] - level) for item in sample])
        weights = np.exp(-gaps / 0.6)
        weights = weights / weights.sum()
        return sample[int(rng.choice(len(sample), p=weights))]


def temporal_split(interactions: list[Interaction], holdout: float = 0.25
                   ) -> tuple[list[Interaction], list[Interaction]]:
    """Keep the tail of each learner's ordered sequence for testing.
    
    A random row split could train on later outcomes while scoring earlier
    ones. Per-learner temporal holdout avoids that ordering leak."""
    by_learner: dict[int, list[Interaction]] = {}
    for row in interactions:
        by_learner.setdefault(row.learner, []).append(row)

    train: list[Interaction] = []
    test: list[Interaction] = []
    for learner in sorted(by_learner):
        sequence = sorted(by_learner[learner], key=lambda r: r.order)
        cut = max(1, int(round(len(sequence) * (1.0 - holdout))))
        train.extend(sequence[:cut])
        test.extend(sequence[cut:])
    return train, test
