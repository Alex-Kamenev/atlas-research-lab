"""Integrity checks for the synthetic research harness.

These tests exercise temporal ordering, latent-field isolation, probability
bounds, metrics, and cluster resampling. They do not validate human learning.
"""

from __future__ import annotations

import numpy as np
import pytest

from atlas_research_lab import metrics
from atlas_research_lab.cohort import Cohort, CohortConfig, Interaction, temporal_split
from atlas_research_lab.items import ItemBank, build_feature_space, load_bank, true_difficulty
from atlas_research_lab.models import (BaseRate, BKT, EloIRT, ItemOnly, Oracle, PFA,
                                       Tracer, classic_models, replay)


@pytest.fixture(scope="module")
def bank() -> ItemBank:
    return load_bank()


@pytest.fixture(scope="module")
def small_run(bank: ItemBank):
    space = build_feature_space(bank)
    difficulty = true_difficulty(bank, space, np.random.default_rng(3))
    cohort = Cohort(bank, difficulty,
                    CohortConfig(learners=30, attempts_per_learner=40), seed=99)
    interactions = cohort.generate()
    train, test = temporal_split(interactions, holdout=0.25)
    return space, difficulty, interactions, train, test


def test_bank_loads_with_all_three_surfaces(bank: ItemBank) -> None:
    assert len(bank) > 100
    assert set(bank.surfaces()) == {"pattern", "sequence", "allocation"}


def test_features_are_standardised_within_each_surface(bank: ItemBank) -> None:
    space = build_feature_space(bank)
    for surface in bank.surfaces():
        rows = [i for i, item in enumerate(bank.items) if item.surface == surface]
        for column, name in enumerate(space.names):
            if not name.startswith(f"{surface}."):
                continue
            values = space.matrix[rows, column]
            if values.std() < 1e-9:
                continue
            assert abs(float(values.mean())) < 1e-9, f"{name} is not centred"
            assert abs(float(values.std()) - 1.0) < 1e-9, f"{name} is not scaled"


def test_a_surfaces_features_are_zero_on_other_surfaces(bank: ItemBank) -> None:
    space = build_feature_space(bank)
    pattern_columns = [i for i, n in enumerate(space.names) if n.startswith("pattern.")]
    for i, item in enumerate(bank.items):
        if item.surface == "pattern":
            continue
        assert np.allclose(space.matrix[i, pattern_columns], 0.0), \
            "a non-pattern item carries pattern features"


def test_cohort_is_deterministic(bank: ItemBank) -> None:
    space = build_feature_space(bank)
    difficulty = true_difficulty(bank, space, np.random.default_rng(5))
    config = CohortConfig(learners=8, attempts_per_learner=20)
    first = Cohort(bank, difficulty, config, seed=1234).generate()
    second = Cohort(bank, difficulty, config, seed=1234).generate()
    assert [(r.learner, r.item_id, r.correct) for r in first] == \
           [(r.learner, r.item_id, r.correct) for r in second]


def test_cohort_produces_a_usable_range_of_outcomes(small_run) -> None:
    _, _, interactions, _, _ = small_run
    rate = float(np.mean([row.correct for row in interactions]))
    # A cohort that is nearly all right or all wrong carries almost no signal and
    # would make every model look identical.
    assert 0.3 < rate < 0.85, f"base rate {rate:.3f} leaves nothing to predict"


def test_learners_differ(small_run) -> None:
    _, _, interactions, _, _ = small_run
    per_learner: dict[int, list[bool]] = {}
    for row in interactions:
        per_learner.setdefault(row.learner, []).append(row.correct)
    rates = [float(np.mean(v)) for v in per_learner.values()]
    assert float(np.std(rates)) > 0.03, "all simulated learners behave the same"


def test_practice_helps(bank: ItemBank) -> None:
    """The generator must actually contain learning, or there is nothing to trace."""
    space = build_feature_space(bank)
    difficulty = true_difficulty(bank, space, np.random.default_rng(7))
    interactions = Cohort(bank, difficulty,
                          CohortConfig(learners=120, attempts_per_learner=80),
                          seed=42).generate()
    early = [r.correct for r in interactions if r.order < 20]
    late = [r.correct for r in interactions if r.order >= 60]
    assert float(np.mean(late)) > float(np.mean(early)) + 0.02, \
        "success rate does not improve with practice"


def test_split_is_temporal_with_no_leakage(small_run) -> None:
    """Every test row follows every training row for its learner.
    
    This catches row splits that train on later outcomes while scoring an
    earlier attempt."""
    _, _, _, train, test = small_run
    latest_train: dict[int, int] = {}
    for row in train:
        latest_train[row.learner] = max(latest_train.get(row.learner, -1), row.order)
    for row in test:
        assert row.order > latest_train[row.learner], \
            f"learner {row.learner}: test row {row.order} sits inside the training window"


def test_split_keeps_every_learner_in_both_halves(small_run) -> None:
    _, _, _, train, test = small_run
    assert {r.learner for r in train} == {r.learner for r in test}


def test_split_loses_nothing(small_run) -> None:
    _, _, interactions, train, test = small_run
    assert len(train) + len(test) == len(interactions)


def test_auc_of_a_perfect_ranking() -> None:
    truth = np.array([0.0, 0.0, 1.0, 1.0])
    assert metrics.auc(truth, np.array([0.1, 0.2, 0.8, 0.9])) == pytest.approx(1.0)


def test_auc_of_a_reversed_ranking() -> None:
    truth = np.array([0.0, 0.0, 1.0, 1.0])
    assert metrics.auc(truth, np.array([0.9, 0.8, 0.2, 0.1])) == pytest.approx(0.0)


def test_auc_of_a_constant_prediction_is_a_half() -> None:
    # Ties must average, or a constant predictor scores 1.0 or 0.0 depending on
    # sort order -- and a "model" that predicts one number would look perfect.
    truth = np.array([0.0, 1.0, 0.0, 1.0])
    assert metrics.auc(truth, np.full(4, 0.5)) == pytest.approx(0.5)


def test_log_loss_rewards_confident_correctness() -> None:
    truth = np.array([1.0, 1.0, 0.0, 0.0])
    confident = np.array([0.95, 0.95, 0.05, 0.05])
    hedged = np.array([0.6, 0.6, 0.4, 0.4])
    assert metrics.log_loss(truth, confident) < metrics.log_loss(truth, hedged)


def test_log_loss_punishes_confident_error() -> None:
    truth = np.array([1.0, 1.0])
    assert metrics.log_loss(truth, np.array([0.01, 0.01])) > \
           metrics.log_loss(truth, np.array([0.4, 0.4]))


def test_calibration_error_is_zero_for_a_calibrated_predictor() -> None:
    rng = np.random.default_rng(0)
    predicted = rng.uniform(0.05, 0.95, size=40000)
    truth = (rng.uniform(size=40000) < predicted).astype(float)
    assert metrics.expected_calibration_error(truth, predicted) < 0.02


def test_calibration_error_catches_overconfidence() -> None:
    rng = np.random.default_rng(1)
    truth = (rng.uniform(size=20000) < 0.6).astype(float)
    overconfident = np.full(20000, 0.95)
    assert metrics.expected_calibration_error(truth, overconfident) > 0.3


def test_bootstrap_resamples_learners_not_rows() -> None:
    """Learner-level resampling must give a much wider interval than row-level.

    Twenty learners, fifty perfectly correlated rows each. Treating rows as
    independent observations would shrink the interval by about sqrt(50), and
    every model comparison would look far more certain than it is. Tested by
    running the same estimator both ways and requiring the honest one to be
    several times wider -- an assertion that a constant prediction cannot
    accidentally satisfy.
    """
    rng = np.random.default_rng(2)
    truth = np.repeat((rng.uniform(size=20) < 0.5).astype(float), 50)
    predicted = np.full(len(truth), 0.7)

    by_learner = np.repeat(np.arange(20), 50)
    by_row = np.arange(len(truth))

    _, learner_low, learner_high = metrics.bootstrap_metric(
        truth, predicted, by_learner, metrics.log_loss, rounds=600)
    _, row_low, row_high = metrics.bootstrap_metric(
        truth, predicted, by_row, metrics.log_loss, rounds=600)

    assert (learner_high - learner_low) > 3.0 * (row_high - row_low), (
        f"learner interval {learner_high - learner_low:.4f} is not meaningfully "
        f"wider than the row interval {row_high - row_low:.4f}"
    )


def test_paired_bootstrap_detects_a_real_difference() -> None:
    rng = np.random.default_rng(4)
    learners = np.repeat(np.arange(60), 20)
    truth = (rng.uniform(size=1200) < 0.65).astype(float)
    good = np.where(truth == 1, 0.75, 0.30)
    bad = np.full(1200, 0.5)
    result = metrics.paired_bootstrap(truth, good, bad, learners, metrics.log_loss, rounds=400)
    assert result["difference"] < 0
    assert result["high"] < 0, "a large real difference should be significant"


def test_paired_bootstrap_finds_no_difference_between_identical_models() -> None:
    rng = np.random.default_rng(6)
    learners = np.repeat(np.arange(40), 20)
    truth = (rng.uniform(size=800) < 0.6).astype(float)
    predicted = rng.uniform(0.2, 0.8, size=800)
    result = metrics.paired_bootstrap(truth, predicted, predicted.copy(), learners,
                                      metrics.log_loss, rounds=200)
    assert abs(result["difference"]) < 1e-12
    assert result["low"] <= 0.0 <= result["high"]


@pytest.mark.parametrize("factory", [BaseRate, ItemOnly, BKT, EloIRT, PFA])
def test_models_produce_valid_probabilities(factory, small_run) -> None:
    space, _, _, train, test = small_run
    model: Tracer = factory()
    model.fit(train, space)
    predicted = replay(model, train, test)
    assert len(predicted) == len(test)
    assert np.all(predicted >= 0.0) and np.all(predicted <= 1.0), \
        f"{model.name} produced a value outside [0, 1]"
    assert np.all(np.isfinite(predicted))


@pytest.mark.parametrize("factory", [BaseRate, ItemOnly, BKT, EloIRT, PFA])
def test_models_beat_or_match_the_floor(factory, small_run) -> None:
    space, _, _, train, test = small_run
    truth = np.array([1.0 if r.correct else 0.0 for r in test])

    floor = BaseRate()
    floor.fit(train, space)
    floor_loss = metrics.log_loss(truth, replay(floor, train, test))

    model: Tracer = factory()
    model.fit(train, space)
    loss = metrics.log_loss(truth, replay(model, train, test))

    # A model materially worse than predicting the base rate is actively harmful.
    assert loss < floor_loss + 0.05, f"{model.name} log loss {loss:.4f} vs floor {floor_loss:.4f}"


def test_the_oracle_beats_every_candidate(small_run) -> None:
    """Check the original fixed-seed synthetic sample against the Oracle.
    
    This is a regression check, not a universal finite-sample likelihood
    bound: the optimal expected-loss predictor can lose on an individual
    realized sample."""
    space, _, _, train, test = small_run
    truth = np.array([1.0 if r.correct else 0.0 for r in test])

    oracle = Oracle()
    oracle.fit(train, space)
    ceiling = metrics.log_loss(truth, replay(oracle, train, test))

    for model in classic_models():
        if isinstance(model, Oracle):
            continue
        model.fit(train, space)
        loss = metrics.log_loss(truth, replay(model, train, test))
        assert loss >= ceiling - 1e-9, \
            f"{model.name} beat the Bayes-optimal predictor; the harness is leaking"


def test_no_model_reads_ground_truth(small_run) -> None:
    """Only the Oracle may touch the simulator's latent fields.

    Enforced by replacing them with values that would poison any model relying on
    them, then checking predictions are unchanged.
    """
    space, _, _, train, test = small_run

    def poison(rows: list[Interaction]) -> list[Interaction]:
        return [
            Interaction(
                learner=r.learner, item_id=r.item_id, concept=r.concept,
                surface=r.surface, correct=r.correct, at=r.at,
                hint_fraction=r.hint_fraction, order=r.order,
                difficulty=999.0, true_probability=0.0,
            )
            for r in rows
        ]

    for factory in (BaseRate, ItemOnly, BKT, EloIRT, PFA):
        clean: Tracer = factory()
        clean.fit(train, space)
        clean_predictions = replay(clean, train, test)

        dirty: Tracer = factory()
        dirty.fit(poison(train), space)
        dirty_predictions = replay(dirty, poison(train), poison(test))

        assert np.allclose(clean_predictions, dirty_predictions, atol=1e-9), \
            f"{clean.name} changed its predictions when ground truth was corrupted"


def test_replay_never_sees_the_future(small_run) -> None:
    """A model that peeks would be caught: flipping later outcomes must not move
    an earlier prediction."""
    space, _, _, train, test = small_run
    model = PFA()
    model.fit(train, space)

    by_learner: dict[int, list[Interaction]] = {}
    for row in test:
        by_learner.setdefault(row.learner, []).append(row)
    learner = sorted(by_learner)[0]
    sequence = sorted(by_learner[learner], key=lambda r: r.order)
    if len(sequence) < 3:
        pytest.skip("sequence too short")

    train_rows = [r for r in train if r.learner == learner]
    first = replay(model, train_rows, sequence)[0]

    flipped = sequence[:1] + [
        Interaction(**{**row.__dict__, "correct": not row.correct}) for row in sequence[1:]
    ]
    first_flipped = replay(model, train_rows, flipped)[0]

    assert first == pytest.approx(first_flipped), \
        "the first prediction moved when later outcomes changed"


def test_every_model_either_has_a_belief_or_admits_it():
    """A model consistently exposes a latent belief or returns None.
    
    This avoids evaluating a model's latent state on an outcome-dependent
    subset of the replayed rows."""
    from atlas_research_lab.models import classic_models

    bank = load_bank()
    space = build_feature_space(bank)
    difficulty = true_difficulty(bank, space, np.random.default_rng(7))
    rows = Cohort(bank, difficulty,
                  CohortConfig(learners=6, attempts_per_learner=20),
                  seed=7).generate()

    for model in classic_models():
        model.fit(rows, space)
        model.start_learner(rows[0].learner)
        answers = []
        for row in [r for r in rows if r.learner == rows[0].learner]:
            answers.append(model.belief_for(row) is not None)
            model.observe(row)
        assert len(set(answers)) == 1, (
            f"{model.name} supplies a belief for some rows and not others"
        )
