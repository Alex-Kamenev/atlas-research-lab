"""Public release boundaries, using only invented descriptors and small CPU runs."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
import os
import re
import socket
import subprocess
import sys

import numpy as np
import pytest

from atlas_research_lab import __main__ as cli
from atlas_research_lab import experiment
from atlas_research_lab.cohort import Cohort
from atlas_research_lab.items import SUPPORTED_SCHEMA, generate_bank, load_bank


TINY_SETTINGS = {"learners": 4, "attempts": 12, "bootstrap": 20, "seed": 73}
MODEL_NAMES = {"oracle", "base_rate", "item_only", "bkt", "elo_irt", "pfa"}
PRIVATE_PATH = re.compile(r"[A-Za-z]:[\\/]|\\\\|/(?:Users|home|Projects)/")


@pytest.fixture(scope="module", autouse=True)
def no_network():
    """In-process release checks must not open network connections."""
    def denied(*args, **kwargs):
        raise AssertionError("release tests must remain offline")

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(socket, "create_connection", denied)
        monkeypatch.setattr(socket.socket, "connect", denied)
        monkeypatch.setattr(socket.socket, "connect_ex", denied)
        yield


@pytest.fixture(scope="module")
def tiny_report():
    return experiment.run_experiment(**TINY_SETTINGS)


def descriptor_payload():
    return {
        "schema": SUPPORTED_SCHEMA,
        "items": [{
            "id": "synthetic-1",
            "surface": "pattern",
            "concepts": ["synthetic-skill"],
            "features": {"complexity": 0.0, "ambiguity": 0.5, "steps": 1.0},
        }],
    }


def reject_work(*args, **kwargs):
    raise AssertionError("invalid input must be rejected before experiment work")


def strict_json(text):
    def reject_constant(value):
        raise AssertionError("report contains a nonstandard JSON numeric constant")

    return json.loads(text, parse_constant=reject_constant)


def walk_report(value):
    if isinstance(value, dict):
        for key, child in value.items():
            yield key
            yield from walk_report(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk_report(child)
    else:
        yield value


def test_fixture_same_seed_is_deterministic():
    first = generate_bank(seed=21, items_per_surface=6)
    second = generate_bank(seed=21, items_per_surface=6)
    assert first == second


def test_fixture_different_seed_changes_attributes_with_same_catalog_shape():
    first = generate_bank(seed=21, items_per_surface=6)
    second = generate_bank(seed=22, items_per_surface=6)
    assert [(item.id, item.surface, item.concepts) for item in first.items] == [
        (item.id, item.surface, item.concepts) for item in second.items
    ]
    assert [item.features for item in first.items] != [item.features for item in second.items]


def test_fixture_has_only_neutral_surfaces_and_finite_unit_attributes():
    bank = generate_bank(seed=21, items_per_surface=6)
    assert set(bank.surfaces()) == {"pattern", "sequence", "allocation"}
    assert len(bank) == 18
    assert len(bank.index()) == len(bank)
    for item in bank.items:
        assert item.concepts
        assert all(np.isfinite(value) and 0.0 <= value <= 1.0
                   for value in item.features.values())


def test_loader_accepts_explicit_descriptors_including_feature_endpoints(tmp_path):
    path = tmp_path / "items.json"
    path.write_text(json.dumps(descriptor_payload()), encoding="utf-8")
    bank = load_bank(path)
    assert len(bank) == 1
    assert bank.items[0].id == "synthetic-1"
    assert bank.items[0].concepts == ("synthetic-skill",)
    assert bank.items[0].features == {"complexity": 0.0, "ambiguity": 0.5, "steps": 1.0}


@pytest.mark.parametrize("invalid", [
    "schema", "empty", "duplicate", "nan", "infinity", "surface",
    "below_range", "above_range", "boolean_feature",
])
def test_loader_rejects_invalid_descriptor_banks(tmp_path, invalid):
    payload = descriptor_payload()
    if invalid == "schema":
        payload["schema"] = "unsupported/99"
    elif invalid == "empty":
        payload["items"] = []
    elif invalid == "duplicate":
        payload["items"].append(deepcopy(payload["items"][0]))
    elif invalid == "surface":
        payload["items"][0]["surface"] = "unsupported"
    else:
        value = {
            "nan": float("nan"), "infinity": float("inf"),
            "below_range": -0.1, "above_range": 1.1, "boolean_feature": True,
        }[invalid]
        payload["items"][0]["features"]["complexity"] = value
    path = tmp_path / "invalid.json"
    # NaN/Infinity intentionally exercise rejection at the loader boundary.
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        load_bank(path)


@pytest.mark.parametrize("settings", [
    {"seed": -1}, {"seed": 2**32}, {"seed": True},
    {"items_per_surface": 5}, {"items_per_surface": 1001},
])
def test_generator_rejects_invalid_bounds(settings):
    with pytest.raises(ValueError):
        generate_bank(**settings)


@pytest.mark.parametrize("settings", [
    {"learners": 1}, {"learners": 5001}, {"learners": True},
    {"attempts": 3}, {"attempts": 1001},
    {"bootstrap": 19}, {"bootstrap": 10001},
    {"seed": -1}, {"seed": 2**32},
    {"learners": 5000, "attempts": 1000},
])
def test_experiment_rejects_invalid_args_before_generating_data(monkeypatch, settings):
    monkeypatch.setattr(experiment, "generate_bank", reject_work)
    with pytest.raises(ValueError):
        experiment.run_experiment(**{**TINY_SETTINGS, **settings})


def test_cli_requires_explicit_output(monkeypatch):
    monkeypatch.setattr(cli, "run_experiment", reject_work)
    with pytest.raises(SystemExit) as stopped:
        cli.main([])
    assert stopped.value.code == 2


@pytest.mark.parametrize("option", ["--export", "--bank"])
def test_cli_refuses_unsupported_export_and_import_flags(tmp_path, monkeypatch, option):
    monkeypatch.setattr(cli, "run_experiment", reject_work)
    destination = tmp_path / "reports"
    with pytest.raises(SystemExit) as stopped:
        cli.main(["--output", str(destination), option, "synthetic.json"])
    assert stopped.value.code == 2
    assert not destination.exists()


@pytest.mark.parametrize("option,value", [
    ("--learners", "1"), ("--attempts", "3"),
    ("--bootstrap", "19"), ("--seed", "-1"),
])
def test_cli_invalid_numeric_args_create_no_output(tmp_path, monkeypatch, option, value):
    monkeypatch.setattr(experiment, "generate_bank", reject_work)
    destination = tmp_path / "reports"
    with pytest.raises(SystemExit) as stopped:
        cli.main(["--output", str(destination), option, value])
    assert stopped.value.code == 2
    assert not destination.exists()


def test_tiny_experiment_has_all_six_roles_and_consistent_counts(tiny_report):
    by_name = {model["name"]: model for model in tiny_report["models"]}
    assert len(tiny_report["models"]) == len(by_name) == 6
    assert set(by_name) == MODEL_NAMES
    assert by_name["oracle"]["role"] == "synthetic_reference"
    assert all(model["role"] == "baseline" for name, model in by_name.items()
               if name != "oracle")
    data = tiny_report["data"]
    assert data["synthetic"] is True
    assert set(data["surfaces"]) == {"pattern", "sequence", "allocation"}
    assert data["training_observations"] > 0 and data["test_observations"] > 0
    assert data["training_observations"] + data["test_observations"] == 4 * 12
    assert all(model["scores"]["n"] == data["test_observations"]
               for model in by_name.values())


def test_report_exposes_aggregates_without_raw_records_or_private_paths(tiny_report):
    values = list(walk_report(tiny_report))
    raw_fields = {"interactions", "learner", "item_id", "true_probability",
                  "hint_fraction", "difficulty", "predictions", "correct", "order", "at"}
    assert not raw_fields.intersection(value for value in values if isinstance(value, str))
    assert not any(PRIVATE_PATH.search(value) for value in values if isinstance(value, str))
    strict_json(json.dumps(tiny_report, allow_nan=False))


def test_undefined_auc_is_null_in_strict_report_json(monkeypatch):
    original_generate = Cohort.generate

    def one_outcome(self):
        return [replace(row, correct=True) for row in original_generate(self)]

    monkeypatch.setattr(Cohort, "generate", one_outcome)
    report = experiment.run_experiment(**TINY_SETTINGS)
    parsed = strict_json(json.dumps(report, allow_nan=False))
    for model in parsed["models"]:
        assert model["scores"]["auc"] is None
        assert model["learner_bootstrap_95_percent"]["auc"] == [None, None]


def test_same_seed_repeats_the_complete_aggregate_report(tiny_report):
    assert experiment.run_experiment(**TINY_SETTINGS) == tiny_report


def test_markdown_exposes_models_and_synthetic_limitations(tiny_report):
    text = experiment.render_markdown(tiny_report)
    assert "synthetic" in text.lower()
    assert "human learning" in text.lower()
    assert all(name in text for name in MODEL_NAMES)
    assert not PRIVATE_PATH.search(text)


def test_real_cli_subprocess_writes_only_two_reports(tmp_path):
    destination = tmp_path / "reports"
    completed = subprocess.run(
        [sys.executable, "-m", "atlas_research_lab", "--output", str(destination),
         "--learners", "4", "--attempts", "12", "--bootstrap", "20", "--seed", "73"],
        cwd=tmp_path, env=os.environ.copy(), capture_output=True, text=True,
        timeout=60, check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert {path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*")
            if path.is_file()} == {"reports/report.json", "reports/report.md"}
    report = strict_json((destination / "report.json").read_text(encoding="utf-8"))
    assert {model["name"] for model in report["models"]} == MODEL_NAMES
    assert report["settings"]["learners"] == 4
    assert "synthetic" in (destination / "report.md").read_text(encoding="utf-8").lower()


@pytest.mark.parametrize("filename", ["report.json", "report.md"])
def test_existing_report_is_never_overwritten(tmp_path, monkeypatch, filename):
    monkeypatch.setattr(cli, "run_experiment", reject_work)
    destination = tmp_path / "reports"
    destination.mkdir()
    existing = destination / filename
    sentinel = b"existing synthetic report\n"
    existing.write_bytes(sentinel)
    with pytest.raises(SystemExit) as stopped:
        cli.main(["--output", str(destination)])
    assert stopped.value.code == 2
    assert existing.read_bytes() == sentinel
    assert {path.name for path in destination.iterdir()} == {filename}
