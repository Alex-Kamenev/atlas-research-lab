"""Neutral, generated item descriptors for synthetic model experiments.

No exercises, text, identifiers, policies or observations are imported from the
ATLAS game. The generated attributes are toy covariates, not measured tasks or
human learning data. Feature scaling follows the original lab's per-surface
design; the fixture and difficulty function were written for this extraction.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
import re

import numpy as np

SUPPORTED_SCHEMA = "atlas.research_items/1"
SURFACE_FEATURES: dict[str, tuple[str, ...]] = {
    "pattern": ("complexity", "ambiguity", "steps"),
    "sequence": ("complexity", "context", "noise"),
    "allocation": ("complexity", "density", "gap"),
}
_LABEL = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")


@dataclass(frozen=True)
class Item:
    id: str
    surface: str
    concepts: tuple[str, ...]
    features: dict[str, float]

    def feature(self, name: str, default: float = 0.0) -> float:
        return float(self.features.get(name, default))


@dataclass
class ItemBank:
    items: list[Item]
    seed: int = 0
    runtime: str = "synthetic-generator-v1"

    def __len__(self) -> int:
        return len(self.items)

    def by_surface(self, surface: str) -> list[Item]:
        return [item for item in self.items if item.surface == surface]

    def surfaces(self) -> list[str]:
        return sorted({item.surface for item in self.items})

    def concepts(self) -> list[str]:
        return sorted({concept for item in self.items for concept in item.concepts})

    def index(self) -> dict[str, Item]:
        return {item.id: item for item in self.items}


def generate_bank(seed: int = 7, items_per_surface: int = 60) -> ItemBank:
    """Generate invented attributes and concept labels, without loading files."""
    if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
        raise ValueError("seed must be an integer between zero and 2**32 - 1")
    if type(items_per_surface) is not int or not 6 <= items_per_surface <= 1000:
        raise ValueError("items_per_surface must be an integer from 6 to 1000")
    rng = np.random.default_rng(seed)
    items: list[Item] = []
    for surface, names in SURFACE_FEATURES.items():
        for number in range(items_per_surface):
            concepts = (f"{surface}_skill_{number % 3}",)
            features = {name: float(rng.uniform(0.0, 1.0)) for name in names}
            items.append(Item(f"{surface}-{number:03d}", surface, concepts, features))
    return ItemBank(items, seed=seed)


def load_bank(path: Path | str | None = None) -> ItemBank:
    """Generate the default bank, or read an explicit small research-item JSON.

    This API accepts only item descriptors. It does not ingest learner records.
    The public command line always generates its own bank and has no data-import
    flag. Error messages never echo supplied item or concept contents.
    """
    if path is None:
        return generate_bank()
    source = Path(path)
    if source.stat().st_size > 2_000_000:
        raise ValueError("item bank exceeds the 2 MB input limit")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if type(payload) is not dict or payload.get("schema") != SUPPORTED_SCHEMA:
        raise ValueError("unknown item bank schema")
    raw = payload.get("items")
    if type(raw) is not list or not 1 <= len(raw) <= 3000:
        raise ValueError("item bank must contain 1 to 3000 items")
    items: list[Item] = []
    seen: set[str] = set()
    for entry in raw:
        if type(entry) is not dict:
            raise ValueError("invalid item record")
        identifier, surface = entry.get("id"), entry.get("surface")
        concepts, features = entry.get("concepts"), entry.get("features")
        if (type(identifier) is not str or not _LABEL.fullmatch(identifier)
                or identifier in seen or type(surface) is not str
                or surface not in SURFACE_FEATURES):
            raise ValueError("invalid or duplicate item identity")
        if (type(concepts) is not list or not 1 <= len(concepts) <= 8
                or any(type(c) is not str or not _LABEL.fullmatch(c) for c in concepts)
                or len(set(concepts)) != len(concepts)):
            raise ValueError("invalid concept labels")
        if type(features) is not dict or set(features) != set(SURFACE_FEATURES[surface]):
            raise ValueError("invalid item feature names")
        if any(type(v) not in (int, float) or not 0.0 <= v <= 1.0
               for v in features.values()):
            raise ValueError("features must be finite numbers in the range zero to one")
        seen.add(identifier)
        items.append(Item(identifier, surface, tuple(concepts),
                          {key: float(value) for key, value in features.items()}))
    return ItemBank(items, runtime="explicit-item-descriptors")


@dataclass
class FeatureSpace:
    names: list[str]
    matrix: np.ndarray
    item_ids: list[str]
    _row_of: dict[str, int] = field(default_factory=dict)

    def row(self, item_id: str) -> np.ndarray:
        return self.matrix[self._row_of[item_id]]

    def __post_init__(self) -> None:
        if not self._row_of:
            self._row_of = {item_id: i for i, item_id in enumerate(self.item_ids)}


def build_feature_space(bank: ItemBank) -> FeatureSpace:
    """Per-surface standardization, with zeroes on unrelated surface columns.

    Item attributes are available for the fixed item catalog before training.
    This is not an unseen-item/cold-start evaluation protocol.
    """
    if not bank.items:
        raise ValueError("item bank must not be empty")
    surfaces = bank.surfaces()
    names = [f"{surface}.{feature}" for surface in surfaces
             for feature in SURFACE_FEATURES[surface]]
    names.extend(f"is_{surface}" for surface in surfaces)
    index = {name: i for i, name in enumerate(names)}
    matrix = np.zeros((len(bank), len(names)), dtype=np.float64)
    item_ids = [item.id for item in bank.items]
    if len(set(item_ids)) != len(item_ids):
        raise ValueError("item identifiers must be unique")
    for i, item in enumerate(bank.items):
        for feature in SURFACE_FEATURES[item.surface]:
            matrix[i, index[f"{item.surface}.{feature}"]] = item.feature(feature)
        matrix[i, index[f"is_{item.surface}"]] = 1.0
    for surface in surfaces:
        rows = np.array([i for i, item in enumerate(bank.items) if item.surface == surface])
        for feature in SURFACE_FEATURES[surface]:
            column = index[f"{surface}.{feature}"]
            values = matrix[rows, column]
            spread = float(values.std())
            matrix[rows, column] = 0.0 if spread < 1e-9 else (values - values.mean()) / spread
    return FeatureSpace(names, matrix, item_ids)


def true_difficulty(bank: ItemBank, space: FeatureSpace, rng: np.random.Generator,
                    noise: float = 0.35) -> dict[str, float]:
    """Invent nonlinear latent difficulty for the simulator, not a real learner.

    The function intentionally differs from the fitted linear baseline. Its
    assumptions still determine the comparison; ranks are not general evidence
    about real educational tasks. ``space`` preserves the original lab API.
    """
    if not np.isfinite(noise) or noise < 0:
        raise ValueError("noise must be finite and nonnegative")
    values = []
    for item in bank.items:
        a, b, c = [item.feature(name) for name in SURFACE_FEATURES[item.surface]]
        values.append(1.5 * a + 0.8 * b * c + 0.35 * np.sin(5.0 * c)
                      + float(rng.normal(0.0, noise)))
    array = np.asarray(values, dtype=np.float64)
    normalized = (array - array.mean()) / (float(array.std()) or 1.0)
    return dict(zip((item.id for item in bank.items), map(float, normalized)))
