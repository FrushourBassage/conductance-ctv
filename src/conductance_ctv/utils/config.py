"""Experiment configuration: YAML files plus command-line overrides.

An experiment names one file under ``configs/experiment`` and inherits the model,
data and training defaults it lists. Overrides are dotted paths applied after the
inheritance, which is the only way a reported value changes.

Ref: the compute profile of the project context; Sec. 6.4 and Sec. 6.6.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from conductance_ctv.data.schema import GridSpec
from conductance_ctv.training.trainer import TrainConfig

CONFIG_ROOT = Path(__file__).resolve().parents[3] / "configs"

_HYDRAULIC_DEFAULTS: dict[str, Any] = {
    "step_mm": None,
    "upsample_factor": 2,
    "field_of_view_mm": 40.0,
    "segment_fractions": [0.0, 0.40, 0.75, 1.0],
    "reference_mode": "max",
    "smooth_window": 5,
    "solve_tolerance": 1e-11,
    "max_sweeps": 20000,
    "resolution_floor_cells": 2,
}


def repo_root() -> Path:
    """Directory holding ``configs/``, ``src/`` and the report artefacts."""
    return CONFIG_ROOT.parent


def _read_yaml(path: Path) -> dict[str, Any]:
    import yaml

    if not path.exists():
        raise FileNotFoundError(f"missing configuration file: {path.relative_to(path.parents[2])}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} does not contain a mapping")
    return payload


def _deep_merge(base: dict[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overlay.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(dict(merged[key]), value)
        else:
            merged[key] = value
    return merged


def parse_overrides(items: Sequence[str]) -> dict[str, Any]:
    """Parse ``a.b=value`` command-line overrides into a nested mapping."""
    out: dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"an override must have the form key=value, got {item!r}")
        key, raw = item.split("=", 1)
        value = _coerce(raw)
        cursor = out
        parts = key.split(".")
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
            if not isinstance(cursor, dict):
                raise ValueError(f"override {key!r} collides with an earlier scalar")
        cursor[parts[-1]] = value
    return out


def _coerce(raw: str) -> Any:
    lowered = raw.strip().lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    if lowered in ("null", "none"):
        return None
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    if raw.startswith("[") and raw.endswith("]"):
        inner = raw[1:-1].strip()
        if not inner:
            return []
        return [_coerce(part) for part in inner.split(",")]
    return raw


def resolve_experiment(name_or_path: str, overrides: Sequence[str] = ()) -> dict[str, Any]:
    """Load an experiment file, apply its inherited defaults, then the overrides."""
    candidate = Path(name_or_path)
    if candidate.suffix in (".yaml", ".yml") and candidate.exists():
        experiment_path = candidate
    else:
        experiment_path = CONFIG_ROOT / "experiment" / f"{name_or_path}.yaml"
    experiment = _read_yaml(experiment_path)

    inherited: dict[str, Any] = {"name": experiment_path.stem}
    for key, folder in (("model", "model"), ("data", "data"), ("train", "train")):
        defaults = experiment.get("defaults", {})
        reference = defaults.get(key)
        if isinstance(reference, str):
            inherited = _deep_merge(inherited, {key: _read_yaml(CONFIG_ROOT / folder / f"{reference}.yaml")})
    merged = _deep_merge(inherited, experiment)
    if overrides:
        merged = _deep_merge(merged, parse_overrides(overrides))
    merged.setdefault("hydraulics", dict(_HYDRAULIC_DEFAULTS))
    merged["hydraulics"] = {**_HYDRAULIC_DEFAULTS, **dict(merged.get("hydraulics", {}))}
    try:
        recorded_path = str(experiment_path.resolve().relative_to(CONFIG_ROOT.parent))
    except ValueError:
        # A configuration outside the tree is recorded by name, never by host path.
        recorded_path = experiment_path.name
    merged["experiment_path"] = recorded_path
    return merged


def grid_from_config(payload: Mapping[str, Any]) -> GridSpec:
    data = dict(payload.get("data", {}))
    shape = tuple(int(value) for value in data.get("shape", (112, 96, 96)))
    spacing = tuple(float(value) for value in data.get("spacing_zyx", (1.0, 0.6, 0.6)))
    return GridSpec(shape=shape, spacing_zyx=spacing)  # type: ignore[arg-type]


def train_config_from_payload(payload: Mapping[str, Any]) -> TrainConfig:
    section = dict(payload.get("train", {}))
    known = set(TrainConfig.__dataclass_fields__)
    filtered = {key: value for key, value in section.items() if key in known}
    return TrainConfig(**filtered)


@dataclass
class ExperimentConfig:
    """The resolved configuration of one experiment."""

    name: str
    raw: dict[str, Any] = field(default_factory=dict)
    path: str | None = None

    @property
    def grid(self) -> GridSpec:
        return grid_from_config(self.raw)

    @property
    def train(self) -> TrainConfig:
        return train_config_from_payload(self.raw)

    @property
    def encoder_name(self) -> str:
        return str(dict(self.raw.get("model", {})).get("encoder", "foundation_trunk"))

    @property
    def width_mult(self) -> float:
        return float(dict(self.raw.get("model", {})).get("width_mult", 1.0))

    @property
    def decoder_width_mult(self) -> float:
        return float(dict(self.raw.get("model", {})).get("decoder_width_mult", 1.0))

    @property
    def reported_total_millions(self) -> float | None:
        value = dict(self.raw.get("model", {})).get("reported_total_millions")
        return None if value is None else float(value)

    @property
    def reference_mode(self) -> str:
        return str(dict(self.raw.get("hydraulics", {})).get("reference_mode", "max"))

    @property
    def upsample_factor(self) -> int:
        return int(dict(self.raw.get("hydraulics", {})).get("upsample_factor", 2))

    @property
    def cohort_size(self) -> int:
        return int(dict(self.raw.get("data", {})).get("n_patients", 240))

    @property
    def seed(self) -> int:
        return int(self.train.seed)

    def documented(self) -> dict[str, Any]:
        """The configuration as it will be reported, with the resolved values."""
        return {
            "experiment": self.name,
            "path": self.path,
            "encoder": self.encoder_name,
            "width_mult": self.width_mult,
            "decoder_width_mult": self.decoder_width_mult,
            "reported_total_millions": self.reported_total_millions,
            "grid_shape": list(self.grid.shape),
            "grid_spacing_zyx": list(self.grid.spacing_zyx),
            "cohort_patients": self.cohort_size,
            "reference_mode": self.reference_mode,
            "upsample_factor": self.upsample_factor,
            "train": self.train.record(),
        }


def load_experiment(name_or_path: str, overrides: Sequence[str] = ()) -> ExperimentConfig:
    payload = resolve_experiment(name_or_path, overrides)
    return ExperimentConfig(
        name=str(payload.get("name", "experiment")), raw=payload, path=payload.get("experiment_path")
    )


def config_files() -> list[Path]:
    """Every shipped configuration file, for the integrity manifest."""
    if not CONFIG_ROOT.exists():
        return []
    return sorted(path for path in CONFIG_ROOT.rglob("*.yaml"))
