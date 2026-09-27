"""The command-line entry points, exercised through their argument parsers."""

from __future__ import annotations

import json

import numpy as np
import pytest

from conductance_ctv.cli.architectures import build_parser as architectures_parser
from conductance_ctv.cli.architectures import measure
from conductance_ctv.cli.evaluate import build_parser as evaluate_parser
from conductance_ctv.cli.hydraulics import build_parser as hydraulics_parser
from conductance_ctv.cli.hydraulics import describe_section_arguments, read_mask, section_from_arguments
from conductance_ctv.cli.train import build_parser as train_parser
from conductance_ctv.cli.verify import build_parser as verify_parser
from conductance_ctv.utils.config import (
    config_files,
    grid_from_config,
    load_experiment,
    parse_overrides,
    repo_root,
    resolve_experiment,
    train_config_from_payload,
)


def test_parsers_accept_their_defaults() -> None:
    assert architectures_parser().parse_args([]).resolve is False
    assert evaluate_parser().parse_args([]).experiment == "main"
    assert hydraulics_parser().parse_args([]).shape == "ellipse"
    assert train_parser().parse_args([]).stage == "both"
    assert verify_parser().parse_args([]).skip_pipeline is False


def test_hydraulics_section_arguments() -> None:
    report = describe_section_arguments(shape="ellipse", a=20.0, b=6.0, step=0.5)
    assert report["channels"] == 1
    assert report["shape_factor"] > 1.0
    assert report["conductance_mm4"] > 0.0
    assert report["elliptic_closed_form_shape_factor"] > report["shape_factor"] - 0.5
    disc = describe_section_arguments(shape="disc", a=10.0, step=0.5)
    finer = describe_section_arguments(shape="disc", a=10.0, step=0.25)
    disc_error = abs(disc["shape_factor"] - 1.0)
    finer_error = abs(finer["shape_factor"] - 1.0)
    # A voxelised disc departs from circularity by the staircase, so kappa sits
    # just above one and the gap halves with the grid step.
    assert 0.0 < disc_error < 0.08
    assert finer_error < disc_error
    septum = describe_section_arguments(shape="septum", channels=3, area=1800.0, step=0.5)
    assert septum["channels"] == 3
    assert abs(septum["shape_factor"] - 3.0) < 0.25
    rectangle = describe_section_arguments(shape="rectangle", a=8.0, b=8.0, step=0.5)
    assert rectangle["conductance_mm4"] > 0.0
    channels = describe_section_arguments(shape="channels", a=4.0, b=3.0, step=0.5)
    assert channels["channels"] == 2
    with pytest.raises(KeyError):
        describe_section_arguments(shape="nonsense")


def test_hydraulics_parser_builds_a_section() -> None:
    arguments = hydraulics_parser().parse_args(["--shape", "disc", "--a", "6", "--step", "0.5"])
    section = section_from_arguments(arguments)
    assert section.occupancy.ndim == 2
    assert section.area > 0.0
    other = hydraulics_parser().parse_args(["--shape", "septum", "--channels", "3", "--area", "900"])
    assert section_from_arguments(other).occupancy.any()


def test_hydraulics_mask_reader(tmp_path) -> None:
    mask = np.zeros((6, 6, 6), dtype=bool)
    mask[1:5, 2:4, 2:4] = True
    path = tmp_path / "mask.npz"
    np.savez_compressed(path, mask=mask, spacing=np.array([1.0, 1.0, 1.0]))
    read, spacing = read_mask(path)
    assert read.shape == mask.shape
    assert spacing == (1.0, 1.0, 1.0)
    np.savez_compressed(tmp_path / "empty.npz", other=mask)
    with pytest.raises(KeyError):
        read_mask(tmp_path / "empty.npz")
    np.savez_compressed(tmp_path / "flat.npz", mask=np.zeros((4, 4), dtype=bool))
    with pytest.raises(ValueError):
        read_mask(tmp_path / "flat.npz")


def test_architecture_measure_without_resolving() -> None:
    rows = measure(resolve=False)
    assert rows
    for row in rows:
        assert row["measured_millions"] > 0.0
        assert "resolution" not in row
    bracketed = measure(resolve=True)
    resolved = [row for row in bracketed if row.get("resolved_width_mult")]
    assert len(resolved) == len(rows)
    for row in resolved:
        assert 2.0 * row["relative_gap"] <= row["ladder_granularity"] + 1e-12
        assert row["resolved_width_mult"] > 0.0


def test_config_overrides_and_resolution() -> None:
    overrides = parse_overrides(["train.epochs=3", "model.width_mult=0.5", "data.n_patients=12"])
    assert overrides["train"]["epochs"] == 3
    assert overrides["model"]["width_mult"] == 0.5
    with pytest.raises(ValueError):
        parse_overrides(["train.epochs"])
    payload = resolve_experiment("_smoke", ["train.epochs=3"])
    assert payload["train"]["epochs"] == 3
    config = load_experiment("_smoke", ["train.epochs=3"])
    assert config.train.epochs == 3
    assert config.name == "_smoke"
    assert config.encoder_name == "foundation_trunk"
    assert config.grid.shape == tuple(payload["data"]["shape"])
    assert config.upsample_factor == 2
    assert config.reference_mode == "max"
    documented = config.documented()
    assert documented["experiment"] == "_smoke"
    assert documented["train"]["epochs"] == 3
    with pytest.raises(FileNotFoundError):
        load_experiment("does_not_exist")


def test_every_experiment_resolves_and_the_smoke_file_is_marked() -> None:
    files = config_files()
    assert len(files) >= 40
    experiments = sorted(path.stem for path in (repo_root() / "configs" / "experiment").glob("*.yaml"))
    assert "main" in experiments
    assert "_smoke" in experiments
    smoke = (repo_root() / "configs" / "experiment" / "_smoke.yaml").read_text(encoding="utf-8")
    assert "for unit-test smoke only; do not use for reporting" in smoke
    for name in experiments:
        config = load_experiment(name)
        assert config.train.batch_size >= 1
        assert config.train.epochs >= 1
        assert grid_from_config(config.raw).shape[0] >= 3


def test_main_configuration_matches_the_reported_profile() -> None:
    config = load_experiment("main")
    train = config.train
    assert train.optimizer == "adamw"
    assert train.scheduler == "warmup_cosine"
    assert train.effective_batch_size == train.batch_size * train.grad_accumulation
    assert train.weight_decay > 0.0
    assert train.seed == 20260101
    assert config.raw["model"]["reported_params_millions"] == 131.0
    assert config.raw["evaluation"]["bootstrap_resamples"] == 2000
    assert len(config.raw["evaluation"]["seeds"]) == 5


def test_train_config_from_payload_ignores_unknown_keys() -> None:
    config = train_config_from_payload({"train": {"epochs": 5, "nonsense": 1}})
    assert config.epochs == 5


def test_evaluate_and_verify_parsers_take_overrides() -> None:
    arguments = evaluate_parser().parse_args(["main", "train.seed=7", "--steps", "3"])
    assert arguments.experiment == "main"
    assert arguments.steps == 3
    assert arguments.override == ["train.seed=7"]
    verify = verify_parser().parse_args(["--skip-pipeline", "--quiet"])
    assert verify.skip_pipeline and verify.quiet


def test_hydraulics_entry_point_runs(capsys) -> None:
    from conductance_ctv.cli.hydraulics import main

    assert main(["--shape", "disc", "--a", "8", "--step", "0.5", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["channels"] == 1
    assert payload["shape"] == "disc"


def test_architectures_entry_point_runs(capsys) -> None:
    from conductance_ctv.cli.architectures import main

    assert main(["--json"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert len(rows) == 6
