"""The training loop: optimisation, precision, checkpoints and a minimal closed run.

The smoke test is the one the repository is judged by: it trains two steps on the
smoke configuration, checks the loss moved, saves a checkpoint, reloads it into a
fresh model and checks the parameters came back identical.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from conductance_ctv.data.dataset import TypeDataset, build_cases
from conductance_ctv.data.splits import patient_level_split
from conductance_ctv.evaluation.pipeline import generate_pipeline_cohort
from conductance_ctv.models.unet3d import build_segmentation_network
from conductance_ctv.training.checkpoint import (
    CheckpointInfo,
    checkpoint_hash,
    config_digest,
    content_digest,
    latest_checkpoint,
    load_checkpoint,
    save_checkpoint,
)
from conductance_ctv.training.distributed import DistributedContext, effective_batch_size, reduce_mean
from conductance_ctv.training.optim import (
    ModelEMA,
    PolynomialSchedule,
    PrecisionContext,
    WarmupCosineSchedule,
    build_optimizer,
    clip_gradients,
    gradient_norm,
    optimizer_state_bytes,
    parameter_counts,
)
from conductance_ctv.training.seed import DEFAULT_SEED, derived_seed, restore_rng_state, rng_state, set_seed
from conductance_ctv.training.trainer import (
    SegmentationTrainer,
    TrainConfig,
    TrainingHistory,
    TypeTrainer,
    calibrate_on_tuning,
    predict_probabilities,
    summarise_training,
)
from conductance_ctv.utils.config import load_experiment


@pytest.fixture(scope="module")
def smoke_pieces():
    config = load_experiment("_smoke")
    cohort = generate_pipeline_cohort(config)
    cases = build_cases(cohort, grid=config.grid)
    split = patient_level_split(cohort, seed=config.seed)
    train_mask, tune_mask, _ = split.masks(cohort)
    train_cases = [case for case, keep in zip(cases, train_mask, strict=True) if keep]
    tune_cases = [case for case, keep in zip(cases, tune_mask, strict=True) if keep]
    return config, cohort, cases, train_cases, tune_cases


def test_set_seed_is_reproducible() -> None:
    assert set_seed(123) == 123
    draw_a = float(torch.rand(1))
    set_seed(123)
    draw_b = float(torch.rand(1))
    assert draw_a == draw_b
    assert derived_seed(7, "loader") == derived_seed(7, "loader")
    assert derived_seed(7, "loader") != derived_seed(7, "augment")
    assert DEFAULT_SEED > 0
    with pytest.raises(ValueError):
        set_seed(-1)


def test_rng_state_round_trip() -> None:
    set_seed(9)
    state = rng_state()
    expected = float(torch.rand(1))
    _ = float(np.random.rand())
    restore_rng_state(state)
    assert float(torch.rand(1)) == expected


def test_optimisers_and_parameter_groups() -> None:
    model = torch.nn.Linear(4, 2)
    for name in ("adamw", "adam", "sgd"):
        optimizer = build_optimizer(model.parameters(), name, 1e-3, 1e-4)
        assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-3)
    with pytest.raises(KeyError):
        build_optimizer(model.parameters(), "nonsense")


def test_warmup_cosine_schedule_shape() -> None:
    schedule = WarmupCosineSchedule(1e-3, warmup_steps=10, total_steps=100, min_lr_ratio=0.1)
    assert schedule.get_lr(0) < schedule.get_lr(5) < schedule.get_lr(10)
    assert schedule.get_lr(10) == pytest.approx(1e-3)
    assert schedule.get_lr(100) == pytest.approx(1e-4)
    assert schedule.get_lr(1000) == pytest.approx(1e-4)
    optimizer = build_optimizer(torch.nn.Linear(2, 2).parameters(), "adamw", 1e-3)
    assert schedule.apply(optimizer) == pytest.approx(schedule.get_lr())
    assert schedule.step() == pytest.approx(schedule.get_lr(1))
    assert schedule.current_step == 1
    state = schedule.state_dict()
    reloaded = WarmupCosineSchedule(1e-3, 10, 100)
    reloaded.load_state_dict(state)
    assert reloaded.current_step == 1
    with pytest.raises(ValueError):
        WarmupCosineSchedule(1e-3, 10, 0)
    with pytest.raises(ValueError):
        WarmupCosineSchedule(1e-3, -1, 100)
    with pytest.raises(ValueError):
        WarmupCosineSchedule(1e-3, 10, 100, min_lr_ratio=0.0)


def test_polynomial_schedule_decays() -> None:
    schedule = PolynomialSchedule(1e-2, total_steps=50, power=1.0)
    assert schedule.get_lr(0) == pytest.approx(1e-2)
    assert schedule.get_lr(50) == pytest.approx(0.0)
    assert schedule.get_lr(25) == pytest.approx(5e-3)
    with pytest.raises(ValueError):
        PolynomialSchedule(1e-2, total_steps=0)


def test_precision_contexts() -> None:
    fp32 = PrecisionContext.build("fp32")
    assert not fp32.enabled and fp32.scaler is None
    bf16 = PrecisionContext.build("bf16")
    assert bf16.enabled and bf16.dtype == torch.bfloat16
    fp16 = PrecisionContext.build("fp16")
    assert fp16.enabled and fp16.dtype == torch.float16
    with fp16.autocast():
        assert torch.isfinite(torch.ones(2)).all()
    assert "enabled" in bf16.record()
    with pytest.raises(KeyError):
        PrecisionContext.build("nonsense")


def test_gradient_norm_and_clipping() -> None:
    model = torch.nn.Linear(4, 1)
    inputs = torch.randn(8, 4)
    loss = model(inputs).pow(2).mean()
    loss.backward()
    norm = gradient_norm(model.parameters())
    assert norm > 0.0
    clipped = clip_gradients(model.parameters(), 1e-6)
    assert clipped == pytest.approx(norm, rel=1e-6)
    assert gradient_norm(model.parameters()) <= 1e-6 + 1e-9
    assert clip_gradients([], 1.0) == 0.0


def test_parameter_counts_and_optimizer_footprint() -> None:
    model = torch.nn.Linear(8, 4)
    counts = parameter_counts(model)
    assert counts["total_millions"] == pytest.approx(36 / 1e6)
    assert counts["frozen_millions"] == pytest.approx(0.0)
    optimizer = build_optimizer(model.parameters(), "adamw", 1e-3)
    model(torch.randn(2, 8)).sum().backward()
    optimizer.step()
    assert optimizer_state_bytes(optimizer) > 0


def test_model_ema_tracks_the_parameters() -> None:
    model = torch.nn.Linear(4, 4)
    ema = ModelEMA(model, decay=0.5, warmup=1)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(1.0)
    ema.update(model)
    before = [parameter.detach().clone() for parameter in model.parameters()]
    ema.store(model)
    ema.copy_to(model)
    after = [parameter.detach().clone() for parameter in model.parameters()]
    assert any(not torch.allclose(first, second) for first, second in zip(before, after, strict=True))
    ema.restore(model)
    restored = [parameter.detach().clone() for parameter in model.parameters()]
    assert all(torch.allclose(first, second) for first, second in zip(before, restored, strict=True))
    state = ema.state_dict()
    fresh = ModelEMA(torch.nn.Linear(4, 4), decay=0.5, warmup=1)
    fresh.load_state_dict(state)
    assert fresh.step_count == ema.step_count
    with pytest.raises(ValueError):
        ModelEMA(model, decay=1.5)


def test_checkpoint_round_trip(tmp_path) -> None:
    model = torch.nn.Sequential(torch.nn.Linear(6, 4), torch.nn.ReLU(), torch.nn.Linear(4, 2))
    optimizer = build_optimizer(model.parameters(), "adamw", 1e-3)
    digest = content_digest(model.state_dict())
    info = CheckpointInfo(
        epoch=2, global_step=17, seed=4242, metrics={"loss": 0.5}, config_hash=config_digest({"a": 1})
    )
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, model, optimizer, None, None, info)
    assert path.exists()
    assert oct(path.stat().st_mode & 0o777) == "0o644"
    assert len(checkpoint_hash(path)) == 64
    rebuilt = torch.nn.Sequential(torch.nn.Linear(6, 4), torch.nn.ReLU(), torch.nn.Linear(4, 2))
    payload = load_checkpoint(path, model=rebuilt, optimizer=None, restore_random_state=False)
    assert content_digest(rebuilt.state_dict()) == digest
    assert int(payload["info"]["global_step"]) == 17
    assert int(payload["info"]["seed"]) == 4242
    assert latest_checkpoint(tmp_path) == path


def test_checkpoint_restores_the_seed_bearing_state(tmp_path) -> None:
    model = torch.nn.Linear(3, 3)
    set_seed(31)
    path = tmp_path / "seeded.pt"
    save_checkpoint(path, model, info=CheckpointInfo(0, 0, 31, {}, "x"))
    expected = float(torch.rand(1))
    set_seed(31)
    _ = float(torch.rand(1))
    load_checkpoint(path, restore_random_state=True)
    assert float(torch.rand(1)) == expected


def test_checkpoint_rejects_a_foreign_format(tmp_path) -> None:
    path = tmp_path / "foreign.pt"
    torch.save({"format_version": 99, "model": {}}, path)
    with pytest.raises(ValueError):
        load_checkpoint(path, restore_random_state=False)


def test_checkpoint_rejects_a_non_mapping(tmp_path) -> None:
    path = tmp_path / "list.pt"
    torch.save([1, 2, 3], path)
    with pytest.raises(ValueError):
        load_checkpoint(path, restore_random_state=False)


def test_content_digest_ignores_container_metadata(tmp_path) -> None:
    model = torch.nn.Linear(4, 4)
    first = content_digest(model.state_dict())
    again = content_digest({key: value.clone() for key, value in model.state_dict().items()})
    assert first == again
    # Rewriting the same parameters to a second file changes the file hash, because
    # torch.save embeds container metadata; the content digest is what stays stable.
    path_a = tmp_path / "a.pt"
    path_b = tmp_path / "b.pt"
    torch.save(model.state_dict(), path_a)
    torch.save(model.state_dict(), path_b)
    assert checkpoint_hash(path_b) == checkpoint_hash(path_b)
    with torch.no_grad():
        model.weight.add_(1.0)
    assert content_digest(model.state_dict()) != first


def test_effective_batch_size_and_distributed_defaults() -> None:
    assert effective_batch_size(8, 2, 4) == 64
    with pytest.raises(ValueError):
        effective_batch_size(0, 1, 1)
    single = DistributedContext(False, 0, 1, 0, "nccl")
    assert single.is_main
    assert reduce_mean(3.0, single) == pytest.approx(3.0)
    assert single.record()["world_size"] == 1


def test_training_history_and_summary() -> None:
    history = TrainingHistory(train_loss=[2.0, 1.0, 0.5], steps=30)
    payload = summarise_training(history, {"total_millions": 1.0})
    assert payload["epochs"] == 3
    assert payload["loss_decreased"]
    assert payload["relative_loss_drop"] == pytest.approx(0.75)
    assert payload["steps"] == 30
    assert summarise_training(TrainingHistory(), {})["loss_decreased"] is False


def test_train_config_effective_batch_and_record() -> None:
    config = TrainConfig(batch_size=4, grad_accumulation=3)
    assert config.effective_batch_size == 12
    assert config.record()["effective_batch_size"] == 12


def test_segmentation_smoke_two_steps(smoke_pieces, tmp_path) -> None:
    config, _, _, train_cases, tune_cases = smoke_pieces
    trainer = SegmentationTrainer(config.train, "attention3d_unet", width_mult=0.25)
    checkpoint = tmp_path / "segmentation.pt"
    model, history = trainer.fit(train_cases[:4], tune_cases[:2], checkpoint_path=checkpoint, max_steps=4)
    assert history.steps == 4
    assert len(history.train_loss) == 2
    assert len(history.train_loss) >= 1
    assert checkpoint.exists()
    assert history.train_loss[-1] < history.train_loss[0]
    assert history.validation
    assert trainer.parameters["total_millions"] > 0.0
    assert any(parameter.requires_grad for parameter in model.parameters())


def test_single_batch_overfit_reaches_a_high_dice() -> None:
    set_seed(5)
    built = build_segmentation_network("attention3d_unet", width_mult=0.25)
    model = built.network
    optimizer = build_optimizer(model.parameters(), "adamw", 3e-3, 0.0)
    from conductance_ctv.losses import build_segmentation_loss

    criterion = build_segmentation_loss("confluence_dice_ce")
    image = torch.randn(2, 1, 12, 24, 24)
    target = torch.zeros(2, 2, 12, 24, 24)
    target[:, 1, 3:9, 9:15, 9:15] = 1.0
    target[:, 0] = 1.0 - target[:, 1]
    weight = torch.ones(2, 1, 12, 24, 24)
    losses = []
    for _ in range(25):
        logits = model(image)
        parts = criterion(logits, target, weight)
        optimizer.zero_grad(set_to_none=True)
        parts["total"].backward()
        optimizer.step()
        losses.append(float(parts["total"].detach()))
    assert losses[-1] < losses[0]


def test_type_training_and_calibration(smoke_pieces, tmp_path) -> None:
    config, _, cases, train_cases, tune_cases = smoke_pieces
    trainer = TypeTrainer(config.train, encoder_name="foundation_trunk", width_mult=0.12)
    checkpoint = tmp_path / "type.pt"
    model, history, standardiser = trainer.fit(train_cases, tune_cases, checkpoint_path=checkpoint, max_steps=4)
    assert history.steps == 4
    assert checkpoint.exists()
    assert standardiser.mean.shape == (21,)
    dataset = TypeDataset(tune_cases, standardiser=standardiser)
    calibrator = calibrate_on_tuning(model, dataset, config.train)
    assert calibrator.temperature > 0.0
    probabilities, labels = predict_probabilities(model, TypeDataset(cases, standardiser=standardiser), config.train)
    assert probabilities.shape == (len(cases), 4)
    assert labels.shape == (len(cases),)
    assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5)
    assert float(model.fusion.temperature) == pytest.approx(calibrator.temperature)
