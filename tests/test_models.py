"""Encoder families, the anisotropic decoder, the branches and the comparators."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from conductance_ctv.models.baselines import (
    DeepComparator,
    RadiomicsComparator,
    diameter_rule_scores,
    fixture_radiomic_dataset,
    radiomic_features,
)
from conductance_ctv.models.branches import (
    DESCRIPTOR_DIM,
    SEVERITY_INDEX,
    AppearanceBranch,
    DescriptorBranch,
    DescriptorOnlySeverity,
    FusionHead,
    branch_shapes,
    descriptor_slice,
    names_for,
)
from conductance_ctv.models.classifier import (
    ConductanceTypeModel,
    TemperatureCalibrator,
    apply_temperature_,
    decision_rule,
    group_thresholds,
    operating_point,
    threshold_sensitivity_band,
)
from conductance_ctv.models.decoder import IN_PLANE_SCALE, AnisotropicDecoder3D, anisotropic_upsample
from conductance_ctv.models.encoders import (
    ENCODER_SPECS,
    AttentionCNN3D,
    DenseNet3D,
    FoundationTrunk3D,
    ResNet3D,
    ViT3D,
    build_encoder,
    count_parameters,
    count_parameters_millions,
    encoder_spec,
    parameter_table,
    receptive_span_voxels,
    resolve_width_bracket,
    stem_reduction,
)
from conductance_ctv.models.unet3d import (
    LUMEN_CLASSES,
    SEGMENTATION_ENCODERS,
    appearance_descriptor,
    build_segmentation_network,
    encoder_from_name,
    feature_map,
    upsampling_ratio,
)


@pytest.fixture(scope="module")
def tiny_input() -> torch.Tensor:
    return torch.zeros(1, 1, 16, 32, 32)


def test_encoder_families_forward_and_report_width(tiny_input) -> None:
    builders = {
        "resnet": lambda: ResNet3D(widths=(16, 32, 64, 128), blocks=(1, 1, 2, 1)),
        "densenet": lambda: DenseNet3D(init_features=16, growth_rate=8, block_config=(2, 2, 2, 2)),
        "attention": lambda: AttentionCNN3D(widths=(8, 16, 32, 64), depths=(1, 1, 2, 1)),
        "foundation": lambda: FoundationTrunk3D(widths=(8, 16, 32, 64, 128), blocks=(1, 1, 2, 1)),
        "vit": lambda: ViT3D(dim=48, depth=2, heads=4, patch=(4, 8, 8)),
    }
    for family, builder in builders.items():
        model = builder()
        assert hasattr(model, "feature_channels")
        out = model(tiny_input)
        assert out.shape[0] == 1
        assert count_parameters(model) > 0
        assert count_parameters_millions(model) > 0.0
        if family == "vit":
            assert model.grid_shape((16, 32, 32)) == (4, 4, 4)
            grid = model.token_grid(tiny_input)
            assert grid.shape[1] == 48
            assert model.pooled(tiny_input).shape == (1, 48)
        else:
            skips = model.forward_skips(tiny_input)
            assert len(skips) >= 2
            assert len(model.ladder) == len(skips)


def test_build_encoder_by_name_and_spec() -> None:
    model = build_encoder("resnet", widths=(16, 32, 64, 128), blocks=(1, 1, 2, 1))
    assert isinstance(model, ResNet3D)
    with pytest.raises(KeyError):
        build_encoder("nonsense")
    spec = encoder_spec("foundation_trunk")
    assert spec.reported_params_millions == 131.0
    with pytest.raises(KeyError):
        encoder_spec("nonsense")


def test_encoder_specs_are_the_reported_families() -> None:
    names = {spec.name for spec in ENCODER_SPECS}
    assert "foundation_trunk" in names
    assert "universal_prompted" in names
    assert all(spec.reported_params_millions > 0.0 for spec in ENCODER_SPECS)


def test_width_resolver_reaches_a_reachable_target() -> None:
    def factory(multiplier: float) -> torch.nn.Module:
        return AttentionCNN3D(widths=(8, 16, 32), depths=(1, 1, 1), width_mult=multiplier)

    baseline = count_parameters_millions(factory(1.0))
    target = baseline * 4.0
    resolution = resolve_width_bracket(factory, target, low=0.5, high=8.0)
    assert resolution.lower_millions <= target <= resolution.upper_millions
    assert min(abs(resolution.lower_millions - target), abs(resolution.upper_millions - target)) == pytest.approx(
        abs(resolution.achieved_millions - target)
    )
    assert resolution.multiplier > 1.0


def test_width_resolver_rejects_an_unreachable_target() -> None:
    def factory(multiplier: float) -> torch.nn.Module:
        return AttentionCNN3D(widths=(8, 16), depths=(1, 1), width_mult=multiplier)

    with pytest.raises(ValueError):
        resolve_width_bracket(factory, 1e6, low=0.5, high=2.0)


def test_parameter_table_reports_measured_counts() -> None:
    table = parameter_table(resolve=False)
    assert set(table) == {spec.name for spec in ENCODER_SPECS}
    for entry in table.values():
        assert entry["measured_millions"] > 0.0
        assert "resolved_width_mult" not in entry


def test_decoder_upsamples_in_plane_only(tiny_input) -> None:
    features = torch.zeros(1, 16, 4, 8, 8)
    upsampled = anisotropic_upsample(features)
    assert upsampled.shape[2] == features.shape[2]
    assert upsampled.shape[3] == features.shape[3] * IN_PLANE_SCALE[1]
    decoder = AnisotropicDecoder3D(encoder_channels=(8, 16), feature_channels=16, decoder_widths=(16, 8))
    logits = decoder(torch.zeros(1, 16, 4, 8, 8))
    assert logits.shape[1] == 2


def test_decoder_with_skips_and_multiscale(tiny_input) -> None:
    encoder = AttentionCNN3D(widths=(8, 16, 32), depths=(1, 1, 1))
    skips = encoder.forward_skips(tiny_input)
    decoder = AnisotropicDecoder3D(
        encoder_channels=tuple(int(value) for value in encoder.ladder),
        feature_channels=int(encoder.feature_channels),
        deep_supervision=True,
    )
    logits = decoder.forward_with_skips(skips)
    assert logits.shape[1] == 2
    multi = decoder.forward_multiscale(skips)
    assert len(multi) >= 1


def test_segmentation_network_shapes_and_parameter_report() -> None:
    for name in ("resnet50_unet", "attention3d_unet", "foundation_trunk", "universal_transferred"):
        built = build_segmentation_network(name, width_mult=0.05)
        logits = built.network(torch.zeros(1, 1, 16, 32, 32))
        assert logits.shape[1] == LUMEN_CLASSES
        assert tuple(logits.shape[2:]) == (16, 32, 32)
        assert built.parameters_millions > 0.0
    assert "foundation_trunk" in SEGMENTATION_ENCODERS


def test_feature_map_and_appearance_descriptor(tiny_input) -> None:
    encoder = encoder_from_name("attention3d_unet", width_mult=0.05)
    features = feature_map(encoder, tiny_input)
    assert features.ndim == 5
    pooled = appearance_descriptor(encoder, tiny_input)
    assert pooled.shape == (1, int(encoder.feature_channels))
    tube = torch.zeros(1, 1, 16, 32, 32)
    tube[:, :, 4:12, 8:24, 8:24] = 1.0
    pooled_tube = appearance_descriptor(encoder, tiny_input, tube)
    assert pooled_tube.shape == pooled.shape
    empty = torch.zeros_like(tube)
    assert appearance_descriptor(encoder, tiny_input, empty).shape == pooled.shape
    ratio = upsampling_ratio(encoder, tiny_input)
    assert ratio >= 1.0


def test_foundation_trunk_projects_the_deepest_stage(tiny_input) -> None:
    trunk = FoundationTrunk3D(widths=(8, 16, 32, 64, 96), blocks=(1, 1, 1, 1), feature_dim=48)
    skips = trunk.forward_skips(tiny_input)
    assert skips[-1].shape[1] == 48
    assert sum(1 for _ in skips) == 5


def test_stem_reduction_and_receptive_span() -> None:
    assert stem_reduction((16, 32, 32)) == (16, 16, 16)
    assert receptive_span_voxels(1) == 3
    assert receptive_span_voxels(3) > receptive_span_voxels(2)


def test_descriptor_branch_shapes() -> None:
    branch = DescriptorBranch(hidden=32, embedding=16)
    out = branch(torch.zeros(4, DESCRIPTOR_DIM))
    assert out.shape == (4, 16)
    with pytest.raises(ValueError):
        branch(torch.zeros(4, 5))


def test_appearance_branch_and_fusion() -> None:
    appearance = AppearanceBranch(input_dim=64, hidden=32, embedding=16)
    descriptor = DescriptorBranch(hidden=32, embedding=16)
    fusion = FusionHead(16, 16, n_classes=4, hidden=24)
    fused = fusion(descriptor(torch.zeros(3, DESCRIPTOR_DIM)), appearance(torch.zeros(3, 64)))
    assert fused["logits"].shape == (3, 4)
    assert fused["severity_raw"].shape == (3,)
    assert torch.allclose(fusion.temperature_scale(fused["logits"]), fused["logits"])


def test_severity_head_ignores_external_input() -> None:
    head = DescriptorOnlySeverity(16)
    embedding = torch.ones(2, 16)
    assert torch.allclose(head(embedding), head(embedding.clone()))


def test_descriptor_slice_and_names() -> None:
    index = descriptor_slice("CIV", "resistance_ratio")
    assert index == SEVERITY_INDEX
    assert names_for([index]) == ["CIV_resistance_ratio"]
    with pytest.raises(KeyError):
        descriptor_slice("AORTA", "shape_factor")
    with pytest.raises(KeyError):
        descriptor_slice("CIV", "nonsense")
    assert branch_shapes(21, 64)["descriptor"] == (21,)


def test_type_model_forward_and_trainable_groups() -> None:
    built = build_segmentation_network("attention3d_unet", width_mult=0.05)
    model = ConductanceTypeModel(built.network)
    output = model(torch.zeros(2, 1, 16, 32, 32), torch.zeros(2, DESCRIPTOR_DIM))
    assert output.logits.shape == (2, 4)
    assert torch.allclose(output.probabilities.sum(dim=1), torch.ones(2))
    groups = model.trainable_parameter_groups(1e-5, 1e-4)
    assert len(groups) == 2
    model.freeze_segmentation(True)
    assert not any(parameter.requires_grad for parameter in model.segmenter.parameters())
    assert torch.isfinite(model.segment(torch.zeros(1, 1, 16, 32, 32))).all()


def test_temperature_calibrator_lowers_the_nll() -> None:
    # Over-confident scores: the labels are drawn from the same scores at a
    # temperature of three, so the fitted temperature has to rise towards it.
    rng = np.random.default_rng(3)
    logits = rng.normal(0.0, 1.0, size=(1500, 4))
    scaled = logits / 3.0
    exponent = np.exp(scaled - scaled.max(axis=1, keepdims=True))
    truth = exponent / exponent.sum(axis=1, keepdims=True)
    labels = np.array([rng.choice(4, p=row) for row in truth], dtype=np.int64)
    calibrator = TemperatureCalibrator(1.0)
    rows = np.arange(labels.shape[0])
    before = TemperatureCalibrator._nll(logits, labels, rows, 1.0)
    fitted = calibrator.fit(logits, labels)
    after = TemperatureCalibrator._nll(logits, labels, rows, fitted)
    assert fitted > 1.0
    assert after < before
    probabilities = calibrator.probabilities(logits)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert calibrator.record()["temperature"] == fitted


def test_apply_temperature_writes_into_the_head() -> None:
    built = build_segmentation_network("attention3d_unet", width_mult=0.05)
    model = ConductanceTypeModel(built.network)
    apply_temperature_(model, 2.5)
    assert float(model.fusion.temperature) == pytest.approx(2.5)


def test_decision_rule_matches_its_definition() -> None:
    rng = np.random.default_rng(5)
    probabilities = rng.dirichlet(np.ones(4), size=50)
    for threshold in (0.1, 0.5, 0.9):
        assert np.array_equal(
            decision_rule(probabilities, threshold),
            (probabilities[:, 2] + probabilities[:, 3] >= threshold).astype(np.int64),
        )
    with pytest.raises(ValueError):
        decision_rule(probabilities, 1.5)
    with pytest.raises(ValueError):
        decision_rule(np.zeros((4, 3)), 0.5)


def test_operating_point_and_threshold_band() -> None:
    rng = np.random.default_rng(7)
    labels = rng.integers(1, 5, size=300)
    probabilities = np.eye(4)[labels - 1] * 0.4 + rng.random((300, 4)) * 0.6
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    point = operating_point(probabilities, labels, target_sensitivity=0.6)
    assert 0.0 <= point["threshold"] <= 1.0
    assert 0.0 <= point["sensitivity"] <= 1.0
    band = threshold_sensitivity_band(probabilities, labels, point["threshold"])
    assert band["sensitivity_range"] >= 0.0
    assert band["youden_range"] >= 0.0


def test_group_thresholds_stay_inside_the_band() -> None:
    rng = np.random.default_rng(11)
    labels = rng.integers(1, 5, size=200)
    probabilities = rng.dirichlet(np.ones(4), size=200)
    groups = np.array(["a"] * 100 + ["b"] * 100)
    thresholds = group_thresholds(probabilities, labels, groups, 0.5, tolerance=0.3)
    assert set(thresholds) == {"a", "b"}
    for value in thresholds.values():
        assert 0.0 <= value <= 1.0


def test_diameter_rule_scores_shape_and_direction() -> None:
    from conductance_ctv.data.fixture import generate_cohort

    cohort = generate_cohort(20, seed=41)
    scores = diameter_rule_scores(list(cohort.limbs))
    assert scores.shape == (cohort.n_limbs, 4)
    assert np.all(scores >= 0.0)
    assert np.all(scores <= 1.0)


def test_radiomic_features_are_finite() -> None:
    rng = np.random.default_rng(13)
    mask = np.zeros((24, 24, 24), dtype=bool)
    mask[6:18, 10:14, 10:14] = True
    image = rng.normal(40.0, 10.0, size=mask.shape)
    image[mask] = rng.normal(300.0, 20.0, size=int(mask.sum()))
    features = radiomic_features(mask, image, (1.0, 1.0, 1.0), lesion_length_mm=12.0, stenosis_fraction=0.4)
    assert features.shape == (21,)
    assert np.all(np.isfinite(features))
    assert features[0] > 0.0


def test_radiomic_comparators_fit_and_score() -> None:
    from conductance_ctv.data.fixture import generate_cohort

    cohort = generate_cohort(60, seed=43, grid=_grid())
    dataset = fixture_radiomic_dataset(list(cohort.limbs), grid=(24, 20, 20), seed=3)
    assert dataset.features.shape[0] == cohort.n_limbs
    for learner in ("lr", "gbm", "svm"):
        model = RadiomicsComparator(learner=learner, seed=1).fit(dataset.features, dataset.labels)
        scores = model.decision_scores(dataset.features)
        assert scores.shape == (cohort.n_limbs, 4)
        assert np.allclose(scores.sum(axis=1), 1.0)
        assert model.predict(dataset.features).shape == (cohort.n_limbs,)
    with pytest.raises(KeyError):
        RadiomicsComparator(learner="nonsense").fit(dataset.features, dataset.labels)
    with pytest.raises(RuntimeError):
        RadiomicsComparator(learner="lr").decision_scores(dataset.features)


def test_deep_comparator_forward() -> None:
    encoder = encoder_from_name("attention3d_unet", width_mult=0.05)
    comparator = DeepComparator(encoder, int(encoder.feature_channels))
    logits = comparator(torch.zeros(2, 1, 16, 32, 32))
    assert logits.shape == (2, 4)
    vit = ViT3D(dim=48, depth=2, heads=4, patch=(4, 8, 8))
    comparator_vit = DeepComparator(vit, 48)
    assert comparator_vit(torch.zeros(2, 1, 16, 32, 32)).shape == (2, 4)


def _grid():
    from conductance_ctv.data.schema import GridSpec

    return GridSpec(shape=(24, 20, 20), spacing_zyx=(1.5, 1.2, 1.2))
