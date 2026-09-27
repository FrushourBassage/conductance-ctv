"""The evaluation protocol: one patient-level split, one sealed region, clustered intervals.

Every baseline is scored on the identical patient-level split, the external
region is scored once, and every interval is a percentile interval over patient
clusters. A same-source random split is never substituted for the external
evaluation, which is the failure mode Algorithm 3 exists to prevent.

Ref: Algorithm 3; Sec. 6.8.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

from conductance_ctv.data.schema import Site
from conductance_ctv.metrics.bootstrap import BOOTSTRAP_RESAMPLES, BootstrapResult, clustered_bootstrap
from conductance_ctv.metrics.inference_stats import DeLongResult, delong_auc_pair, mcnemar
from conductance_ctv.metrics.special import two_sided_normal_p
from conductance_ctv.utils.types import FloatArray, IntArray

METRIC_SET: tuple[str, ...] = (
    "macro_auroc",
    "macro_f1",
    "kappa",
    "balanced_accuracy",
    "per_type_auroc",
)


@dataclass(frozen=True)
class PredictionBundle:
    """Predictions of one model on one partition, with the units they belong to."""

    name: str
    limb_ids: tuple[str, ...]
    patient_index: IntArray
    labels: IntArray
    probabilities: FloatArray
    sites: tuple[Site, ...]
    phases: tuple[str, ...] = ()
    vendors: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        count = len(self.limb_ids)
        if self.probabilities.shape[0] != count or self.labels.shape[0] != count:
            raise ValueError("the prediction arrays must agree on the unit count")
        if self.patient_index.shape[0] != count:
            raise ValueError("every unit needs a patient index")

    @property
    def predictions(self) -> IntArray:
        assignment: IntArray = np.argmax(self.probabilities, axis=1) + 1
        return assignment

    def take(self, rows: IntArray) -> PredictionBundle:
        """Select rows by index, with repetition; this is the resampling operation.

        A clustered resample necessarily repeats a patient, so the selected bundle
        is longer than the original and its patient index repeats too.
        """
        index = np.asarray(rows, dtype=np.int64)
        return PredictionBundle(
            name=self.name,
            limb_ids=tuple(self.limb_ids[int(value)] for value in index),
            patient_index=self.patient_index[index],
            labels=self.labels[index],
            probabilities=self.probabilities[index],
            sites=tuple(self.sites[int(value)] for value in index),
            phases=tuple(self.phases[int(value)] for value in index) if self.phases else (),
            vendors=tuple(self.vendors[int(value)] for value in index) if self.vendors else (),
        )

    def subset(self, mask: IntArray) -> PredictionBundle:
        rows = np.asarray(mask, dtype=bool)
        if rows.shape[0] != len(self.limb_ids):
            raise ValueError("a boolean selection must cover every unit")
        return self.take(np.flatnonzero(rows).astype(np.int64))

    def with_name(self, name: str) -> PredictionBundle:
        return PredictionBundle(
            name=name,
            limb_ids=self.limb_ids,
            patient_index=self.patient_index,
            labels=self.labels,
            probabilities=self.probabilities,
            sites=self.sites,
            phases=self.phases,
            vendors=self.vendors,
        )


@dataclass(frozen=True)
class MetricInterval:
    """One metric of one model with its clustered interval."""

    model: str
    metric: str
    unit: str
    result: BootstrapResult

    def record(self) -> dict[str, object]:
        return {"model": self.model, "metric": self.metric, **self.result.record()}


@dataclass(frozen=True)
class PairedComparison:
    """The proposed model against one comparator on the identical units."""

    comparator: str
    metric: str
    delong: DeLongResult | None
    mcnemar: dict[str, float] | None

    def record(self) -> dict[str, object]:
        payload: dict[str, object] = {"comparator": self.comparator, "metric": self.metric}
        if self.delong is not None:
            payload["delong"] = self.delong.record()
        if self.mcnemar is not None:
            payload["mcnemar"] = self.mcnemar
        return payload


@dataclass
class ProtocolResult:
    """Everything the protocol produced for one evaluation run."""

    metrics: list[MetricInterval] = field(default_factory=list)
    paired: list[PairedComparison] = field(default_factory=list)
    units: dict[str, int] = field(default_factory=dict)

    def record(self) -> dict[str, object]:
        return {
            "metrics": [entry.record() for entry in self.metrics],
            "paired": [entry.record() for entry in self.paired],
            "units": self.units,
        }


def metric_function(name: str) -> Callable[[PredictionBundle], float]:
    """Look up a scalar metric of a prediction bundle."""
    from conductance_ctv.metrics.classification import (
        balanced_accuracy,
        cohen_kappa,
        macro_auroc,
        macro_f1,
    )

    table: dict[str, Callable[[PredictionBundle], float]] = {
        "macro_auroc": lambda bundle: macro_auroc(bundle.probabilities, bundle.labels),
        "macro_f1": lambda bundle: macro_f1(bundle.predictions, bundle.labels),
        "kappa": lambda bundle: cohen_kappa(bundle.predictions, bundle.labels),
        "balanced_accuracy": lambda bundle: balanced_accuracy(bundle.predictions, bundle.labels),
    }
    if name not in table:
        raise KeyError(f"unknown metric {name!r}")
    return table[name]


def clustered_metric(
    bundle: PredictionBundle,
    metric: str,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = 0,
) -> MetricInterval:
    """One metric with a percentile interval over patient clusters."""
    function = metric_function(metric)

    def statistic(rows: IntArray) -> float:
        return function(bundle.take(rows))

    result = clustered_bootstrap(statistic, bundle.patient_index, resamples=resamples, seed=seed)
    return MetricInterval(model=bundle.name, metric=metric, unit="patient", result=result)


def unit_counts(bundle: PredictionBundle, limbs_per_patient: IntArray | None = None) -> dict[str, int]:
    """The three nested unit counts the tables report."""
    patients = int(np.unique(bundle.patient_index).size)
    limbs = int(bundle.labels.shape[0])
    del limbs_per_patient
    return {"patients": patients, "limbs": limbs, "segments": limbs * 3}


def paired_comparison(
    proposed: PredictionBundle,
    comparator: PredictionBundle,
    metric: str = "macro_auroc",
    seed: int = 0,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> PairedComparison:
    """DeLong on the primary metric and McNemar on the paired calls.

    Both models must be scored on the identical units; the correspondence is
    checked rather than assumed, because a misaligned pair would produce a
    plausible-looking interval for the wrong question.
    """
    if proposed.limb_ids != comparator.limb_ids:
        raise ValueError("the two bundles must cover the identical units in the identical order")
    if metric == "macro_auroc":
        first = proposed.probabilities[:, 2] + proposed.probabilities[:, 3]
        second = comparator.probabilities[:, 2] + comparator.probabilities[:, 3]
        target = (proposed.labels >= 3).astype(np.int64)
        delong = delong_auc_pair(first, second, target)
    else:
        delong = None
    first_correct = (proposed.predictions == proposed.labels).astype(np.int64)
    second_correct = (comparator.predictions == comparator.labels).astype(np.int64)
    outcome = mcnemar(first_correct, second_correct)
    del seed, resamples
    return PairedComparison(
        comparator=comparator.name,
        metric=metric,
        delong=delong,
        mcnemar=outcome.record(),
    )


def run_protocol(
    models: Sequence[PredictionBundle],
    metrics: Sequence[str] = METRIC_SET,
    reference: str | None = None,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = 0,
) -> ProtocolResult:
    """Score every model with every metric and compare the reference to the rest.

    The reference is the proposed model; the comparison target is the best
    unconstrained comparator, which is selected on the primary metric rather than
    named in advance.
    """
    result = ProtocolResult()
    if not models:
        return result
    result.units = unit_counts(models[0])
    for bundle in models:
        for metric in metrics:
            if metric == "per_type_auroc":
                continue
            result.metrics.append(clustered_metric(bundle, metric, resamples=resamples, seed=seed))
    chosen = reference if reference is not None else models[0].name
    proposed = next((bundle for bundle in models if bundle.name == chosen), models[0])
    best_name = None
    best_value = -np.inf
    for bundle in models:
        if bundle.name == proposed.name:
            continue
        value = metric_function("macro_auroc")(bundle)
        if np.isfinite(value) and value > best_value:
            best_value = value
            best_name = bundle.name
    if best_name is not None:
        best_bundle = next(bundle for bundle in models if bundle.name == best_name)
        result.paired.append(paired_comparison(proposed, best_bundle))
    return result


def subgroup_bundles(bundle: PredictionBundle, attribute: str) -> dict[str, PredictionBundle]:
    """Split a bundle by site, phase or vendor, which is how Table 4's columns are read."""
    if attribute == "site":
        keys = [str(value) for value in bundle.sites]
    elif attribute == "phase":
        keys = list(bundle.phases)
    elif attribute == "vendor":
        keys = list(bundle.vendors)
    else:
        raise KeyError(f"unknown subgroup attribute {attribute!r}")
    out: dict[str, PredictionBundle] = {}
    for key in dict.fromkeys(keys):
        mask = np.array([value == key for value in keys], dtype=bool)
        if int(np.count_nonzero(mask)) == 0:
            continue
        out[key] = bundle.subset(mask)
    return out


def leave_one_site_out(bundle: PredictionBundle) -> dict[str, dict[str, float]]:
    """Per-metric values with every site held out in turn.

    A loss that is uniform across the held-out site folds points to a regional
    effect rather than a site-specific one, which is the distinction the
    manuscript draws from this table.

    Ref: Sec. 3.3 and Supplementary Table S15.
    """
    out: dict[str, dict[str, float]] = {}
    for key, subgroup in subgroup_bundles(bundle, "site").items():
        entry: dict[str, float] = {}
        for metric in ("macro_auroc", "macro_f1", "kappa"):
            entry[metric] = metric_function(metric)(subgroup)
        entry["limbs"] = float(subgroup.labels.shape[0])
        out[key] = entry
    return out


def temporal_halves(
    bundle: PredictionBundle, halves: tuple[Sequence[str], Sequence[str]]
) -> dict[str, dict[str, float]]:
    """Replicate a metric on two accrual halves, for the temporal check of Sec. 3.3."""
    first = set(halves[0])
    second = set(halves[1])
    out: dict[str, dict[str, float]] = {}
    for name, members in (("first_half", first), ("second_half", second)):
        mask = np.array([value in members for value in bundle.limb_ids], dtype=bool)
        if int(np.count_nonzero(mask)) == 0:
            continue
        subgroup = bundle.subset(mask)
        out[name] = {
            metric: float(metric_function(metric)(subgroup)) for metric in ("macro_auroc", "macro_f1", "kappa")
        }
        out[name]["limbs"] = float(subgroup.labels.shape[0])
    if "first_half" in out and "second_half" in out:
        out["difference"] = {
            metric: out["second_half"][metric] - out["first_half"][metric]
            for metric in ("macro_auroc", "macro_f1", "kappa")
        }
    return out


def significance_of_difference(difference: float, standard_error: float) -> float:
    """Two-sided p-value of a difference given its standard error."""
    if standard_error <= 0.0:
        return float("nan")
    return two_sided_normal_p(difference / standard_error)


def probability_average(bundles: Sequence[PredictionBundle]) -> PredictionBundle:
    """Average calibrated probabilities over independently initialised runs.

    Probability averaging is chosen over logit averaging because the decision
    rule operates on the probability scale; the manuscript records that the two
    agree on the aggregate to within 0.001.

    Ref: Sec. 3.5 and Supplementary Table S1.
    """
    if not bundles:
        raise ValueError("at least one bundle is required")
    first = bundles[0]
    stacked = np.stack([bundle.probabilities for bundle in bundles], axis=0)
    return PredictionBundle(
        name=f"{first.name}_avg{len(bundles)}",
        limb_ids=first.limb_ids,
        patient_index=first.patient_index,
        labels=first.labels,
        probabilities=stacked.mean(axis=0),
        sites=first.sites,
        phases=first.phases,
        vendors=first.vendors,
    )


def logit_average(bundles: Sequence[PredictionBundle]) -> PredictionBundle:
    """Average in logit space, the alternative to probability averaging."""
    if not bundles:
        raise ValueError("at least one bundle is required")
    first = bundles[0]
    stacked = np.stack([np.log(np.clip(bundle.probabilities, 1e-12, 1.0)) for bundle in bundles], axis=0).mean(axis=0)
    exponent = np.exp(stacked - stacked.max(axis=1, keepdims=True))
    probabilities = exponent / exponent.sum(axis=1, keepdims=True)
    return PredictionBundle(
        name=f"{first.name}_logit{len(bundles)}",
        limb_ids=first.limb_ids,
        patient_index=first.patient_index,
        labels=first.labels,
        probabilities=probabilities,
        sites=first.sites,
        phases=first.phases,
        vendors=first.vendors,
    )
