"""Resistance along a venous path, and the dimensionless ratio built from it.

For a sequence of perpendicular sections the series resistance of a span is the
sum of the per-sample hydraulic resistances,

    R = sum over s_k in the span of ds_k / G(s_k),

where a sample inside a non-opacified span contributes no finite resistance and
is reported separately as occlusion length instead. The dimensionless resistance
ratio

    rho_R = R / R_ref

compares that resistance with the resistance of a paired lesion-free span of the
same length, which is what makes it comparable across limbs and across sites.

Ref: Sec. 6.5 (descriptor list and Eq. (1) context), Algorithm 1 lines 9-12,
Sec. 3.3 (transport of the dimensionless families).
"""

from __future__ import annotations

import math

import numpy as np

from conductance_ctv.utils.types import BoolArray, FloatArray

_RESOLUTION = 1e-9


class SpanError(ValueError):
    """Raised when a span cannot carry a resistance by construction."""


def patent_mask(conductance: FloatArray, floor: float = _RESOLUTION) -> BoolArray:
    """Samples that carry flow; a closed span has ``G = 0`` by definition."""
    return np.asarray(conductance, dtype=np.float64) > floor


def occlusion_length(conductance: FloatArray, ds: float) -> float:
    """Total length of the non-opacified part of a span, in mm."""
    closed = ~patent_mask(conductance)
    return float(np.count_nonzero(closed)) * float(ds)


def series_resistance(conductance: FloatArray, ds: float) -> float:
    """``R = sum ds / G`` over the patent samples of a span, in mm^-3.

    A span with no patent sample has infinite resistance; the caller is expected
    to record the occlusion length instead of a resistance, matching Algorithm 1
    line 10.
    """
    values = np.asarray(conductance, dtype=np.float64)
    if values.size == 0:
        raise SpanError("cannot take the resistance of an empty span")
    if ds <= 0.0:
        raise SpanError("the arc-length step must be strictly positive")
    open_values = values[patent_mask(values)]
    if open_values.size == 0:
        return math.inf
    return float(np.sum(float(ds) / open_values))


def mean_conductance(conductance: FloatArray) -> float:
    values = np.asarray(conductance, dtype=np.float64)
    open_values = values[patent_mask(values)]
    if open_values.size == 0:
        return 0.0
    return float(np.mean(open_values))


def reference_span(
    candidates: dict[str, FloatArray],
    span_length: float,
    ds: float,
    mode: str = "max",
    order: tuple[str, ...] = (),
) -> str:
    """Pick the lesion-free span a resistance ratio is formed against.

    ``mode`` is one of ``max`` (the candidate with the highest mean conductance),
    ``upstream`` (the first candidate preceding the lesion in path order) or
    ``explicit`` (the first candidate of ``order``). The manuscript states that
    the reference is taken from a lesion-free part of the opposite or the
    upstream segment without fixing which; ``max`` is the default and is exposed
    as ``hydraulics.reference_mode``.

    Ref: Sec. 6.5 ("detected between a lesion-free area of the opposite or
    upstream segment").
    """
    if not candidates:
        raise SpanError("no lesion-free span is available as a reference")
    if span_length <= 0.0:
        raise SpanError("the lesion span must have positive length")
    usable = {name: values for name, values in candidates.items() if mean_conductance(values) > 0.0}
    if not usable:
        raise SpanError("every candidate reference span is non-opacified")
    if mode == "max":
        return max(usable, key=lambda name: mean_conductance(usable[name]))
    if mode == "upstream":
        for name in order:
            if name in usable:
                return name
        raise SpanError("no upstream candidate reference span is patent")
    if mode == "explicit":
        if not order:
            raise SpanError("explicit reference selection needs a span order")
        if order[0] not in usable:
            raise SpanError(f"the requested reference span {order[0]!r} is not patent")
        return order[0]
    raise SpanError(f"unknown reference-span mode {mode!r}")


def reference_resistance(conductance: FloatArray, span_length: float, ds: float) -> float:
    """Resistance of a lesion-free span of the same length as the lesion span.

    Forming ``R_ref`` over the matching length, rather than over whatever length
    the reference happens to have, is what leaves ``rho_R`` dimensionless and
    length-normalised.
    """
    values = np.asarray(conductance, dtype=np.float64)
    mean = mean_conductance(values)
    if mean <= 0.0:
        raise SpanError("the reference span carries no flow")
    samples = max(1.0, round(float(span_length) / float(ds)))
    return float(samples * ds / mean)


def resistance_ratio(resistance: float, reference: float) -> float:
    """``rho_R = R / R_ref``; infinite when the reference carries no flow."""
    if not math.isfinite(resistance):
        return math.inf
    if reference <= 0.0 or not math.isfinite(reference):
        return math.inf
    return float(resistance) / float(reference)


def conductance_weighted_length(conductance: FloatArray, ds: float, reference: float) -> float:
    """Length of healthy vessel carrying the same hydraulic burden as the span.

    The manuscript lists a conductance-weighted length without fixing its
    normalisation. The form used here is the equivalent healthy length

        L_w = L * (R_ref / R),   R_ref = L / G_ref,

    which returns the span's own patent length when every sample is as
    conductive as the lesion-free reference and shrinks as the lumen narrows, so
    it is a length that carries the burden the resistance already describes.
    A weighting ``sum_k (G(s_k) / G_ref) ds`` would instead return the span
    length exactly, because the weights sum to the sample count times their
    mean; that form is degenerate and is not used.

    Ref: Sec. 6.5 (descriptor list); the normalisation is recorded as a release
    default in the claim register.
    """
    values = np.asarray(conductance, dtype=np.float64)
    if values.size == 0:
        return 0.0
    if reference <= 0.0 or ds <= 0.0:
        return 0.0
    open_values = values[patent_mask(values)]
    if open_values.size == 0:
        return 0.0
    patent_length = float(open_values.size) * float(ds)
    resistance = float(np.sum(float(ds) / open_values))
    if resistance <= 0.0 or not math.isfinite(resistance):
        return 0.0
    reference_resistance_value = patent_length / float(reference)
    return float(patent_length * reference_resistance_value / resistance)


def minimal_conductance_index(conductance: FloatArray, patent: BoolArray | None = None) -> int:
    """``s* = argmin G``; ties resolve to the earliest sample.

    Ref: Algorithm 1 line 8.
    """
    values = np.asarray(conductance, dtype=np.float64)
    if values.size == 0:
        raise SpanError("cannot locate a minimum in an empty span")
    candidates = values
    if patent is not None:
        flags = np.asarray(patent, dtype=bool)
        if bool(flags.any()):
            candidates = np.where(flags, values, np.inf)
    finite = np.isfinite(candidates)
    if not bool(finite.any()):
        return int(np.argmin(values))
    return int(np.argmin(np.where(finite, candidates, np.inf)))


def minimal_area_index(area: FloatArray) -> int:
    """``argmin A``: the arclength of the smallest area, kept as a diagnostic.

    Algorithm 1 selects ``s*`` on the conductance, not on the area, so this is
    reported alongside rather than used as the descriptor location.
    """
    values = np.asarray(area, dtype=np.float64)
    if values.size == 0:
        raise SpanError("cannot locate a minimum in an empty span")
    return int(np.argmin(values))


def propagated_resistance_error(area: float, area_error: float) -> float:
    """Relative error that an area error carries into the resistance.

    ``R`` scales as ``A^-2``, so a relative area error ``e`` becomes ``2 e`` in
    the resistance. The manuscript quotes 4.1% at the common iliac vein and 8.2%
    in ``rho_R``, which is this factor of two.

    Ref: Sec. 3.2 and Sec. 4.1; Table 5 propagated-error column.
    """
    if area <= 0.0:
        return math.inf
    return 2.0 * float(area_error)


def resistance_sensitivity(area: float, area_error: float) -> float:
    """Absolute change in the inverse-area resistance term for a given area error."""
    if area <= 0.0:
        return math.inf
    base = 1.0 / (area * area)
    perturbed = 1.0 / ((area * (1.0 + float(area_error))) ** 2)
    return abs(perturbed - base) / base
