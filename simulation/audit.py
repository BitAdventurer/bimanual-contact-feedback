"""Canonical native-float comparisons. Discrete task-success comparisons are separate."""

import numpy as np


def canonical_channels():
    """Yield (array, component, last axis slice, unit, atol, rtol)."""
    for stage in ("root", "mid"):
        yield stage + "_proprio", "angle", slice(0, 7), "rad", 1e-7, 0.0
        yield stage + "_proprio", "velocity", slice(7, 14), "rad/s", 1e-7, 0.0
        yield stage + "_w", "force", slice(0, 3), "N", 1e-7, 0.0
        yield stage + "_w", "torque", slice(3, 6), "N m", 1e-7, 0.0
        for sensor in ("p", "partner_p"):
            yield stage + "_" + sensor, None, slice(None), "N/cell", 1e-7, 0.0
    for point in ("before", "after"):
        for k in ("clearance_mm", "depth_mm"):
            yield point + "_" + k, None, slice(None), "mm", 1e-6, 0.0
        yield point + "_angle_rad", None, slice(None), "rad", 1e-7, 0.0
        yield point + "_pad_N", None, slice(None), "N", 1e-6, 1e-6


def compare_canonical(data, reference):
    """Check physical errors elementwise against the archived value of that element."""
    schema = list(canonical_channels())
    expected = {name for name, *_ in schema}
    available = {k[6:] for k in reference if k.startswith("smoke_") and k != "smoke_id"}
    if expected != available:
        raise ValueError(
            f"Canonical reference schema mismatch: {sorted(expected ^ available)}"
        )
    per_field = {}
    channels = {}
    for name, component, axis, unit, atol, rtol in schema:
        actual = np.asarray(data[name])
        archived = np.asarray(reference["smoke_" + name])
        if name not in per_field:
            if actual.shape != archived.shape or actual.size == 0:
                raise ValueError(
                    f"Invalid canonical shape for {name}: {actual.shape} vs {archived.shape}"
                )
            if not (np.isfinite(actual).all() and np.isfinite(archived).all()):
                raise ValueError(f"Nonfinite canonical measurement in {name}")
            per_field[name] = dict(
                max_abs=float(np.abs(actual - archived).max()),
                bit_exact=actual.dtype == archived.dtype
                and actual.tobytes() == archived.tobytes(),
            )
        measured = actual[..., axis]
        target = archived[..., axis]
        error = np.abs(measured - target)
        limit = atol + rtol * np.abs(target)
        ratio = error / limit
        key = name if component is None else name + "." + component
        channels[key] = dict(
            unit=unit,
            absolute_limit=atol,
            relative_limit=rtol,
            elements=int(error.size),
            exceedances=int((error > limit).sum()),
            max_abs=float(error.max()),
            max_error_over_limit=float(ratio.max()),
        )
    original_max = max(f["max_abs"] for f in per_field.values())
    return dict(
        fields=per_field,
        channels=channels,
        bit_exact=all(f["bit_exact"] for f in per_field.values()),
        within_tolerance=all(c["exceedances"] == 0 for c in channels.values()),
        legacy_untyped_max_abs=original_max,
        legacy_max_abs_lt_1e_6=original_max < 1e-6,
    )


def verification_passed(discrepant_families, canonical, strict=False):
    return not discrepant_families and (
        canonical["bit_exact"] if strict else canonical["within_tolerance"]
    )
