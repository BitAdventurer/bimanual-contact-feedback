"""Read a newly integrated held-out grid without consulting packaged test outcomes."""

import re
import numpy as np

from observations import GRID, TEST_MATERIALS

N = 256


def _read(archive, key, shape, dtype=None):
    if key not in archive:
        raise ValueError(f"Native grid is missing {key}")
    x = archive[key]
    if x.shape != shape or (dtype is not None and x.dtype.kind not in dtype):
        raise ValueError(f"Invalid native {key} array: {x.shape} {x.dtype}")
    if x.dtype.kind == "f" and not np.isfinite(x).all():
        raise ValueError(f"Nonfinite native {key}")
    return x


def _endpoint(archive, stage, shape):
    def field(key):
        return _read(archive, f"{stage}_{key}", shape, "f")

    return (
        _read(archive, f"{stage}_full", shape, "b")
        & (field("depth_mm") >= 25.0)
        & (field("clearance_mm") >= 0.10)
        & (field("angle_rad") < 0.10)
        & (_read(archive, f"{stage}_pad_N", shape + (2, 2), "f") > 0.01).all(
            axis=(-2, -1)
        )
    )


def load_native_test(path):
    """Validate and reorder material-major or permuted output into the frozen roster."""
    with np.load(path, allow_pickle=False) as archive:
        family = _read(archive, "family_id", (N,), "U")
        if not np.array_equal(family, _read(archive, "case_id", (N,), "U")):
            raise ValueError("Native family and case IDs disagree")
        by_id = {name: i for i, name in enumerate(family)}
        if len(by_id) != N:
            raise ValueError("Duplicate native family IDs")
        expected = [
            (f"v93r_test_{layout:02d}_{i:03d}", condition, layout, i)
            for layout in (7, 8)
            for i in range(32)
            for condition in TEST_MATERIALS
        ]
        names = [f"{root}_test_{c}" for root, c, _, _ in expected]
        if set(names) != set(by_id):
            raise ValueError("Native families differ from the held-out roster")
        order = np.asarray([by_id[key] for key in names])
        data = {}
        for j, key in enumerate(
            ("paired_root", "condition", "layout", "variant_index")
        ):
            values = _read(archive, key, (N,), "U" if j < 2 else "iu")[order]
            if not np.array_equal(values, [row[j] for row in expected]):
                raise ValueError(f"Native {key} does not match the assigned family")
            data[key] = values
        if not (_read(archive, "split", (N,), "U") == "test").all():
            raise ValueError("Native archive contains a non-test split")
        mass, friction = [
            np.asarray([int(c[1:5]) / 1000 for _, c, _, _ in expected]),
            np.asarray([int(c[7:]) / 100 for _, c, _, _ in expected]),
        ]
        for name, target in (("mass_scale", mass), ("friction_scale", friction)):
            if not np.array_equal(_read(archive, name, (N,), "f")[order], target):
                raise ValueError(f"Native {name} does not match the assigned factor")
        if not np.allclose(
            _read(archive, "material_mass_kg", (N, 2), "f")[order],
            mass[:, None] * [0.085, 0.125],
            atol=1e-12,
            rtol=0,
        ):
            raise ValueError("Measured masses differ from assigned materials")
        contacts = _read(archive, "root_pad_contacts", (N, 2, 2), "iu")[order] > 0
        measured = _read(archive, "root_observed_pad_mu", (N, 2, 2, 2), "f")[order]
        expected_mu = np.broadcast_to(2 * friction[:, None, None, None], measured.shape)
        if not np.allclose(
            measured[contacts], expected_mu[contacts], atol=1e-12, rtol=0
        ):
            raise ValueError("Measured grip friction differs from assigned materials")
        forks = _read(archive, "fork_digests", (N, 5, 3), "U")
        if not (forks == forks[:, :1]).all() or not all(
            re.fullmatch("[0-9a-f]{64}", s) for s in forks[:, 0].flat
        ):
            raise ValueError("Partner motions must share a physical first-action fork")

        for stage, axes in (("root", (N,)), ("mid", (N, 5, 3))):
            for source, key, shape in (
                ("proprio", "proprio", (6, 14)),
                ("p", "b_map", (6, 8, 8)),
                ("partner_p", "a_map", (6, 8, 8)),
            ):
                x = _read(archive, f"{stage}_{source}", axes + shape, "f")[order]
                if stage == "mid" and not (x[:, :, :, :3] == x[:, :1, :, :3]).all():
                    raise ValueError("Sensor histories differ before A moves")
                if key == "proprio":
                    x = np.rint(x / GRID) * GRID
                data[f"{stage}_{key}"] = x
        shape = (N, 5, 3, 3)
        passed = _endpoint(archive, "before", shape) & _endpoint(
            archive, "after", shape
        )
        if not np.array_equal(passed, _read(archive, "success", shape, "b")):
            raise ValueError(
                "Native labels disagree with the two physical endpoint checks"
            )
        data["success"] = passed[order]
        root_validity = _endpoint(archive, "root", (N,))
        if not np.array_equal(root_validity, _read(archive, "root_ok", (N,), "b")):
            raise ValueError("Native initial state diagnostics disagree")
        audit = dict(
            rows=N,
            physical_leaves=passed.size,
            initial_states_inside_guards=int(root_validity.sum()),
            initial_states_outside_guards=int((~root_validity).sum()),
            physical_endpoint_checks=2 * 5 * passed.size,
        )
        return data, audit
