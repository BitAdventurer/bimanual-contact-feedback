"""Headless MuJoCo reproduction of held-out two-Panda insertion-hold trials."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

if __package__:
    from .physics import Panda, simulate_family
    from .task import DATA, TEST_MATERIALS, family_row
else:
    from physics import Panda, simulate_family
    from task import DATA, TEST_MATERIALS, family_row


def criterion(data, prefix):
    return np.stack(
        [
            data[prefix + "full"],
            data[prefix + "depth_mm"] >= 25,
            data[prefix + "clearance_mm"] >= 0.10,
            data[prefix + "angle_rad"] < 0.10,
            data[prefix + "pad_N"].min(axis=(-1, -2)) > 0.01,
        ],
        axis=-1,
    )


@lru_cache(maxsize=1)
def native_model(layout, mass, friction, asset_root):
    return Panda(layout, mass, friction, asset_root)


def simulate(job):
    layout, variant, mass, friction, asset_root = job
    row = family_row(layout, variant, mass, friction)
    model = native_model(layout, mass, friction, asset_root)
    data = simulate_family(model, row)
    if not np.array_equal(
        criterion(data, "before_").all(-1) & criterion(data, "after_").all(-1),
        data["success"],
    ):
        raise RuntimeError("Raw guard measurements disagree with the simulator result.")
    if not np.all(data["fork_digests"] == data["fork_digests"][:1]):
        raise RuntimeError("Partner branches did not fork from the same native state.")
    return data


def test_id(layout, variant, mass, friction):
    return f"v93r_test_{layout:02d}_{variant:03d}_test_m{round(mass * 1000):04d}_f{round(friction * 100):03d}"


def read_reference():
    expected = json.loads((DATA / "provenance.json").read_text())["data_sha256"][
        "reference.npz"
    ]
    path = DATA / "reference.npz"
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise RuntimeError("Stored comparison reference failed its integrity check.")
    with np.load(path, allow_pickle=False) as f:
        return {k: f[k] for k in f.files}


def numerical_comparison(data, reference):
    max_error = {}
    for key in reference:
        if not key.startswith("smoke_") or key == "smoke_id":
            continue
        name = key[6:]
        max_error[name] = float(np.abs(data[name] - reference[key]).max())
    return max_error


def compare(data, reference, idx):
    before = criterion(data, "before_")
    after = criterion(data, "after_")
    return dict(
        identical_labels=int((data["success"] == reference["success"][idx]).sum()),
        identical_before_gates=int((before == reference["before_gates"][idx]).sum()),
        identical_after_gates=int((after == reference["after_gates"][idx]).sum()),
    )


def metadata(layout, variant, mass, friction):
    condition = f"m{round(1000 * mass):04d}_f{round(100 * friction):03d}"
    return dict(
        family_id=test_id(layout, variant, mass, friction),
        case_id=test_id(layout, variant, mass, friction),
        paired_root=f"v93r_test_{layout:02d}_{variant:03d}",
        split="test",
        layout=layout,
        variant_index=variant,
        mass_scale=mass,
        friction_scale=friction,
        condition=condition,
    )


def write_npz(path, data):
    if path is not None:
        with path.open("xb") as f:
            np.savez_compressed(f, **data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("smoke", "family", "test-grid"))
    parser.add_argument("--layout", type=int, default=7)
    parser.add_argument("--variant", type=int, default=27)
    parser.add_argument("--mass", type=float, default=1.125)
    parser.add_argument("--friction", type=float, default=0.8)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Independent native CPU workers for test-grid.",
    )
    parser.add_argument(
        "--asset-root",
        type=Path,
        help="Path to the root of unpacked robosuite/models/assets.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Save raw simulated observations and endpoints as a new NPZ.",
    )
    parser.add_argument(
        "--report", type=Path, help="Save comparison summary as a new JSON file."
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    for path in (args.output, args.report):
        if path is not None and path.exists():
            parser.error(f"Output already exists: {path}")
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
    if (
        args.output is not None
        and args.report is not None
        and args.output.resolve() == args.report.resolve()
    ):
        parser.error("--output and --report must be different files.")
    if args.command == "family":
        job = (args.layout, args.variant, args.mass, args.friction, args.asset_root)
        data = simulate(job)
        report = dict(
            layout=args.layout,
            variant=args.variant,
            mass=args.mass,
            friction=args.friction,
            branches=45,
            successful_branches=int(data["success"].sum()),
            passed=True,
            reference_comparison=False,
            order=["partner future", "first action", "second action"],
        )
        write_npz(args.output, data)
    else:
        single = args.command == "smoke"
        jobs = (
            [(7, 27, 1.125, 0.8, args.asset_root)]
            if single
            else [
                (layout, variant, mass, friction, args.asset_root)
                for layout in (7, 8)
                for mass, friction in TEST_MATERIALS
                for variant in range(32)
            ]
        )
        pool = (
            ProcessPoolExecutor(max_workers=args.workers)
            if args.workers > 1 and not single
            else None
        )
        rows = pool.map(simulate, jobs) if pool else map(simulate, jobs)
        reference = None
        totals = dict(
            identical_labels=0, identical_before_gates=0, identical_after_gates=0
        )
        joined = {}
        differences = []
        labels = []
        smoke_error = {}
        try:
            for job, data in zip(jobs, rows):
                if reference is None:
                    reference = read_reference()
                    ids = {str(v): i for i, v in enumerate(reference["family_id"])}
                meta = metadata(*job[:4])
                idx = ids[meta["family_id"]]
                counts = compare(data, reference, idx)
                for k, v in counts.items():
                    totals[k] += v
                if (
                    counts["identical_labels"] != 45
                    or counts["identical_before_gates"] != 225
                    or counts["identical_after_gates"] != 225
                ):
                    differences.append(meta["family_id"])
                if meta["family_id"] == str(reference["smoke_id"]):
                    smoke_error = numerical_comparison(data, reference)
                labels.append((idx, data["success"].copy()))
                if args.output:
                    for k, v in dict(data, **meta).items():
                        joined.setdefault(k, []).append(v)
        finally:
            if pool:
                pool.shutdown()
        ordered = np.stack([a for _, a in sorted(labels, key=lambda p: p[0])])
        original = reference["success"] if not single else reference["success"][[idx]]

        def digest(value):
            return hashlib.sha256(
                np.ascontiguousarray(value, dtype=np.uint8).tobytes()
            ).hexdigest()

        report = dict(
            families=len(jobs),
            physical_branches=len(jobs) * 45,
            checks_each_endpoint=len(jobs) * 225,
            **totals,
            simulated_label_sha256=digest(ordered),
            reference_label_sha256=digest(original),
            discrepant_families=differences,
            canonical_max_abs=smoke_error,
        )
        report["passed"] = not differences and max(smoke_error.values()) < 1e-6
        write_npz(args.output, {k: np.stack(v) for k, v in joined.items()})
    message = json.dumps(report, indent=2)
    if args.report:
        with args.report.open("x") as f:
            f.write(message + "\n")
    print(message)
    if not report.get("passed", True):
        sys.exit(1)


if __name__ == "__main__":
    main()
