"""Fit and score two-stage bimanual feedback policies using sealed simulation trees."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import json
import os
import tempfile

# Set before loading numerical libraries. A training worker uses one CPU core.
for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np
from native_grid import load_native_test
from observations import DATA, arrays, digest, features, fit_projectors, load_split
from scoring import (
    cohort_metrics,
    decisions,
    ordered_rows,
    paired_effect,
    response_alignment,
    success,
    validation_score,
)

SEEDS = [9201, 9202, 9203]
TRAINING = dict(
    updates=192,
    batch_episodes=384,
    epochs=4,
    minibatch=192,
    learning_rate=0.0005,
    ppo_clip_ratio=0.2,
    value_coefficient=0.5,
    entropy_coefficient=0.01,
    gradient_norm=0.5,
    checkpoints=[48, 96, 144, 192],
)


def write_json(path, item):
    with path.open("x") as stream:
        json.dump(item, stream, indent=2)
        stream.write("\n")


def paths_for(regime, mode, method, step, seeds):
    return [f"{regime}/{mode}/{method}/{seed}/step_{step:03d}.npz" for seed in seeds]


def fit_jobs(inputs, workers, device):
    from agent import fit

    if device == "cuda":
        # One process owns the GPU and runs all requested policies and seeds.
        for item in inputs:
            yield fit(item, device=device)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            yield from pool.map(fit, inputs)


def train(out, workers, controls=False, smoke=False, device="cpu"):
    from agent import resolve_device

    device = resolve_device(device)
    if out.exists():
        raise FileExistsError(f"Choose a new output directory: {out}")
    out.mkdir(parents=True)
    if device == "cuda":
        print("Training on CUDA with one GPU worker.", flush=True)
    else:
        print(f"Training on CPU with up to {workers} worker(s).", flush=True)
    seeds = [SEEDS[0]] if smoke else SEEDS
    cfg = dict(TRAINING)
    if smoke:
        cfg.update(
            updates=2, batch_episodes=48, epochs=1, minibatch=24, checkpoints=[1, 2]
        )
    validation = load_split("validation")
    selected, checks, scores, learning = {}, {}, {}, {}
    plan = [("ppo", "bilateral")]
    extras = [
        ("supervised", "bilateral"),
        ("ppo_committed", "bilateral"),
        ("ppo", "joints"),
        ("ppo", "totals"),
    ]
    first_roots = None
    for regime, split in (("randomized", "train"), ("nominal", "nominal_train")):
        data = load_split(split)
        roots = list(zip(data["paired_root"], data["layout"]))
        if first_roots is not None and roots != first_roots:
            raise ValueError(
                "Training cohorts must have identical nuisance roots and order"
            )
        if set(data["paired_root"]) & set(validation["paired_root"]):
            raise ValueError("Overlapping training and validation roots")
        first_roots = roots
        jobs = plan + (extras if controls and regime == "randomized" else [])
        enc = fit_projectors(data, totals=any(mode == "totals" for _, mode in jobs))
        folder = out / regime
        folder.mkdir()
        enc_name = f"{regime}/encoder.npz"
        np.savez_compressed(out / enc_name, **enc)
        checks[enc_name] = digest(out / enc_name)
        train_x = {
            mode: features(data, enc, mode) for mode in sorted(set(m for _, m in jobs))
        }
        valid_x = {mode: features(validation, enc, mode) for mode in train_x}
        inputs = (
            (method, mode, seed, cfg, *train_x[mode], data["success"], str(folder))
            for method, mode in jobs
            for seed in seeds
        )
        for mode, method, seed, records in fit_jobs(inputs, workers, device):
            name = f"{regime}/{method}/{mode}/{seed}"
            if method.startswith("ppo"):
                t = arrays(folder / mode / method / str(seed) / "onpolicy.npz")[
                    "transitions"
                ]
                i, z, a, b, y = t.transpose(2, 0, 1)
                if t.shape != (
                    cfg["updates"],
                    cfg["batch_episodes"],
                    5,
                ) or not np.array_equal(data["success"][i, z, a, b], y):
                    raise ValueError("Selected terminal reward audit failed")
                learning[name] = {
                    "sampled_terminal_rewards": int(y.size),
                    "first_window_success": float(y[:16].mean()),
                    "last_window_success": float(y[-16:].mean()),
                }
            else:
                learning[name] = {
                    "gradient_updates": len(records)
                    * cfg["epochs"]
                    * (cfg["batch_episodes"] // cfg["minibatch"])
                }
        for method, mode in jobs:
            name = f"{regime}/{method}/{mode}"
            board = {}
            for step in cfg["checkpoints"]:
                paths = paths_for(regime, mode, method, step, seeds)
                nets = [arrays(out / p) for p in paths]
                board[step] = validation_score(
                    validation, decisions(nets, method, *valid_x[mode])
                )
            step = max(cfg["checkpoints"], key=lambda s: (board[s], -s))
            paths = paths_for(regime, mode, method, step, seeds)
            checks.update({p: digest(out / p) for p in paths})
            selected[name] = dict(
                step=step, paths=paths, regime=regime, mode=mode, method=method
            )
            scores[name] = board
            print(
                f"{name}: validation {board[step][0]}/320, selected update {step}",
                flush=True,
            )
    write_json(out / "validation.json", scores)
    write_json(out / "training_summary.json", learning)
    for name in ("training_summary.json", "validation.json"):
        checks[name] = digest(out / name)
    lock = dict(
        smoke=smoke,
        settings=cfg,
        seeds=seeds,
        selected=selected,
        files=checks,
        dataset_manifest=digest(DATA / "manifest.json"),
    )
    if device == "cuda":
        lock["training_device"] = device
    write_json(out / "selected.json", lock)


def evaluate(out, native_test=None):
    lock = json.loads((out / "selected.json").read_text())
    for path, expected in lock["files"].items():
        if digest(out / path) != expected:
            raise ValueError(f"Changed model, calibration or validation record: {path}")
    if digest(DATA / "manifest.json") != lock["dataset_manifest"]:
        raise ValueError("Dataset manifest changed since training")
    if native_test is None:
        data = load_split("test")
        source = {"kind": "packaged", "sha256": digest(DATA / "test.npz")}
    else:
        data, audit = load_native_test(native_test)
        source = dict(kind="native_replay", sha256=digest(native_test), **audit)
    ids = ordered_rows(data)
    metrics, winners = {}, {}
    for name, spec in lock["selected"].items():
        enc = arrays(out / spec["regime"] / "encoder.npz")
        x = features(data, enc, spec["mode"])
        nets = [arrays(out / path) for path in spec["paths"]]
        actions = decisions(nets, spec["method"], *x)
        y = success(data, actions)
        if name == "randomized/ppo/bilateral":
            alignment = response_alignment(data, ids, actions)
        metrics[name] = cohort_metrics(y, ids)
        metrics[name]["interpolated_by_seed"] = [
            cohort_metrics(success(data, decisions([net], spec["method"], *x)), ids)[
                "interpolated"
            ]
            for net in nets
        ]
        winners[name] = y
    metrics["fixed_hold"] = cohort_metrics(data["success"][:, :, 1, 1], ids)
    comparison = paired_effect(
        winners["randomized/ppo/bilateral"],
        winners["nominal/ppo/bilateral"],
        ids,
        data["layout"],
    )
    return dict(
        smoke=lock["smoke"],
        selected_sha256=digest(out / "selected.json"),
        test_source=source,
        policies=metrics,
        randomized_minus_matched_nominal=comparison,
        closed_loop_alignment=alignment,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    smoke = sub.add_parser(
        "smoke", help="Fit and score a two-update PPO in a temporary folder"
    )
    fit = sub.add_parser("fit", help="Fit fresh models without opening test data")
    for command in (smoke, fit):
        command.add_argument(
            "--device",
            choices=("cpu", "cuda", "auto"),
            default="cpu",
            help="Policy training device (default: cpu). Auto uses CUDA if available.",
        )
    fit.add_argument("--out", type=Path, required=True, help="A new run directory")
    fit.add_argument(
        "--workers", type=int, default=3, help="CPU training workers. CUDA uses one."
    )
    fit.add_argument(
        "--controls",
        action="store_true",
        help="Also fit four matched randomized-training controls",
    )
    test = sub.add_parser(
        "evaluate", help="Score the sealed ensemble and compute the paired interval"
    )
    test.add_argument("--run", type=Path, required=True)
    test.add_argument(
        "--native-test",
        type=Path,
        help="Score fresh physics histories and endpoints from replay.py test-grid --output",
    )
    test.add_argument("--json", type=Path, help="Write the report to a new JSON file")
    args = parser.parse_args()
    if args.command == "smoke":
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run"
            train(out, 1, smoke=True, device=args.device)
            report = evaluate(out)
            assert (
                report["randomized_minus_matched_nominal"]["evaluated_futures"] == 960
            )
            print(
                "Two-update optimization, validation selection and 960 paired evaluation outcomes passed."
            )
    elif args.command == "fit":
        if args.workers < 1:
            parser.error("--workers must be positive")
        train(args.out, args.workers, args.controls, device=args.device)
    else:
        report = evaluate(args.run, args.native_test)
        if args.json is not None:
            write_json(args.json, report)
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
