"""Causal physical observations and outcome-blind, training-only sensor calibration."""

from pathlib import Path
import hashlib
import json

import numpy as np

DATA = Path(__file__).resolve().parent / "data"
GRID = np.repeat([0.001, 0.01], 7)
BLOCKS = ("proprio", "b_map", "a_map")
SLOTS = {"proprio": 0, "b_map": 32, "a_map": 48}
WIDTH = {"proprio": 84, "b_map": 384, "a_map": 384}
TEST_MATERIALS = ("m1125_f080", "m1375_f080", "m1625_f080", "m1500_f060")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def load_split(name):
    receipt = json.loads((DATA / "manifest.json").read_text())[name]
    path = DATA / receipt["file"]
    if digest(path) != receipt["sha256"]:
        raise ValueError(f"Checksum mismatch for {name}")
    data = arrays(path)
    n = receipt["rows"]
    if data["success"].shape != (n, 5, 3, 3) or data["success"].dtype != np.bool_:
        raise ValueError("Expected a binary terminal outcome table")
    if len(set(zip(data["paired_root"], data["condition"]))) != n:
        raise ValueError("Duplicate physical condition for a matched root")
    for stage, prefix in (("root", (n,)), ("mid", (n, 5, 3))):
        bins = data.pop(stage + "_joint_bins")
        if bins.shape != prefix + (6, 14) or bins.dtype != np.int16:
            raise ValueError("Invalid quantized joint packets")
        data[stage + "_proprio"] = bins * GRID
        for arm in ("a", "b"):
            x = data[stage + "_" + arm + "_map"]
            if x.shape != prefix + (6, 8, 8) or not np.isfinite(x).all():
                raise ValueError("Invalid physical contact map packets")
    return data


def matrix(data, stage, key, selection=None):
    channel = key[:-6] if key.endswith("_total") else key
    x = data[stage + "_" + channel]
    if selection is not None:
        x = x[selection]
    if key.endswith("_total"):
        x = x.sum(axis=(-2, -1))
        width = 6
    else:
        width = WIDTH[channel]
    return np.asarray(x.reshape(-1, width), dtype=np.float64)


def fit_projector(x):
    from scipy.linalg import eigh

    mean, std = x.mean(0), x.std(0)
    scale = np.maximum(std, max(1e-8, 1e-3 * np.sqrt(np.mean(std**2))))
    centered = np.clip((x - mean) / scale, -8, 8)
    shift = centered.mean(0)
    centered -= shift
    covariance = centered.T @ centered / max(1, len(centered) - 1)
    cols = x.shape[1]
    eig, vec = eigh(
        covariance, subset_by_index=(max(0, cols - 16), cols - 1), check_finite=False
    )
    order = np.arange(len(eig) - 1, -1, -1)
    components = vec[:, order]
    signs = np.sign(
        components[np.argmax(np.abs(components), axis=0), np.arange(len(order))]
    )
    components = components * np.where(signs == 0, 1, signs)
    pc_std = np.maximum(np.sqrt(np.maximum(0, eig[order])), 1e-4)
    return dict(
        mean=mean, scale=scale, shift=shift, components=components, pc_std=pc_std
    )


def fit_projectors(data, totals=False):
    n = len(data["layout"])
    roots = np.repeat(np.arange(n), 4)
    random = np.random.default_rng(921703)
    futures = random.integers(5, size=len(roots))
    actions = random.integers(3, size=len(roots))
    keys = BLOCKS + (("b_map_total", "a_map_total") if totals else ())
    enc = {}
    for stage, selection in (
        ("root", np.arange(n)),
        ("mid", (roots, futures, actions)),
    ):
        for key in keys:
            fit = fit_projector(matrix(data, stage, key, selection))
            for name, value in fit.items():
                enc[f"{stage}_{key}_{name}"] = value
    return enc


def features(data, enc, mode):
    stages = []
    for stage in ("root", "mid"):
        prefix = data[stage + "_proprio"].shape[:-2]
        x = np.zeros((int(np.prod(prefix)), 64), dtype=np.float32)
        for channel in BLOCKS if mode != "joints" else ("proprio",):
            key = (
                channel + "_total"
                if mode == "totals" and channel != "proprio"
                else channel
            )
            params = {
                k: enc[f"{stage}_{key}_{k}"]
                for k in ("mean", "scale", "shift", "components", "pc_std")
            }
            raw = matrix(data, stage, key)
            z = (
                np.clip((raw - params["mean"]) / params["scale"], -8, 8)
                - params["shift"]
            )
            projection = np.clip((z @ params["components"]) / params["pc_std"], -8, 8)
            off = SLOTS[channel]
            x[:, off : off + projection.shape[-1]] = projection
        stages.append(x.reshape(prefix + (64,)))
    r, m = stages
    a = np.broadcast_to(np.eye(3, dtype=np.float32)[None, None], (len(r), 5, 3, 3))
    return r, np.concatenate(
        (np.broadcast_to(r[:, None, None, :], m.shape), m, a), axis=-1
    )
