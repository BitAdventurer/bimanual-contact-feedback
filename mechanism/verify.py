"""Re-infer socket-hand actions from recorded intact / cut sensor packets with NumPy."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


MANIFEST_SHA256 = "d7e104963c8d2d3d279917935f6081ed5bf80745be87b54f952316c7fb9c2da7"
GRID = np.repeat([0.001, 0.01], 7)
MATERIALS = {"m1125_f080", "m1500_f060"}
FUTURES = (0, 2, 4)  # left large, no partner motion, right large
TIE_ORDER = np.array([1, 0, 2])  # hold, left, right


def arrays(path):
    with np.load(path, allow_pickle=False) as values:
        return {key: values[key] for key in values.files}


def encode(enc, proprio, pad_map):
    """Map six observations to the two visible slots of a 64-value policy input."""
    if proprio.shape[-2:] != (6, 14) or pad_map.shape[-3:] != (6, 8, 8):
        raise ValueError("Expected six joint and six 8-by-8 socket-pad observations")
    shape = proprio.shape[:-2]
    if pad_map.shape[:-3] != shape:
        raise ValueError("Unmatched packet dimensions")
    maps = pad_map.reshape(-1, 384)
    joint = (np.rint(proprio / GRID) * GRID).reshape(-1, 84)
    encoded = np.zeros((len(maps), 64), dtype=np.float32)
    for key, values, start in [("proprio", joint, 0), ("p", maps, 32)]:
        z = np.clip((values - enc[key + "_mean"]) / enc[key + "_scale"], -8, 8)
        projection = (z - enc[key + "_shift"]) @ enc[key + "_components"]
        encoded[:, start : start + 16] = np.clip(
            projection / enc[key + "_pc_std"], -8, 8
        )
    return encoded.reshape(shape + (64,))


def predict(networks, x, stage):
    """Average three frozen categorical PPO distributions and pick their greedy action."""
    scores = []
    for model in networks:
        h = np.tanh(x @ model[stage + "W0"].T + model[stage + "b0"])
        h = np.tanh(h @ model[stage + "W1"].T + model[stage + "b1"])
        logits = h @ model[stage + "W2"].T + model[stage + "b2"]
        exp = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        scores.append(exp / exp.sum(axis=-1, keepdims=True))
    probability = np.mean(scores, axis=0)
    return TIE_ORDER[np.argmax(probability[..., TIE_ORDER], axis=-1)], probability


def infer(case, encoders, networks):
    root = encode(encoders[0], case["root_proprio"], case["root_p"])
    own_first = int(predict(networks, root, "r")[0])
    packets = {}
    for key in ("proprio", "p"):
        # [world, first B action, partner motion, time, ...]. The pre-motion
        # packet is shared between all worlds and all subsequent partner moves.
        post = case[key]
        pre = np.broadcast_to(case["pre_" + key][None, :, None], post.shape)
        packets[key] = np.concatenate((pre, post), axis=3)
    mid = encode(encoders[1], packets["proprio"], packets["p"])
    one_hot = np.broadcast_to(
        np.eye(3, dtype=np.float32)[None, :, None, :], (2, 3, 3, 3)
    )
    inputs = np.concatenate([np.broadcast_to(root, mid.shape), mid, one_hot], axis=-1)
    followup, probabilities = predict(networks, inputs, "m")
    return own_first, followup, probabilities


def references(path):
    result = {}
    with path.open(newline="") as source:
        for row in csv.DictReader(source):
            key = (
                row["family_id"],
                int(row["world_0_intact_1_cut"]),
                int(row["first"]),
                int(row["future"]),
            )
            if key in result or row["policy"] != "local_b":
                raise ValueError("Duplicate or wrong policy in the reference table")
            result[key] = (
                row["policy_first"] == "True",
                int(row["next"]),
                np.array([float(row["p_" + a]) for a in ("left", "hold", "right")]),
            )
    return result


def verify(folder):
    metadata = (folder / "manifest.json").read_bytes()
    if hashlib.sha256(metadata).hexdigest() != MANIFEST_SHA256:
        raise ValueError("Unexpected evidence manifest")
    manifest = json.loads(metadata)
    for name, expected in manifest["files"].items():
        if hashlib.sha256((folder / name).read_bytes()).hexdigest() != expected:
            raise ValueError("Checksum mismatch for " + name)
    encoders = [
        arrays(folder / "models" / (stage + "_encoder.npz"))
        for stage in ("root", "mid")
    ]
    networks = [
        arrays(folder / "models" / f"seed_{seed}.npz") for seed in manifest["seeds"]
    ]
    ref = references(folder / "reference.csv")
    decisions, chosen, materials = [], [], []
    loaded, grip, q_diff, tactile_diff, force_diff, parts_diff = [], [], [], [], [], []
    roots = {}
    max_probability_error = 0.0
    count = 0
    for name in sorted(manifest["files"]):
        if not name.startswith("cases/"):
            continue
        case = arrays(folder / name)
        family = str(case["family_id"])
        if family + ".npz" != Path(name).name:
            raise ValueError("Case identifier does not match file")
        group = (int(case["layout"]), str(case["paired_root"]))
        material = str(case["condition"])
        if material in roots.setdefault(group, set()):
            raise ValueError("Duplicated root-material combination")
        roots[group].add(material)
        own_first, actions, probabilities = infer(case, encoders, networks)
        for w, a, z in np.ndindex(2, 3, 3):
            own_ref, action_ref, prob_ref = ref.pop((family, w, a, FUTURES[z]))
            if own_ref != (a == own_first) or action_ref != actions[w, a, z]:
                raise ValueError("Archival action mismatch at " + family)
            err = np.max(np.abs(probabilities[w, a, z] - prob_ref))
            max_probability_error = max(max_probability_error, float(err))
            count += 1
        decisions.append(actions)
        chosen.append(own_first)
        materials.append(material)
        trace = case["force_trace"]
        if trace.shape != (2, 3, 3, 640, 4):
            raise ValueError("Unexpected physics trace dimensions")
        loaded.append(np.any(trace[..., 1] > 0, axis=-1))
        grip.append(case["pad_min_N"].min(axis=(1, 2)))
        dx = case["parts_mm"][:, :, 2, :, :, 0] - case["parts_mm"][:, :, 0, :, :, 0]
        parts_diff.append(np.abs(dx).max(axis=2))
        q = np.rint(case["proprio"] / GRID)
        q_diff.append((q[:, :, 0] != q[:, :, 2]).any(axis=(-2, -1)))
        dp = case["p"][:, :, 2] - case["p"][:, :, 0]
        tactile_diff.append(np.abs(dp).sum(axis=(-2, -1)).max(axis=-1))
        dw = case["w"][:, :, 2, :, :3] - case["w"][:, :, 0, :, :3]
        force_diff.append(np.linalg.norm(dw, axis=-1).max(axis=-1))

    if ref or count != 576 or max_probability_error > 1.0e-5:
        raise ValueError(
            f"Incomplete or unequal policy replay: {count} rows, error {max_probability_error}"
        )
    if any(v != MATERIALS for v in roots.values()):
        raise ValueError("Unbalanced material roster")
    layouts = [layout for layout, root in roots]
    if sorted(set(layouts)) != [7, 8] or layouts.count(7) != 8 or layouts.count(8) != 8:
        raise ValueError("Unbalanced independent root roster")

    actions = np.stack(decisions)  # [physical case, world, forced first action, future]
    changes = actions[..., 0] != actions[..., 2]
    total = np.sum(changes, axis=(0, 2)).tolist()
    own = changes[np.arange(len(chosen)), :, chosen].sum(axis=0).tolist()
    if total != [85, 0] or own != [31, 0]:
        raise ValueError(f"Unexpected directional response counts: {total}, {own}")
    force = np.stack(force_diff)
    pad = np.stack(tactile_diff)
    loads = np.stack(loaded).sum(axis=(0, 2, 3)).tolist()
    pad_minimum = np.stack(grip).min(axis=0)
    if loads != [156, 0] or pad_minimum.min() <= 16:
        raise ValueError("Contact or grasp load differs from the reported experiment")
    by_material = {
        key: changes[np.array(materials) == key].sum(axis=(0, 2)).tolist()
        for key in sorted(MATERIALS)
    }
    return {
        "world_order": ["intact", "cut"],
        "independent_roots": len(roots),
        "materials_per_root": 2,
        "forced_first_forks": int(changes[:, 0].size),
        "own_first_forks": len(chosen),
        "opposite_motion_changes_all_first_actions": total,
        "opposite_motion_changes_own_first": own,
        "opposite_motion_changes_per_material_48_forks_each": by_material,
        "reference_policy_rows_reproduced": count,
        "max_abs_action_probability_error": max_probability_error,
        "physical_intervals_per_world": int(np.stack(loaded)[:, 0].size),
        "intervals_with_positive_mating_force": loads,
        "minimum_load_each_pad_N": pad_minimum.tolist(),
        "median_peak_peg_x_contrast_mm": np.median(
            np.stack(parts_diff)[..., 0], axis=(0, 2)
        ).tolist(),
        "median_peak_socket_x_contrast_mm": np.median(
            np.stack(parts_diff)[..., 1], axis=(0, 2)
        ).tolist(),
        "median_peak_socket_hand_force_contrast_N": np.median(
            force, axis=(0, 2)
        ).tolist(),
        "max_peak_socket_hand_force_contrast_N": force.max(axis=(0, 2)).tolist(),
        "median_peak_socket_pad_map_L1_contrast_N": np.median(
            pad, axis=(0, 2)
        ).tolist(),
        "forks_with_distinct_quantized_socket_joints": np.stack(q_diff)
        .sum(axis=(0, 2))
        .tolist(),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, default=Path(__file__).resolve().parent / "data"
    )
    print(json.dumps(verify(parser.parse_args().data), indent=2))
