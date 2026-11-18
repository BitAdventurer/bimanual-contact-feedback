"""Re-simulate both sides of the randomized contact-cut comparison, then infer socket decisions."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import time

import mujoco as mj
import numpy as np

from simulation.physics import INTEGRATION, Panda, SENSORS, commands, prepare
from simulation.task import family_row
from . import verify


PACKET_DIR = Path(__file__).resolve().parent / "data"
PHYSICS_ATOL = 2e-6  # N, mm, rad or rad/s according to channel, absolute tolerance


def roster():
    """Select 8 of 32 test draws per layout by stable hash, prior to any simulation."""
    rows = []
    for layout in (7, 8):

        def priority(variant):
            key = f"v93r-contact-cut-mechanism/v93r_test_{layout:02d}_{variant:03d}"
            return hashlib.sha256(key.encode()).digest()

        for variant in sorted(range(32), key=priority)[:8]:
            for mass, friction in ((1.125, 0.8), (1.5, 0.6)):
                cid = f"v93r_test_{layout:02d}_{variant:03d}_test_m{round(1000 * mass):04d}_f{round(100 * friction):03d}"
                rows.append((cid, layout, variant, mass, friction))
    return sorted(rows)


@lru_cache(maxsize=1)
def scene_pair(layout, mass, friction, asset_root):
    scenes = [
        Panda(layout, mass, friction, asset_root, mating_contact=enabled)
        for enabled in (True, False)
    ]
    on, off = (scene.m for scene in scenes)
    if (
        on.npair != 22
        or off.npair != 4
        or (on.nq, on.nv, on.nu) != (off.nq, off.nv, off.nu)
    ):
        raise ValueError("Unexpected intervention geometry or dynamics")
    for key in (
        "body_mass",
        "body_inertia",
        "actuator_gainprm",
        "geom_friction",
        "eq_obj1id",
        "eq_obj2id",
    ):
        np.testing.assert_array_equal(getattr(on, key), getattr(off, key))
    for i in range(off.npair):
        j = on.pair(off.pair(i).name).id
        for key in (
            "pair_geom1",
            "pair_geom2",
            "pair_friction",
            "pair_solref",
            "pair_solimp",
        ):
            np.testing.assert_array_equal(getattr(off, key)[i], getattr(on, key)[j])
    return scenes


def pose_history(packets):
    return {key: np.stack([sample[key] for sample in packets]) for key in SENSORS}


def partner_interval(scene, first, target):
    """Sample contacts before each observation's forward pass, as in the native acquisition."""
    m, d = scene.m, scene.d
    steps = round(0.32 / m.opt.timestep)
    ramp = np.minimum(1.0, np.arange(1, steps + 1) * m.opt.timestep / 0.20)
    ramp = ramp * ramp * (3 - 2 * ramp)
    command = first[None, :] * (1 - ramp[:, None]) + target[None, :] * ramp[:, None]
    desired = scene.origin[None] + command[:, :, None] * scene.offset[None] / 1000.0
    peg, socket = scene.body
    pad_slots = {
        (a, b): 2 * arm + finger for (a, b), (arm, finger) in scene.pads.items()
    }
    pad_slots.update({(b, a): slot for (a, b), slot in list(pad_slots.items())})
    frames = {round(i * steps / 3) for i in (1, 2, 3)}
    packets, part_positions = [], []
    forces = np.empty(6)
    trace = np.zeros((steps, 4))
    pad_minimum = np.full(4, np.inf)
    for t in range(steps):
        torque = (
            scene.eq_torque
            + scene.gains * (desired[t] - d.qpos[scene.q])
            - scene.damping * d.qvel[scene.v]
        )
        d.ctrl[scene.act] = np.clip(torque, scene.limits[..., 0], scene.limits[..., 1])
        mj.mj_step(m, d)
        net = np.zeros(3)
        pad_load = np.zeros(4)
        for i, contact in enumerate(d.contact[: d.ncon]):
            g1, g2 = int(contact.geom1), int(contact.geom2)
            b1, b2 = int(m.geom_bodyid[g1]), int(m.geom_bodyid[g2])
            is_mating = (b1 == peg and b2 == socket) or (b2 == peg and b1 == socket)
            slot = pad_slots.get((g1, g2))
            if not is_mating and slot is None:
                continue
            mj.mj_contactForce(m, d, i, forces)
            if is_mating:
                trace[t, 0] += 1
                trace[t, 1] += forces[0] > 1e-8
                trace[t, 2] += max(0.0, forces[0])
                net += (1 if b2 == socket else -1) * (
                    contact.frame.reshape(3, 3).T @ forces[:3]
                )
            if slot is not None:
                pad_load[slot] += max(0.0, forces[0])
        trace[t, 3] = np.linalg.norm(net)
        pad_minimum = np.minimum(pad_minimum, pad_load)
        if t + 1 in frames:
            packets.append(scene.packet())
            part_positions.append(d.xpos[scene.body].copy() * 1000.0)
    if any(w.number for w in d.warning):
        raise RuntimeError("MuJoCo warning during a cut-comparison interval")
    return dict(
        **pose_history(packets),
        parts_mm=np.stack(part_positions),
        force_trace=trace,
        pad_min_N=pad_minimum,
    )


def simulate(row, layout, mass, friction, asset_root):
    native, cut = scene_pair(layout, mass, friction, asset_root)
    native.restore(native.initial)
    offset, packets = prepare(native, row)
    tape = {"root_" + k: v for k, v in pose_history(packets).items()}
    root = native.snapshot()
    for a in range(3):
        native.restore(root)
        first, _, _ = commands(row, offset, a, 2, 1)
        pre, _ = native.phase_obs(offset, first, 0.22, 0.14, 3)
        fork = native.snapshot()
        for key, packet in pose_history(pre).items():
            if a == 0:
                tape["pre_" + key] = np.empty((3,) + packet.shape)
            tape["pre_" + key][a] = packet
        for world, scene in enumerate((native, cut)):
            for j, future in enumerate(verify.FUTURES):
                # Restore before integration. A forward call here would recompute the solver warm-start.
                mj.mj_setState(scene.m, scene.d, fork, INTEGRATION)
                np.testing.assert_array_equal(scene.snapshot(), fork)
                _, target, _ = commands(row, offset, a, future, 1)
                interval = partner_interval(scene, first, target)
                for key, values in interval.items():
                    if key not in tape:
                        tape[key] = np.empty((2, 3, 3) + values.shape)
                    tape[key][world, a, j] = values
    return tape


@lru_cache(maxsize=1)
def policy():
    encoders = [
        verify.arrays(PACKET_DIR / "models" / (s + "_encoder.npz"))
        for s in ("root", "mid")
    ]
    models = [
        verify.arrays(PACKET_DIR / "models" / f"seed_{s}.npz")
        for s in (9201, 9202, 9203)
    ]
    return encoders, models


def worker(job):
    record, asset_root = job
    family, layout, variant, mass, friction = record
    tape = simulate(
        family_row(layout, variant, mass, friction), layout, mass, friction, asset_root
    )
    own, actions, probs = verify.infer(tape, *policy())
    # Open archived observations only after physically generating both worlds and choosing all actions.
    original = verify.arrays(PACKET_DIR / "cases" / (family + ".npz"))
    error = {k: float(np.max(np.abs(x - original[k]))) for k, x in tape.items()}
    for k, worst in error.items():
        if not np.isfinite(worst) or worst > PHYSICS_ATOL:
            raise ValueError(
                f"{family}: {k} max absolute native/archive difference {worst:g} exceeds {PHYSICS_ATOL:g}"
            )
    changes = actions[:, :, 0] != actions[:, :, 2]
    trace = tape["force_trace"]
    active = np.any(trace[..., 1] > 0, axis=-1)
    force = tape["w"][:, :, 2, :, :3] - tape["w"][:, :, 0, :, :3]
    pad = tape["p"][:, :, 2] - tape["p"][:, :, 0]
    return dict(
        family=family,
        own=own,
        actions=actions,
        probs=probs,
        changes=changes,
        active=active,
        force=np.linalg.norm(force, axis=-1).max(axis=-1),
        map_L1=np.abs(pad).sum(axis=(-2, -1)).max(axis=-1),
        min_pad=tape["pad_min_N"].min(axis=(1, 2)),
        errors=error,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Only the first predeclared root/material for a quick check.",
    )
    parser.add_argument(
        "--asset-root",
        type=Path,
        help="Optional path to robosuite 1.4.1 models/assets.",
    )
    flags = parser.parse_args()
    if flags.workers < 1:
        parser.error("--workers must be positive")
    data = PACKET_DIR / "manifest.json"
    if hashlib.sha256(data.read_bytes()).hexdigest() != verify.MANIFEST_SHA256:
        raise ValueError("Unexpected evidence manifest")
    manifest = json.loads(data.read_bytes())
    rows = roster()
    if {r[0] for r in rows} != {
        Path(n).stem for n in manifest["files"] if n.startswith("cases/")
    }:
        raise ValueError(
            "Regenerated roster differs from original preregistered physical cases"
        )
    for f, digest in manifest["files"].items():
        if hashlib.sha256((PACKET_DIR / f).read_bytes()).hexdigest() != digest:
            raise ValueError("Checksum mismatch for " + f)
    if flags.smoke:
        rows = rows[:1]
    jobs = [(row, flags.asset_root) for row in rows]
    start = time.monotonic()
    if len(jobs) == 1 or flags.workers == 1:
        results = list(map(worker, jobs))
    else:
        with ProcessPoolExecutor(max_workers=min(len(jobs), flags.workers)) as pool:
            results = list(pool.map(worker, jobs))
    refs = verify.references(PACKET_DIR / "reference.csv")
    max_prob_error = 0.0
    for item in results:
        for w, a, j in np.ndindex(2, 3, 3):
            r_own, r_action, r_prob = refs[(item["family"], w, a, verify.FUTURES[j])]
            if r_own != (item["own"] == a) or r_action != item["actions"][w, a, j]:
                raise ValueError(
                    "Fresh policy action differs from the original at " + item["family"]
                )
            max_prob_error = max(
                max_prob_error, float(np.max(np.abs(r_prob - item["probs"][w, a, j])))
            )
    if max_prob_error > 1e-5:
        raise ValueError(
            f"Fresh action probability error {max_prob_error} exceeds 1e-5"
        )
    changes = np.stack([r["changes"] for r in results])
    all_first = changes.sum(axis=(0, 2)).tolist()
    policy_first = (
        np.stack([r["changes"][:, r["own"]] for r in results]).sum(axis=0).tolist()
    )
    loads = np.stack([r["active"] for r in results]).sum(axis=(0, 2, 3)).tolist()
    if not flags.smoke and (all_first, policy_first, loads) != (
        [85, 0],
        [31, 0],
        [156, 0],
    ):
        raise ValueError("Native mechanism readout does not match reported counts")
    print(
        json.dumps(
            dict(
                world_order=["intact", "cut"],
                mujoco=mj.__version__,
                numpy=np.__version__,
                independent_draws=len(set((r[1], r[2]) for r in rows)),
                material_rows=len(rows),
                forced_first_forks=3 * len(rows),
                own_first_forks=len(rows),
                opposite_direction_changes=all_first,
                own_first_direction_changes=policy_first,
                native_intervals_per_world=9 * len(rows),
                with_mating_force=loads,
                fresh_policy_decisions_matching_archive=18 * len(rows),
                max_probability_error=max_prob_error,
                median_peak_B_force_contrast_N=np.median(
                    np.stack([r["force"] for r in results]), axis=(0, 2)
                ).tolist(),
                median_peak_B_pad_map_contrast_N=np.median(
                    np.stack([r["map_L1"] for r in results]), axis=(0, 2)
                ).tolist(),
                minimum_all_four_pad_loads_N=np.stack([r["min_pad"] for r in results])
                .min(axis=0)
                .tolist(),
                max_absolute_archive_errors={
                    k: max(r["errors"][k] for r in results)
                    for k in results[0]["errors"]
                },
                wall_seconds=round(time.monotonic() - start, 2),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
