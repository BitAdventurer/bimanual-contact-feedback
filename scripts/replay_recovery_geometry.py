"""Extract socket-frame cross sections for the released recovery film at its native frame times.

Run with the simulation Python, e.g. python scripts/replay_recovery_geometry.py --out geometry.json.
Each outline is center + basis @ [cos(theta), sin(theta)] in millimetres.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from simulation.physics import Panda, commands, prepare, shaft_square_clearance, tail  # noqa: E402
from simulation.task import family_row  # noqa: E402


def cross_section(scene, data, qpos):
    # This separate MjData keeps visualization reads out of the native integrator.
    data.qpos[:] = qpos
    mujoco.mj_kinematics(scene.m, data)
    pin, socket = data.xmat[scene.body].reshape(2, 3, 3)
    tip = data.xpos[scene.body[0]] + pin[:, 2] * 0.135
    tip_local = socket.T @ (tip - data.geom_xpos[scene.rims].mean(axis=0))
    rotation = socket.T @ pin
    axis = rotation[:, 2]
    section = shaft_square_clearance(tip_local, axis)
    if not section["full_cross_section_spans"]:
        raise ValueError(
            "The finite pin does not span both face sections at this frame"
        )

    centers = 1000 * section["face_centers_xy"]
    radius = 1000 * scene.m.geom("free_peg_shaft").size[0]
    # Project the two radius vectors onto each transverse plane along the cylinder axis.
    basis = radius * (rotation[:2, :2] - np.outer(axis[:2] / axis[2], rotation[2, :2]))
    extent = np.linalg.norm(basis, axis=1)
    neg = 17.0 + centers - extent
    pos = 17.0 - centers - extent
    walls = np.stack((neg[:, 0], pos[:, 0], neg[:, 1], pos[:, 1]), axis=1)
    face, wall = np.unravel_index(walls.argmin(), walls.shape)
    clearance = 1000 * section["exact_clearance_m"]
    np.testing.assert_allclose(walls[face, wall], clearance, rtol=0, atol=1e-10)
    return dict(
        centers_mm=centers.round(8).tolist(),
        basis_mm=basis.round(8).tolist(),
        min_clearance_mm=round(float(clearance), 8),
        limiting_face=int(face),
        limiting_wall=("-x", "+x", "-y", "+y")[wall],
        part_world_x_mm=(1000 * data.xpos[scene.body, 0]).round(8).tolist(),
    )


def replay(receipt):
    with np.load(ROOT / "analysis/material.npz", allow_pickle=False) as material:
        row_index = material["names"].tolist().index(receipt["family_id"])
        pin_commands = (
            "large left",
            "small left",
            "stationary command",
            "small right",
            "large right",
        )
        motion = pin_commands.index(receipt["actions"]["pin_motion"])
        first, learned, success = map(int, material["actions"][row_index, motion])
        physical = material["physical"][row_index, motion, first]
        if physical[1] or not physical[learned] or not success:
            raise ValueError(
                "The archive must record a hold failure and learned recovery"
            )
    layout, mass, friction = (
        receipt[key] for key in ("layout", "material_mass_scale", "friction_scale")
    )
    row = family_row(layout, receipt["variant"], mass, friction)
    scene = Panda(layout, mass, friction)
    scene.restore(scene.initial)
    off, _ = prepare(scene, row)
    scene.restore(scene.snapshot())
    begin, mid, _ = commands(row, off, first, motion, 1)

    view_data = mujoco.MjData(scene.m)
    frame_dt = receipt["display_interval_s"]
    stride = round(frame_dt / scene.m.opt.timestep)
    np.testing.assert_allclose(
        stride * scene.m.opt.timestep, frame_dt, rtol=0, atol=1e-12
    )
    frames = [cross_section(scene, view_data, scene.d.qpos)]
    step = mujoco.mj_step
    steps = 0

    def step_and_record(model, data):
        nonlocal steps
        step(model, data)
        steps += 1
        if steps % stride == 0:
            frames.append(cross_section(scene, view_data, data.qpos))

    with patch.object(mujoco, "mj_step", step_and_record):
        scene.phase_obs(off, begin, 0.22, 0.14, 3)
        scene.restore(scene.snapshot())
        scene.phase_obs(begin, mid, 0.32, 0.20, 3, phase="partner")
        fork = scene.snapshot()
        if (
            hashlib.sha256(fork.tobytes()).hexdigest()
            != receipt["matched_state_sha256"]
        ):
            raise ValueError(
                "The native decision snapshot differs from the film receipt"
            )
        prefix = list(frames)
        if len(prefix) != round(receipt["second_decision_s"] / frame_dt) + 1:
            raise ValueError("Unexpected number of frames before branching")
        branches = {}
        for name, action in (("hold", 1), ("ppo", learned)):
            steps = 0
            frames = list(prefix)
            scene.restore(fork)
            _, _, goal = commands(row, off, first, motion, action)
            endpoints = tail(scene, mid, goal)
            for when, actual, ref in zip(
                receipt["insertion_checks_s"], endpoints, receipt["endpoints"][name]
            ):
                if actual["ok"] != ref["ok"]:
                    raise ValueError("Retention outcome differs for " + name)
                for key in ("clearance_mm", "depth_mm", "angle_rad", "pad_N"):
                    np.testing.assert_allclose(actual[key], ref[key], rtol=0, atol=1e-7)
                sample = frames[round(when / frame_dt)]
                np.testing.assert_allclose(
                    sample["min_clearance_mm"], ref["clearance_mm"], rtol=0, atol=1e-7
                )
            branches[name] = frames
    times = [round(i * frame_dt, 6) for i in range(len(branches["hold"]))]
    if (
        len(branches["ppo"]) != len(times)
        or times[-1] != receipt["insertion_checks_s"][-1]
    ):
        raise ValueError("Replay duration differs from the film")
    return dict(
        family_id=receipt["family_id"],
        matched_state_sha256=receipt["matched_state_sha256"],
        half_opening_mm=17.0,
        face_z_mm=[-9.0, 9.0],
        clearance_threshold_mm=0.1,
        outline="centers_mm[face] + basis_mm @ [cos(theta), sin(theta)] in socket coordinates",
        part_world_x_order=["pin", "socket"],
        time_s=times,
        branches=branches,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="new JSON output file")
    args = parser.parse_args()
    receipt = json.loads((ROOT / "paper/media/follow-up-comparison.json").read_text())
    result = replay(receipt)
    with args.out.open("x") as dst:
        json.dump(result, dst, separators=(",", ":"), allow_nan=False)
        dst.write("\n")
