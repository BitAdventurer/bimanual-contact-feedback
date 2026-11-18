"""Native two-Panda contact dynamics, observations, and paired action branches."""

import hashlib
import numpy as np
import mujoco as mj

if __package__:
    from .sensor import NativePadSensor
    from .task import DATA, make_model
else:
    from sensor import NativePadSensor
    from task import DATA, make_model

INTEGRATION = mj.mjtState.mjSTATE_INTEGRATION
SENSORS = ("proprio", "w", "p", "partner_p")
FUTURES = (
    "negative_large",
    "negative_small",
    "neutral",
    "positive_small",
    "positive_large",
)
END_KEYS = ("depth_mm", "clearance_mm", "angle_rad", "mating_N", "pad_N", "full")


def shaft_square_clearance(
    tip_local,
    axis_local,
    half_width=0.017,
    plate_half=0.009,
    radius=0.014,
    length=0.120,
):
    """Analytic cylinder/parallel-plane geometry, independent of MuJoCo state.

    Each coordinate extent of the oblique cylinder section is
    r*sqrt(1+(v_xy/v_z)**2). The isotropic r/v_z bound is retained for audit.
    Full-section spanning additionally excludes intersection with either endcap.
    """
    tip = np.asarray(tip_local, dtype=float)
    axis = np.asarray(axis_local, dtype=float)
    if (
        tip.shape != (3,)
        or axis.shape != (3,)
        or not np.isfinite(np.r_[tip, axis]).all()
        or np.linalg.norm(axis) < 1e-12
    ):
        raise ValueError("finite tip and nonzero axis vectors required")
    axis = axis / np.linalg.norm(axis)
    if axis[2] <= 0.5:
        return dict(
            exact_clearance_m=-1.0,
            conservative_clearance_m=-1.0,
            spans=False,
            full_cross_section_spans=False,
            face_centers_xy=np.zeros((2, 2)),
            face_parameters=np.zeros(2),
            axis_extents_xy=np.zeros(2),
        )
    parameters = (np.asarray([-plate_half, plate_half]) - tip[2]) / axis[2]
    faces = tip[:2] + parameters[:, None] * axis[:2]
    extents = radius * np.sqrt(1 + (axis[:2] / axis[2]) ** 2)
    cap_margin = radius * np.sqrt(max(0.0, 1 - axis[2] ** 2)) / axis[2]
    spans = bool(np.all(parameters >= -length) and np.all(parameters <= 0))
    full = bool(
        np.all(parameters - cap_margin >= -length)
        and np.all(parameters + cap_margin <= 0)
    )
    return dict(
        exact_clearance_m=float(np.min(half_width - np.abs(faces) - extents)),
        conservative_clearance_m=float(
            np.min(half_width - np.abs(faces) - radius / axis[2])
        ),
        spans=spans,
        full_cross_section_spans=full,
        face_centers_xy=faces,
        face_parameters=parameters,
        axis_extents_xy=extents,
    )


class Scene:
    def __init__(self, layout, mass, friction, asset_root=None, *, mating_contact=True):
        self.m = m = make_model(
            layout, mass, friction, asset_root, mating_contact=mating_contact
        )
        self.d = mj.MjData(m)
        with np.load(DATA / "initial.npz", allow_pickle=False) as z:
            for name in ("qpos", "qvel", "ctrl"):
                getattr(self.d, name)[:] = z[f"{layout}_{name}"]
            self.d.time = float(z[f"{layout}_time"])
        assert abs(m.opt.timestep - 0.0005) < 1.0e-9
        mj.mj_forward(m, self.d)
        self.ids = np.array(
            [
                [m.joint("robot%d_joint%d" % (a, j)).id for j in range(1, 8)]
                for a in range(2)
            ]
        )
        self.q = m.jnt_qposadr[self.ids]
        self.v = m.jnt_dofadr[self.ids]
        self.act = np.array(
            [
                [m.actuator("robot%d_torq_j%d" % (a, j)).id for j in range(1, 8)]
                for a in range(2)
            ]
        )
        self.sites = [m.site("gripper%d_grip_site" % a).id for a in range(2)]
        self.origin = self.d.qpos[self.q].copy()
        self.eq_torque = self.d.ctrl[self.act].copy()
        self.gains = np.array([420.0, 420.0, 360.0, 340.0, 180.0, 120.0, 80.0])
        self.damping = np.array([30.0, 30.0, 24.0, 22.0, 11.0, 8.0, 6.0])
        self.offset = []
        for arm in range(2):
            jp = np.zeros((3, m.nv))
            jr = jp.copy()
            mj.mj_jacSite(m, self.d, jp, jr, self.sites[arm])
            J = np.concatenate((jp[:, self.v[arm]], jr[:, self.v[arm]]))
            self.offset.append(
                J.T
                @ np.linalg.solve(
                    J @ J.T + 1.0e-5 * np.eye(6), np.array([1.0, 0, 0, 0, 0, 0])
                )
            )
        self.offset = np.array(self.offset)
        self.body = [m.body(s).id for s in ["free_peg", "free_socket"]]
        self.rims = [m.geom("free_socket_rim%d" % j).id for j in range(4)]
        self.peg = {m.geom(s).id for s in ["free_peg_shaft", "free_peg_shoulder_v10"]}
        self.h = [m.geom(s).id for s in ["free_peg_handle", "free_socket_handle"]]
        self.pads = {
            (m.geom("gripper%d_finger%d_pad_collision" % (a, j + 1)).id, self.h[a]): (
                a,
                j,
            )
            for a in range(2)
            for j in range(2)
        }

    def snapshot(self):
        out = np.zeros(mj.mj_stateSize(self.m, INTEGRATION))
        mj.mj_getState(self.m, self.d, out, INTEGRATION)
        return out

    def restore(self, state):
        mj.mj_setState(self.m, self.d, state, INTEGRATION)
        mj.mj_forward(self.m, self.d)

    def contact(self):
        pads = np.zeros((2, 2))
        cross = 0.0
        force = np.empty(6)
        for i, c in enumerate(self.d.contact[: self.d.ncon]):
            a, b = int(c.geom1), int(c.geom2)
            pad = self.pads.get((a, b), self.pads.get((b, a)))
            mating = (a in self.peg and b in self.rims) or (
                b in self.peg and a in self.rims
            )
            if pad is None and not mating:
                continue
            mj.mj_contactForce(self.m, self.d, i, force)
            if pad is not None:
                pads[pad] += force[0]
            if mating:
                cross += force[0]
        return pads, cross

    def measured(self):
        mj.mj_forward(self.m, self.d)
        pos = self.d.xpos[self.body]
        peg, socket = self.d.xmat[self.body].reshape(2, 3, 3)
        tip = pos[0] + peg @ np.array([0.0, 0.0, 0.135])
        local = socket.T @ (tip - self.d.geom_xpos[self.rims].mean(0))
        axis = socket.T @ peg[:, 2]
        g = shaft_square_clearance(local, axis)
        pads, cf = self.contact()
        angle = np.arccos(np.clip(peg[:, 2] @ socket[:, 2], -1.0, 1.0))
        depth = local[2] + 0.009
        gap = g["exact_clearance_m"]
        return dict(
            ok=bool(
                g["full_cross_section_spans"]
                and depth >= 0.025
                and gap >= 0.0001
                and angle < 0.10
                and pads.min() > 0.01
            ),
            depth_mm=float(depth * 1000),
            clearance_mm=float(gap * 1000),
            angle_rad=float(angle),
            pad_N=pads.tolist(),
            mating_N=float(cf),
            hand_x_mm=(1000 * self.d.site_xpos[self.sites, 0]).tolist(),
            part_x_mm=(1000 * pos[:, 0]).tolist(),
        )


class Panda(Scene):
    def __init__(self, layout, mass, friction, asset_root=None, *, mating_contact=True):
        super().__init__(
            layout, mass, friction, asset_root, mating_contact=mating_contact
        )
        ids = np.array(
            [
                self.m.geom(f"gripper{a}_finger{j + 1}_pad_collision").id
                for a in (0, 1)
                for j in (0, 1)
            ]
        )
        self.sensor = NativePadSensor(self.m, ids)
        self.bp = set(ids[2:])
        self.initial = self.snapshot()
        self.limits = self.m.actuator_ctrlrange[self.act]

    def packet(self):
        mj.mj_forward(self.m, self.d)
        self.sensor.begin_control()
        self.sensor.sample(self.d)
        grids, diagnostic = self.sensor.packet()
        net = np.zeros(6)
        force = np.empty(6)
        origin = self.d.site_xpos[self.sites[1]]
        for i, c in enumerate(self.d.contact[: self.d.ncon]):
            g1, g2 = int(c.geom1), int(c.geom2)
            if not (g1 in self.bp or g2 in self.bp):
                continue
            mj.mj_contactForce(self.m, self.d, i, force)
            orientation = c.frame.reshape(3, 3).T
            for geom, sgn in ((g1, -1), (g2, 1)):
                if geom not in self.bp:
                    continue
                linear = sgn * (orientation @ force[:3])
                net[:3] += linear
                net[3:] += sgn * (orientation @ force[3:]) + np.cross(
                    c.pos - origin, linear
                )
        rot = self.d.site_xmat[self.sites[1]].reshape(3, 3).T
        net[:3] = rot @ net[:3]
        net[3:] = rot @ net[3:]
        return dict(
            w=net,
            proprio=np.r_[self.d.qpos[self.q[1]], self.d.qvel[self.v[1]]],
            p=grids[2:, :, :, 0].sum(0),
            partner_p=grids[:2, :, :, 0].sum(0),
        )

    def phase_obs(self, start, goal, secs, ramp, count=0, cb=None, phase=""):
        dt = self.m.opt.timestep
        n = round(secs / dt)
        assert abs(n * dt - secs) < 1.0e-9
        ticks = (
            {int(round(i * n / count)) for i in range(1, count + 1)} if count else set()
        )
        obs = []
        peak = 0.0
        # Targets are identical to the parent's regulator; precompute smoothly ramped joint path.
        alpha = np.minimum(1.0, np.arange(1, n + 1) * dt / ramp)
        alpha = alpha * alpha * (3 - 2 * alpha)
        dx = start[None, :] * (1 - alpha[:, None]) + goal[None, :] * alpha[:, None]
        qs = self.origin[None] + dx[:, :, None] * self.offset[None] / 1000.0
        for t in range(n):
            torque = (
                self.eq_torque
                + self.gains * (qs[t] - self.d.qpos[self.q])
                - self.damping * self.d.qvel[self.v]
            )
            self.d.ctrl[self.act] = np.minimum(
                np.maximum(torque, self.limits[:, :, 0]), self.limits[:, :, 1]
            )
            mj.mj_step(self.m, self.d)
            if (t + 1) % 10 == 0 and phase == "partner":
                peak = max(peak, self.contact()[1])
            if t + 1 in ticks:
                o = self.packet()
                obs.append(o)
                if cb:
                    cb(self, phase, o)
        if np.any(np.array([warn.number for warn in self.d.warning])):
            raise ArithmeticError("MuJoCo warning during branch")
        return obs, peak

    def truth(self):
        measure = self.measured()
        peg, socket = self.d.xmat[self.body].reshape(2, 3, 3)
        pos = self.d.xpos[self.body]
        tip = pos[0] + peg @ np.array([0.0, 0.0, 0.135])
        local = socket.T @ (tip - self.d.geom_xpos[self.rims].mean(0))
        g = shaft_square_clearance(local, socket.T @ peg[:, 2])
        measure["full"] = bool(g["full_cross_section_spans"])
        return measure


def prepare(sc, r, cb=None):
    z = np.zeros(2)
    off = np.array([r["common_mm"], r["common_mm"] + r["offset_mm"]])
    sc.phase_obs(z, z, 0.20, 0.14)
    sc.phase_obs(z, off, 0.22, 0.14)
    root, _ = sc.phase_obs(off, off, 0.12, 0.10, 6, cb, "root")
    return off, root


def rollout_policy(scene, row, future, choose_first, choose_second):
    """Run two policy calls on measured sensor histories, with no outcomes or hidden attributes as inputs.

    choose_first(root) and choose_second(first, root, mid) must return 0, 1 or 2.
    Returns both actions and physically measured before-/after-hold diagnostics.
    """
    if future not in range(5):
        raise ValueError("The physical partner future must be an index from 0 to 4.")
    scene.restore(scene.initial)
    offset, initial_samples = prepare(scene, row)

    def history(samples):
        return {
            name: np.stack([sample[name] for sample in samples]) for name in SENSORS
        }

    root = history(initial_samples)
    first_action = int(choose_first(root))
    if first_action not in range(3):
        raise ValueError("Invalid first action.")
    first, mid, _ = commands(row, offset, first_action, future, 1)
    initial, _ = scene.phase_obs(offset, first, 0.22, 0.14, 3)
    partner, _ = scene.phase_obs(first, mid, 0.32, 0.20, 3, phase="partner")
    second_action = int(choose_second(first_action, root, history(initial + partner)))
    if second_action not in range(3):
        raise ValueError("Invalid second action.")
    _, _, goal = commands(row, offset, first_action, future, second_action)
    before, after = tail(scene, mid, goal)
    return first_action, second_action, before, after


def commands(row, off, a, z, b):
    big, small = row["partner_mm"], row["small_mm"]
    first = off + np.array([0.0, (a - 1) * row["first_mm"]])
    mid = first + np.array([(-big, -small, 0.0, small, big)[z], 0.0])
    goal = mid + np.array([0.0, (b - 1) * row["second_mm"]])
    return first, mid, goal


def tail(sc, mid, goal, cb=None):
    sc.phase_obs(mid, goal, 0.34, 0.20)
    before = sc.truth()
    if cb:
        cb(sc, "finish", sc.packet())
    sc.phase_obs(goal, goal, 0.12, 0.10)
    after = sc.truth()
    if cb:
        cb(sc, "hold", sc.packet())
    return before, after


def simulate_family(sc, row):
    sc.restore(sc.initial)
    off, root = prepare(sc, row)
    initial = sc.snapshot()
    f = len(FUTURES)
    shape = (f, 3, 3)
    d = dict(
        success=np.empty(shape, bool),
        fork_digests=np.empty((f, 3), dtype="U64"),
        partner_part_x_mm=np.empty((f, 3, 2)),
        partner_part_dx_mm=np.empty((f, 3, 2)),
        partner_arm_q_delta_rad=np.empty((f, 3, 2, 7)),
        partner_mating_peak_N=np.empty((f, 3)),
    )
    # Capture initial validity without filtering. Every first-action branch still
    # restores the full archived integration state saved before these diagnostics.
    entry = sc.truth()
    for name in END_KEYS:
        d["root_" + name] = np.asarray(entry[name])
    counts = np.zeros((2, 2), dtype=int)
    mu = np.full((2, 2, 2), -1.0)
    for contact in sc.d.contact[: sc.d.ncon]:
        g1, g2 = int(contact.geom1), int(contact.geom2)
        where = sc.pads.get((g1, g2), sc.pads.get((g2, g1)))
        if where is not None:
            counts[where] += 1
            mu[where] = contact.friction[:2]
    d.update(
        root_ok=bool(entry["ok"]),
        root_pad_contacts=counts,
        root_observed_pad_mu=mu,
        material_mass_kg=sc.m.body_mass[sc.body].copy(),
        material_inertia_kgm2=sc.m.body_inertia[sc.body].copy(),
        stratum=2 * int(row["offset_mm"] > 0) + int(abs(row["offset_mm"]) >= 1.3),
    )
    for k in SENSORS:
        arr = np.stack([s[k] for s in root])
        d["root_" + k] = arr
        d["mid_" + k] = np.empty((f, 3) + arr.shape, arr.dtype)
    for phase in ("before_", "after_"):
        for k in END_KEYS:
            d[phase + k] = np.empty(
                shape + ((2, 2) if k == "pad_N" else ()), bool if k == "full" else float
            )
    for a in range(3):
        sc.restore(initial)
        first, _, _ = commands(row, off, a, 2, 1)
        start, _ = sc.phase_obs(off, first, 0.22, 0.14, 3)
        fork = sc.snapshot()
        for z in range(f):
            sc.restore(fork)
            d["fork_digests"][z, a] = hashlib.sha256(
                sc.snapshot().tobytes()
            ).hexdigest()
            q0 = sc.d.qpos[sc.q].copy()
            pos0 = sc.d.xpos[sc.body, 0].copy()
            _, mid, _ = commands(row, off, a, z, 1)
            following, peak = sc.phase_obs(first, mid, 0.32, 0.20, 3, phase="partner")
            saved = sc.snapshot()
            d["partner_mating_peak_N"][z, a] = peak
            d["partner_arm_q_delta_rad"][z, a] = sc.d.qpos[sc.q] - q0
            d["partner_part_x_mm"][z, a] = sc.d.xpos[sc.body, 0] * 1000.0
            d["partner_part_dx_mm"][z, a] = (sc.d.xpos[sc.body, 0] - pos0) * 1000.0
            for k in SENSORS:
                d["mid_" + k][z, a] = np.stack([step[k] for step in start + following])
            for b in range(3):
                sc.restore(saved)
                _, _, goal = commands(row, off, a, z, b)
                before, after = tail(sc, mid, goal)
                for phase, record in (("before_", before), ("after_", after)):
                    for k in END_KEYS:
                        d[phase + k][z, a, b] = record[k]
                d["success"][z, a, b] = before["ok"] and after["ok"]
    return d
