"""Native per-pad normal and tangential forces on an 8 by 8 grid."""

import numpy as np

GRID = 8


def require(ok, message):
    if not ok:
        raise ValueError(message)


def deposit(grid, pad, position_local, force_local, normal_force, halfwidth):
    """Deposit one contact, retaining edges and all contact partner identities."""
    require(grid.shape == (4, 8, 8, 3), "fixed taxel shape")
    p = np.asarray(position_local, dtype=float)
    f = np.asarray(force_local, dtype=float)
    h = np.asarray(halfwidth, dtype=float)
    require(p.shape == f.shape == (3,) and h.shape == (2,), "contact dimensions")
    require(
        np.isfinite(p).all()
        and np.isfinite(f).all()
        and np.isfinite(normal_force)
        and normal_force >= 0
        and np.all(h > 0),
        "finite contact",
    )
    uv = (p[[0, 2]] / h + 1.0) * 0.5 * (GRID - 1)
    clipped = bool(np.any((uv < 0) | (uv > GRID - 1)))
    uv = np.clip(uv, 0.0, GRID - 1.0)
    lo = np.floor(uv).astype(int)
    hi = np.minimum(lo + 1, GRID - 1)
    frac = uv - lo
    channels = np.array([normal_force, f[0], f[2]])
    for x, wx in ((lo[0], 1 - frac[0]), (hi[0], frac[0])):
        for z, wz in ((lo[1], 1 - frac[1]), (hi[1], frac[1])):
            grid[pad, x, z] += wx * wz * channels
    return clipped


class NativePadSensor:
    def __init__(self, model, pad_ids):
        self.model = model
        self.pad_ids = np.asarray(pad_ids, int).reshape(-1)
        require(
            self.pad_ids.shape == (4,) and len(set(self.pad_ids)) == 4,
            "four unique physical pads",
        )
        self.lookup = {int(g): i for i, g in enumerate(self.pad_ids)}
        self.halfwidth = model.geom_size[self.pad_ids][:, [0, 2]].copy()
        require(
            np.allclose(self.halfwidth, 0.008, atol=1e-12, rtol=0.0),
            "pinned Panda 16mm square pad",
        )
        self.begin_control()

    def begin_control(self):
        self.total = np.zeros((4, 8, 8, 3), dtype=float)
        self.substeps = 0
        self.contacts = 0
        self.clamped_contacts = 0
        self.component_sum = np.zeros((4, 3), dtype=float)

    def sample(self, data):
        import mujoco

        for i, contact in enumerate(data.contact[: data.ncon]):
            a, b = int(contact.geom1), int(contact.geom2)
            if a not in self.lookup and b not in self.lookup:
                continue
            force = np.zeros(6)
            mujoco.mj_contactForce(self.model, data, i, force)
            if force[0] <= 0.0:
                continue
            # MuJoCo contact frame force is force on geom2. No object-id filter.
            world_on_b = np.asarray(contact.frame).reshape(3, 3).T @ force[:3]
            for geom, sign in ((a, -1.0), (b, 1.0)):
                if geom not in self.lookup:
                    continue
                pad = self.lookup[geom]
                rotation = data.geom_xmat[geom].reshape(3, 3)
                local_p = rotation.T @ (contact.pos - data.geom_xpos[geom])
                local_f = rotation.T @ (sign * world_on_b)
                self.clamped_contacts += int(
                    deposit(
                        self.total,
                        pad,
                        local_p,
                        local_f,
                        float(force[0]),
                        self.halfwidth[pad],
                    )
                )
                self.component_sum[pad] += [force[0], local_f[0], local_f[2]]
                self.contacts += 1
        self.substeps += 1

    def packet(self):
        grid = self.total / max(1, self.substeps)
        expected = self.component_sum / max(1, self.substeps)
        error = float(np.max(np.abs(grid.sum(axis=(1, 2)) - expected)))
        require(error < 1e-10, "taxel force component conservation")
        return grid.copy(), dict(
            native_substeps=self.substeps,
            contacts=self.contacts,
            clamped_contacts=self.clamped_contacts,
            conservation_error_n=error,
            force_components_n=expected.tolist(),
            all_contact_partners_included=True,
            sensor_units="N per cell, not pressure or optical pixel intensity",
        )
