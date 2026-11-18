"""Deterministic decisions, exact terminal scoring and paired bootstrap uncertainty."""

import numpy as np
from observations import TEST_MATERIALS

TIE3 = np.array([1, 0, 2])
TIE9 = np.array(
    sorted(range(9), key=lambda k: ((k // 3 != 1) + (k % 3 != 1), k // 3 != 1, k))
)


def choose(nets, x, head, probability):
    scores = []
    for net in nets:
        h = np.tanh(x @ net[head + "W0"].T + net[head + "b0"])
        h = np.tanh(h @ net[head + "W1"].T + net[head + "b1"])
        value = h @ net[head + "W2"].T + net[head + "b2"]
        if probability:
            e = np.exp(value - value.max(-1, keepdims=True))
            value = e / e.sum(-1, keepdims=True)
        scores.append(value)
    scores = np.mean(scores, axis=0)
    order = TIE9 if scores.shape[-1] == 9 else TIE3
    return order[np.argmax(scores[..., order], axis=-1)]


def decisions(nets, method, r, m):
    probability = method.startswith("ppo")
    a = choose(nets, r, "r", probability)
    if method.endswith("_committed"):
        pair = np.stack((a // 3, a % 3), axis=-1)
        return np.broadcast_to(pair[:, None], (len(r), 5, 2)).copy()
    reached = m[np.arange(len(r))[:, None], np.arange(5)[None, :], a[:, None]]
    b = choose(nets, reached, "m", probability)
    return np.stack((np.broadcast_to(a[:, None], b.shape), b), axis=-1)


def success(data, acts):
    return data["success"][
        np.arange(len(acts))[:, None],
        np.arange(5)[None, :],
        acts[:, :, 0],
        acts[:, :, 1],
    ]


def validation_score(data, acts):
    y = success(data, acts)
    return (
        int(y.sum()),
        -int((~y & data["success"][:, :, 1, 1]).sum()),
        -int((acts != 1).sum()),
    )


def ordered_rows(data):
    paired = np.unique(data["paired_root"])
    lookup = {
        pair: i for i, pair in enumerate(zip(data["paired_root"], data["condition"]))
    }
    ids = np.array([[lookup[(root, c)] for c in TEST_MATERIALS] for root in paired])
    if (
        ids.shape != (64, 4)
        or len(np.unique(ids)) != 256
        or not (data["layout"][ids] == data["layout"][ids[:, :1]]).all()
    ):
        raise ValueError(
            "Expected 64 paired roots crossed with all four material settings"
        )
    return ids


def cohort_metrics(y, ids):
    result = y[ids]
    return dict(
        interpolated=int(result[:, :3].sum()),
        interpolated_n=result[:, :3].size,
        familiar=int(result[:, 3].sum()),
        familiar_n=result[:, 3].size,
        by_material={c: int(result[:, i].sum()) for i, c in enumerate(TEST_MATERIALS)},
    )


def paired_effect(dr, nominal, ids, layouts):
    d = dr[ids].astype(int) - nominal[ids].astype(int)
    d = d[:, :3, :]
    layout = layouts[ids[:, 0]]
    values = d.mean((1, 2))
    rng = np.random.default_rng(930093)
    weights = np.zeros((10000, len(values)))
    for label in np.unique(layout):
        members = np.flatnonzero(layout == label)
        w = rng.multinomial(
            len(members), np.full(len(members), 1 / len(members)), size=len(weights)
        )
        weights[:, members] = w / len(values)
    return dict(
        base_roots=len(values),
        evaluated_futures=d.size,
        gains=int((d > 0).sum()),
        losses=int((d < 0).sum()),
        difference_pp=100 * float(values.mean()),
        paired_ci95_pp=np.percentile(100 * (weights @ values), [2.5, 97.5]).tolist(),
    )


def response_alignment(data, ids, acts):
    """Keep B's first action fixed; use B's follow-up from a different A motion."""
    rows = ids[:, :3].reshape(-1)
    a = acts[rows, 0, 0]
    if not np.all(acts[rows, :, 0] == a[:, None]):
        raise ValueError("First action changes with a future that has yet to occur")
    b = acts[rows, :, 1]
    reward = data["success"][rows]
    # Axes after initial condition: actual future, donor history.
    crossed = reward[
        np.arange(len(rows))[:, None, None],
        np.arange(5)[None, :, None],
        a[:, None, None],
        b[:, None, :],
    ]
    fixed_pair = reward.sum(axis=1).max(axis=(1, 2))
    possible = reward[
        np.arange(len(rows))[:, None], np.arange(5)[None, :], a[:, None]
    ].any(axis=-1)
    return dict(
        actual_history=int(np.trace(crossed.sum(axis=0))),
        independent_history_expected=float(crossed.sum() / 5),
        best_fixed_pair=int(fixed_pair.sum()),
        best_followup_given_first=int(possible.sum()),
        outcome_count=len(rows) * 5,
        actual_future_by_observation_donor=crossed.sum(axis=0).tolist(),
    )
