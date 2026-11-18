"""Exact retrospective controls for aligning a locked second decision with its own future."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
PACKET_HASHES = {
    "material": "a3fafd0da323ebc46d11e459c00d2e7c06dc10e871d4621b05d4461eccdd77a5",
    "fresh": "145dc8ae62ed972e8f56410a1b8d0eeb76dc45bee0639757898fad57ff0150f1",
}


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def score(study):
    main = study == "material"
    packet = BASE / (study + ".npz")
    packet_digest = file_hash(packet)
    if packet_digest != PACKET_HASHES[study]:
        raise ValueError(f"Input fingerprint mismatch for {packet.name}")
    with np.load(packet, allow_pickle=False) as data:
        physical, names, cluster, layouts, chosen = [data[key] for key in ("physical", "names", "cluster", "layouts", "actions")]
    assert physical.shape == (len(names), 5, 3, 3)
    assert chosen.shape == (len(names), 5, 3)
    assert physical.dtype == bool and np.isin(chosen[:, :, :2], [0, 1, 2]).all()
    _, per_root = np.unique(cluster, return_counts=True)
    assert len(per_root) == (64 if main else 12)
    assert np.all(per_root == (3 if main else 6))
    counts = []
    crossings = []
    for decisions, leaves in zip(chosen, physical):
        a = decisions[0][0]
        assert all(q[0] == a for q in decisions)
        b = np.array([q[1] for q in decisions])
        branch = leaves[:, a, :]
        observed = branch[np.arange(5), b]
        assert np.array_equal(observed, [q[2] for q in decisions])
        # Row z specifies the evaluated world; column t supplies the action the same
        # model chose on a sensor history arising from partner motion t at this root.
        cross = branch[:, b]
        crossings.append(cross)
        counts.append([observed.sum(), cross.sum()/5, branch.sum(0).max(),
                       leaves.sum(0).max(), branch.any(1).sum()])
    rows = np.asarray(counts)
    aligned, unrelated, fixed_second, fixed_pair, informed = rows.T
    assert np.all(unrelated <= fixed_second + 1.e-10)
    assert np.all(fixed_second <= fixed_pair)
    assert np.all(aligned <= informed)
    ordered = sorted(set(cluster))
    # Preserve the matched materials and all futures for each nuisance draw.
    clustered = np.array([rows[cluster == root].mean(0)/5 for root in ordered])
    groups = np.array([int(layouts[cluster == root][0]) if main else 0 for root in ordered])
    rng = np.random.default_rng(261119)
    draws = np.concatenate([rng.choice(np.flatnonzero(groups == k),
                            size=(10000, (groups == k).sum()), replace=True) for k in np.unique(groups)], axis=1)
    matrix = np.mean(crossings, axis=0)
    assert np.isclose(matrix.mean() * 5 * len(rows), unrelated.sum())
    assert np.isclose(matrix.trace() * len(rows), aligned.sum())
    labels = ["aligned", "independent_history", "best_fixed_second", "best_fixed_pair", "best_informed_second"]
    by_cluster = {name: v.tolist() for name,v in zip(ordered,clustered)}
    results = {lab: float(col.sum()) for lab,col in zip(labels,rows.T)}
    deltas = {}
    for j in range(1,len(labels)):
        diff = (clustered[:,0] - clustered[:,j])*100
        boot = diff[draws].mean(1)
        deltas[labels[j]] = dict(pp=float(diff.mean()), ci95=np.percentile(boot,[2.5,97.5]).tolist())
    return dict(assigned_futures=5*len(rows), n_roots=len(ordered), scores=results, aligned_minus=deltas, response_matrix=matrix.tolist(),
                cluster_columns=labels, cluster_rates={str(k):v for k,v in by_cluster.items()},
                sha256=packet_digest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/alignment.json"))
    args = parser.parse_args()
    d = dict(status="Retrospective deterministic readout of frozen policies and complete archived branches. "
                    "Each individual evaluated motion receives its own matched history with probability 1/5. "
                    "The best fixed pair uses full outcome labels per root-material but cannot depend on future direction. "
                    "The best informed second uses the policy's own first action. "
                    "Intervals condition on the trained ensemble and observed geometries.",
             material=score("material"), fresh=score("fresh"))
    assert d["material"]["scores"]["aligned"]==709 and d["fresh"]["scores"]["aligned"]==289
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(d, indent=2)+"\n")
    for k in ("material","fresh"):
        print(k,d[k]["scores"],d[k]["aligned_minus"])
