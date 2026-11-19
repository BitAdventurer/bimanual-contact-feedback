"""Check the matched gallery replays against the released material-test branches."""

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MEDIA = ROOT / "paper/media"
ACTIONS = ("left", "hold", "right")
MOTIONS = ("large left", "small left", "stationary command", "small right", "large right")
# Values are (hold success, learned follow-up success).
EXAMPLES = {
    "follow-up-comparison": (False, True),
    "preserved-success": (True, True),
    "shared-failure": (False, False),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def first_examples(material):
    names, decisions, physical = (material[key] for key in ("names", "actions", "physical"))
    require(len(set(names)) == len(names), "Duplicate material-test family ID")
    examples = {}
    for row, (name, choices, leaves) in enumerate(zip(names, decisions, physical)):
        for motion, (first, follow_up, outcome) in enumerate(choices):
            require(outcome == leaves[motion, first, follow_up], f"Archived policy outcome mismatch in {name}")
            pair = (bool(leaves[motion, first, 1]), bool(outcome))
            examples.setdefault(pair, (row, str(name), motion, first, follow_up))
    return examples


def endpoint_gates(measurement):
    pads = np.asarray(measurement["pad_N"])
    metrics = [measurement[key] for key in ("depth_mm", "clearance_mm", "angle_rad")]
    require(pads.shape == (2, 2) and np.isfinite(pads).all() and np.isfinite(metrics).all(),
            "Invalid endpoint measurements")
    depth, clearance, angle = metrics
    return [measurement["full"], depth >= 25, clearance >= 0.10, angle < 0.10, pads.min() > 0.01]


def check_endpoints(label, endpoints, path, checks, outcome):
    require(len(endpoints) == len(checks), f"{label}: expected two endpoints")
    for endpoint, reference in zip(endpoints, checks):
        gates = endpoint_gates(endpoint)
        require(np.array_equal(gates, reference[path]), f"{label}: endpoint gates disagree")
        require(endpoint["ok"] == all(gates), f"{label}: reported endpoint result disagrees")
    require(outcome == all(point["ok"] for point in endpoints), f"{label}: final outcome disagrees")


def source_digests():
    selected = json.loads((ROOT / "training/reference/selected.json").read_text())
    actors = selected["selected"]["randomized/ppo/bilateral"]["paths"]
    paths = ["analysis/material.npz", "simulation/data/reference.npz", "training/reference/randomized/encoder.npz"]
    paths.extend("training/reference/" + actor for actor in actors)
    return {path: digest(ROOT / path) for path in paths}


def main():
    sources = source_digests()
    with np.load(ROOT / "analysis/material.npz", allow_pickle=False) as material:
        examples = first_examples(material)
    with np.load(ROOT / "simulation/data/reference.npz", allow_pickle=False) as native:
        families = native["family_id"].tolist()
        require(len(set(families)) == len(families), "Duplicate native family ID")
        checks = (native["before_gates"], native["after_gates"])
        success = native["success"]

    seen = set()
    for stem, pair in EXAMPLES.items():
        receipt = json.loads((MEDIA / (stem + ".json")).read_text())
        row, family, motion, first, follow_up = examples[pair]
        actions = receipt["actions"]
        key = (receipt["family_id"], actions["pin_motion"])
        require(key not in seen, f"{stem}: duplicate replay")
        seen.add(key)
        require(key == (family, MOTIONS[motion]), f"{stem}: not the first archived path for its outcomes")
        require((actions["first_socket"], actions["alt_1"], actions["ppo"]) ==
                (ACTIONS[first], "hold", ACTIONS[follow_up]), f"{stem}: action mismatch")
        require(receipt["source_sha256"] == sources, f"{stem}: source fingerprint mismatch")
        require(receipt["video_sha256"] == digest(MEDIA / (stem + ".mp4")), f"{stem}: MP4 fingerprint mismatch")
        require(receipt["insertion_checks_s"] == [0.88, 1.0], f"{stem}: unexpected scoring times")
        family_index = families.index(family)
        for branch, action, outcome in zip(("hold", "ppo"), (1, follow_up), pair):
            path = (family_index, motion, first, action)
            require(success[path] == outcome, f"{stem}: native {branch} label disagrees")
            check_endpoints(f"{stem} {branch}", receipt["endpoints"][branch], path, checks, outcome)

        if stem != "follow-up-comparison":
            require((receipt["material_packet_row"], receipt["motion_index"]) == (row, motion),
                    f"{stem}: archive indices disagree")
            for index, action in enumerate(ACTIONS):
                option = receipt["all_followups_on_same_motion"][action]
                require(option["action_index"] == index and
                        option["archived_success"] == success[family_index, motion, first, index],
                        f"{stem}: {action} counterfactual disagrees")
        if stem == "shared-failure":
            alternate = receipt["alternative_replay"]
            require(alternate["action"] == "left", "Shared failure: expected a left counterfactual")
            path = (family_index, motion, first, ACTIONS.index("left"))
            check_endpoints(f"{stem} left", [alternate["before"], alternate["after"]], path, checks, success[path])
        print(f"{stem}: {family}, {MOTIONS[motion]}, hold={pair[0]}, learned={pair[1]}")


if __name__ == "__main__":
    main()
