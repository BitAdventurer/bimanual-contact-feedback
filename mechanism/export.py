"""Rebuild the distributed contact-cut subset from the versioned research archive."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil

# Fingerprint of the original policy comparison. Its linked receipts specify
# the physical cases, the archived predictions, and the selected checkpoints.
READOUT_SHA256 = "391d3f27041b3f3331e8cf068d5ecd28b0bf18282803c96dbb8fc5277fcf9d98"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked(path, expected):
    if digest(path) != expected:
        raise ValueError(f"Archive digest mismatch for {path.name}")
    return path


def read_json(path, expected):
    return json.loads(checked(path, expected).read_text())


def export(study, destination):
    archive = study / "artifacts"
    physics = archive / "cooperative_panda_transmission_v93r"
    policy = archive / "cooperative_panda_one_hand_domain_v93r"
    original = read_json(policy / "contact_cut.json", READOUT_SHA256)
    results = read_json(physics / "results.json", original["physics_sha256"])
    release = read_json(policy / "release.json", original["policy_sha256"])
    selection = read_json(policy / "models/selected.json", release["models"])
    checkpoint = selection["selected"]["ppo_local_b"]
    if len(results["cases_sha"]) != 32 or checkpoint["step"] != 192:
        raise ValueError("Unexpected physical sample or selected step")

    destination.mkdir(parents=True, exist_ok=False)
    (destination / "cases").mkdir()
    (destination / "models").mkdir()
    files = {}

    def copy(source, name, expected):
        shutil.copyfile(checked(source, expected), destination / name)
        files[name] = expected

    for family, sha256 in sorted(results["cases_sha"].items()):
        name = "cases/" + family + ".npz"
        copy(physics / name, name, sha256)
    for stage in ("root", "mid"):
        name = stage + "_encoder.npz"
        copy(policy / "models" / name, "models/" + name, selection["files"][name])
    if len(checkpoint["paths"]) != 3:
        raise ValueError("Expected three independently initialized policy checkpoints")
    for seed, source in zip((9201, 9202, 9203), checkpoint["paths"]):
        if source != f"local_b/ppo/{seed}/step_192.npz":
            raise ValueError("Unexpected checkpoint path")
        copy(
            policy / "models" / source,
            f"models/seed_{seed}.npz",
            selection["files"][source],
        )

    # Archival decisions serve only as external targets for the verification pass.
    # All probabilities in the numerical readout are recomputed from observations.
    old_csv = checked(policy / "cut_predictions.csv", original["data_sha256"])
    reference_path = destination / "reference.csv"
    with old_csv.open(newline="") as src, reference_path.open("w", newline="") as out:
        reader = csv.DictReader(src)
        writer = csv.DictWriter(out, fieldnames=reader.fieldnames, lineterminator="\n")
        writer.writeheader()
        count = 0
        for row in reader:
            if row["policy"] == "local_b":
                writer.writerow(row)
                count += 1
    if count != 576:
        raise ValueError(f"Expected 576 socket-local references, found {count}")
    files["reference.csv"] = digest(destination / "reference.csv")
    metadata = dict(
        source={
            "policy_readout": READOUT_SHA256,
            "policy_release": original["policy_sha256"],
            "checkpoint_selection": release["models"],
            "native_physics_results": original["physics_sha256"],
            "full_original_prediction_table": original["data_sha256"],
            "original_inference_script": original["software_sha256"],
            "contact_decision_manifest": original["manifest_sha256"],
        },
        selected_steps=checkpoint["step"],
        seeds=[9201, 9202, 9203],
        files=files,
    )
    (destination / "manifest.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "study", type=Path, help="Study directory containing the original artifacts/"
    )
    parser.add_argument(
        "destination", type=Path, help="New, nonexisting output directory"
    )
    args = parser.parse_args()
    export(args.study, args.destination)
