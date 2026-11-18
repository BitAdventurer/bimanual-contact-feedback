"""Build a physical scene and regenerate a nuisance draw without loading any outcomes."""

import hashlib
import importlib.util
import json
from functools import lru_cache
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

DATA = Path(__file__).resolve().parent / "data"
TEST_MATERIALS = ((1.125, 0.8), (1.375, 0.8), (1.625, 0.8), (1.5, 0.6))


@lru_cache(maxsize=2)
def assets(directory=None):
    if directory is None:
        package = importlib.util.find_spec("robosuite")
        if package is None or package.origin is None:
            raise RuntimeError(
                "Install the robosuite 1.4.1 asset wheel or supply --asset-root."
            )
        directory = Path(package.origin).parent / "models" / "assets"
    root = Path(directory).resolve()
    hashes = json.loads((DATA / "assets.sha256.json").read_text())
    for name, expected in hashes.items():
        asset = root / name
        if (
            not asset.is_file()
            or hashlib.sha256(asset.read_bytes()).hexdigest() != expected
        ):
            raise ValueError(f"Missing or changed robosuite 1.4.1 asset: {name}")
    return root


def family_row(layout, variant, mass, friction):
    if (
        not 1 <= layout <= 8
        or variant < 0
        or not np.isfinite([mass, friction]).all()
        or min(mass, friction) <= 0
    ):
        raise ValueError(
            "Use layout 1–8, a nonnegative variant index and positive scales."
        )
    test = layout in (7, 8)
    seed = [936371 if test else 935371, layout, variant]
    g = np.random.default_rng(np.random.SeedSequence(seed))
    minor = np.random.default_rng(np.random.SeedSequence(seed + [1]))
    offset = (-1, 1)[variant % 2] * g.uniform(0.1, 2.5)
    return dict(
        layout=layout,
        variant_index=variant,
        mass_scale=mass,
        friction_scale=friction,
        offset_mm=offset,
        common_mm=g.uniform(-0.8, 0.8),
        first_mm=g.uniform(1.6, 2.4),
        partner_mm=g.uniform(4.3, 5.7),
        second_mm=g.uniform(4.3, 5.7),
        small_mm=minor.uniform(0.75, 2.25),
    )


def make_model(layout, mass, friction, asset_root=None, *, mating_contact=True):
    root = assets(asset_root)
    xml = ET.parse(DATA / "scene.xml")
    positions = json.loads((DATA / "layouts.json").read_text())[str(layout)]
    for geom in xml.iter("geom"):
        if geom.get("name") in positions:
            geom.set("pos", positions[geom.get("name")])
    for el in xml.iter():
        name = el.get("file")
        if name:
            el.set("file", str(root / name))
    for part in ("free_peg", "free_socket"):
        for geom in xml.find(f".//body[@name='{part}']").findall("geom"):
            geom.set("mass", format(float(geom.attrib["mass"]) * mass, ".17g"))
    for pair in xml.findall("./contact/pair"):
        if pair.get("name", "").startswith("v93_grip_"):
            values = [float(x) for x in pair.attrib["friction"].split()]
            values[:2] = [2.0 * friction] * 2
            pair.set("friction", " ".join(format(v, ".17g") for v in values))
    if not mating_contact:
        contact = xml.find("contact")
        excluded = any(
            {x.get("body1"), x.get("body2")} == {"free_peg", "free_socket"}
            for x in contact.findall("exclude")
        )
        pairs = []
        for pair in contact.findall("pair"):
            left, right = sorted((pair.attrib["geom1"], pair.attrib["geom2"]))
            if left.startswith("free_peg_") and right.startswith("free_socket_"):
                pairs.append(pair)
        if not excluded or len(pairs) != 18:
            raise RuntimeError("Could not isolate the expected peg–socket contacts.")
        for pair in pairs:
            contact.remove(pair)
    model = mujoco.MjModel.from_xml_string(
        ET.tostring(xml.getroot(), encoding="unicode")
    )
    got = [model.body(part).mass[0] for part in ("free_peg", "free_socket")]
    if model.npair != (22 if mating_contact else 4):
        raise RuntimeError("Unexpected explicit contact topology.")
    if not np.allclose(got, np.array([0.085, 0.125]) * mass, rtol=1e-12, atol=0):
        raise RuntimeError(
            "Compiled free-body masses do not match the requested material."
        )
    return model
