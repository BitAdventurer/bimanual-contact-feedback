# Native two-Panda replay

The simulator re-integrates the contact-recovery segment with MuJoCo at 0.5 ms. Arm A holds a free peg and arm B holds a free socket. Both have seven torque-controlled joints and two finger pads. Each episode starts from an archived grasp. A first residual motion of B precedes an unannounced movement of A. B can measure six samples of its joints, its six-axis wrench, and both hands’ 8×8 normal-force maps before choosing a second residual. The two gripper fingers contribute to each hand map. Cell units are newtons.

## Install and run

From the checkout root on Linux x86_64, use Python 3.10 with binary wheels:

```sh
python3.10 -m venv .venv-sim
. .venv-sim/bin/activate
python -m pip install --only-binary=:all: -r simulation/requirements.txt
python -m pip install --only-binary=:all: --no-deps robosuite==1.4.1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python simulation/replay.py smoke
```

The same dependency pins and commands apply to Python 3.8.20. The original research environment used that interpreter. A new Python 3.10.12 environment installed directly from the public wheels reproduced all 47 archived fields of the full held-out grid bit-for-bit after matching on family ID. No GPU or graphical display is required. The runner reads the installed robosuite wheel's assets directly and verifies all 66 against their archived hashes. It does not import the robosuite environment. If the asset wheel is unpacked outside the environment, supply `--asset-root /path/to/robosuite/models/assets`.

## Reproduce the physical trials

```sh
python simulation/replay.py family --layout 8 --variant 12 --mass 1.375 --friction 0.8 --output family.npz
python simulation/replay.py test-grid --workers 8 --report grid.json
```

The family command simulates 45 terminal branches and writes their native histories and measurements. The axes are five partner movements [negative large, negative small, hold, positive small, positive large], three first actions of B [negative, hold, positive] and the same three final action choices. All five futures begin from the same MuJoCo integration snapshot after the selected first action. Output filenames must be unused.

The complete test grid covers 32 nuisance variants in each of two historical layouts (7 and 8), crossed with four mass/friction scales (1.125/0.8, 1.375/0.8, 1.625/0.8 and 1.5/0.6). The first three pairs were withheld during domain-randomized training. The fourth uses a familiar material on fresh initial conditions. The resulting 256 physical configurations yield 11,520 terminal branches. All are retained. Add `--output grid.npz` to save observations and raw endpoints, or omit it for a compact JSON comparison. The assembled rows are keyed by `family_id`; each `paired_root` repeats across four materials.

Success requires the full peg cross-section to pass through the socket plate, insertion depth ≥25 mm, wall clearance ≥0.10 mm, relative angle <0.10 rad and all four physical pad–handle normal forces >0.01 N. These conditions are tested immediately after B’s second action and after holding for another 0.12 s. The full-grid report compares all 115,200 individual guard decisions and 11,520 terminal labels with the frozen measurements. The canonical smoke family also checks force, tactile and joint histories and the raw endpoint measurements. The reference file is first read by the comparison step after simulation has started; `family` runs without it.

| Check | Expected result |
| --- | --- |
| Canonical family | 45/45 terminal labels; 225/225 guard decisions at each endpoint |
| Whole test grid | 11,520/11,520 terminal labels; 57,600/57,600 guard decisions at each endpoint |
| Whole-grid label SHA-256 (native reference order, uint8) | `e7633d124d0cf70dfb43385dec252a8e437ea208570391b9c1efb3074be9bd61` |

### Numerical comparison across hosts

Every archived terminal label and all ten component decisions per branch must match exactly in both modes. One disagreement fails the run, including a threshold crossing caused by a small continuous error. The reported continuous comparison covers the archived **canonical family**. It checks each element against `atol + rtol × |reference element|`. Shape mismatches and nonfinite values fail.

| Canonical measurement | Unit | Absolute tolerance | Relative tolerance |
| --- | --- | ---: | ---: |
| B joint angle / angular velocity | rad / rad s⁻¹ | 10⁻⁷ in each unit | 0 |
| B wrench force / torque | N / N m | 10⁻⁷ in each unit | 0 |
| A and B hand normal-force maps | N per cell | 10⁻⁷ | 0 |
| Endpoint depth and wall clearance | mm | 10⁻⁶ | 0 |
| Endpoint relative angle | rad | 10⁻⁷ | 0 |
| Endpoint pad–handle normal force | N per pad | 10⁻⁶ | 10⁻⁶ |

The observation check uses the study's 10⁻⁷ native-unit sensor difference scale. Endpoint distance and angle errors are checked separately against the 0.10 mm clearance and 0.10 rad angle criteria. For endpoint grip forces, the 1 µN floor is 1/10,000 of the 0.01 N grip criterion. The relative term allows 1 ppm of the **individual** recorded pad's load. On a 20 N contact that gives a 21 µN total allowance. The grip decision itself still uses the exact 0.01 N threshold at each endpoint.

`--strict` on `smoke` or `test-grid` requires bit-identical canonical float arrays in addition to the exact discrete checks. Each JSON report retains the original per-array `canonical_max_abs`, the historical untyped `max_abs < 1e-6` diagnostic, per-channel limits and exceedance counts, and canonical bit equality. This lets different hosts report their actual numerical disagreement even when all physical outcomes reproduce. Progress for the full grid goes to standard error every 32 physical profiles. Standard output remains a JSON document.

On the local Linux test machine, selecting the OpenBLAS Haswell kernel reproduced a 9.5044 × 10⁻⁶ N canonical endpoint force difference with exact guard decisions and a passing unit-aware check. Selecting its Nehalem kernel retained the same guard decisions but exceeded the limits in five endpoint distance or angle values. These kernel checks used the same pinned package versions.

```sh
python simulation/replay.py smoke --strict
python -m unittest simulation.test_replay_audit
```

## Run a closed-loop policy

```python
from simulation.physics import Panda, rollout_policy
from simulation.task import family_row

row = family_row(layout=7, variant=27, mass=1.125, friction=0.8)
scene = Panda(row["layout"], row["mass_scale"], row["friction_scale"])

def first(root):
    return 1  # Leave the socket at its current target.

def second(first_action, root, middle):
    return 1

a, b, before, after = rollout_policy(scene, row, 4, first, second)
print(a, b, before["ok"] and after["ok"])
```

Replace the two hold callbacks with a policy to obtain a fresh native episode. Callbacks receive only sensor histories. The four keys are `proprio` (six samples of seven B angles plus seven rates), `w` (six samples of a B-frame wrench), `p` (six B force maps), and `partner_p` (six A force maps). The middle history contains three samples from B’s first motion followed by three during A’s maneuver. Any quantization or feature transform used in training belongs inside the policy.

## Scene and initial-state provenance

`data/scene.xml` carries the combined physical scene and relative references to upstream visual and collision assets. `layouts.json` contains the 12 fixture-geometry positions that vary across the eight historical scenes. `initial.npz` retains joint state, velocities, actuator controls and simulation clock from archived frame 740 for each layout. Material compilation scales the free peg and socket masses and inertias, plus the sliding friction in four pad–handle pairs. Peg–socket material parameters stay at their original values. These are preserved conditions from the insertion-hold phase; this module does not simulate the preceding acquisition and subsequent delivery.

File and source-archive SHA-256 values are in `data/provenance.json`. Asset provenance and upstream notices are in [Third-party assets](THIRD_PARTY.md).
