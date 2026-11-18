# Does the socket hand react to a contact transmitted through the parts?

Run this experiment from the repository root with Python 3 and NumPy installed.

```sh
python mechanism/verify.py
```

This NumPy-only readout runs the frozen socket-hand PPO policy on recorded joint and pad-force histories. It verifies every input file's SHA-256, recomputes all 576 follow-up actions and action distributions, and compares them with the original run. It also recomputes the physical contrasts below from the recorded force traces, pad maps and part positions. The command takes about a second on a CPU. It writes its result to standard output.

## Re-simulate the intervention

Install the [native simulator requirements and robosuite asset wheel](../simulation/README.md). From the repository root, run:

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m mechanism.replay --smoke
python -m mechanism.replay --workers 8
```

The first command runs one preselected root and material. The second runs all 32 pairs. A headless Linux run took 1.73 seconds for the single pair and 7.39 seconds for all pairs with eight CPU workers (MuJoCo 3.2.3, NumPy 1.22.4). Use `--workers 1` for sequential execution, or `--asset-root` for an unpacked robosuite 1.4.1 asset directory.

The runner selects the original 16 base draws by a hash of their IDs, constructs the two independent MuJoCo worlds, and regenerates the first move, partner command and sensor observations from the archived initial grasp. It restores the same complete integration state immediately before each branch. It computes joint/contact observations and inference from the newly simulated paths. Reference packets are opened for comparison after each pair’s simulations and policy predictions. Across 576 motion intervals, the reproduced 15 physics and sensor fields had zero maximum absolute difference from the source arrays on the reference setup, including all 640 time steps per contact trace. The policy reproduced all 576 archived follow-up actions. Floating-point probabilities can differ slightly across BLAS implementations; verification checks the actions exactly and uses an absolute tolerance of 10⁻⁵ for probabilities. Raw physics and sensor fields have a 2 × 10⁻⁶ absolute check in each field’s native units.

## Paired experiment

In the original MuJoCo experiment, the peg and socket were already grasped and engaged. Hand B held the socket. The same complete simulator state after hand B’s first move was restored before each of three possible moves by hand A (large left, hold, large right). Each branch was simulated with peg-to-socket contact intact and with the 18 explicit mating geometry pairs removed. The grasp pairs and each arm’s controller remained active. The two worlds thus share a fork *within* each replay. The NumPy verifier reads those recorded worlds. The native replay regenerates their trajectories. The worlds end at the observation for the second decision. No post-decision task-success label exists for the cut world.

There were 16 independent initial-condition draws across two archived layouts. Each draw was tested at two materials and three forced first actions, giving 96 comparisons between opposite motions. Selecting the first action chosen by this specific policy gives 32 comparisons. Both materials and all first actions for one draw share that draw.

| Recomputed quantity | Intact | Cut |
|---|---:|---:|
| Socket-hand follow-up differed between large left and large right | 85/96 | 0/96 |
| Same comparison after the policy’s own first move | 31/32 | 0/32 |
| Partner-motion intervals with positive peg–socket contact force | 156/288 | 0/288 |
| Median peak difference in socket-hand resultant force | 2.002 N | < 10⁻¹² N |
| Median peak difference in socket-hand 8×8 normal-force map (sum of absolute cell differences) | 0.0190 N | < 10⁻¹² N |
| Forks with distinguishable rounded socket-hand joint histories | 90/96 | 0/96 |

The force and pad-map statistics compare opposite motions at the three corresponding post-command samples, take the maximum difference over those times, and then the median over 96 forks. Some cut forks have a small numerical residual. The largest cut resultant-force difference is 8.12 × 10⁻⁶ N. All four pads retained at least 16 N in both worlds. The socket policy observes its joints and its local map together. The result establishes sensitivity to interpart mechanical coupling in these simulated grasps. Both joint and map signals can convey that coupling.

## Inputs and provenance

- `data/cases/` contains the 32 original, byte-identical simulator output archives. Each includes the shared root sensor packet, pre-motion packets, both worlds’ post-motion packets, fork IDs, the 640-step contact records and per-pad minimum loads. Frames are addressed by world (intact, cut), first socket-hand move (left, hold, right), and peg motion (large left, hold, large right). The sensor channels `p` and `partner_p` are socket-hand and peg-hand normal-force maps in newtons per cell. `proprio` is the socket-hand joint state; `w` is the socket-hand six-axis net contact wrench.
- `data/models/` contains the original TRAIN-calibrated root and intermediate encoders and all three policy checkpoints selected on validation, at update 192. Six sensor samples make up each decision window. The second window concatenates three samples after the first socket-hand move with three after the partner command. Each network receives projected, rounded B joint values and the B map, and the second network receives the first action. Other sensory slots are zero. Action probabilities are averaged across the three seeds before greedy selection.
- `data/reference.csv` contains the socket policy’s 576 selected rows from the original five-policy prediction file. It is used solely to check the newly computed predictions. `data/manifest.json` records the file hashes from the original physical results and model-selection records, the derived table’s hash, and the fingerprints of the source receipts and full prediction file. The original mechanism is an exploratory subset of an exposed simulation evaluation.

For a holder of the original study archive, the supplied extraction script rebuilds these inputs in a new directory. It checks the original evaluation receipt, case, policy-selection and predictions digests before extracting or copying data. The published packet and checkout are sufficient for ordinary verification.

```sh
python mechanism/export.py /path/to/study /tmp/new-mechanism-inputs
python mechanism/verify.py --data /tmp/new-mechanism-inputs
```
