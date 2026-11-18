# Train the bimanual feedback policies

Arm A holds the pin and moves after arm B makes its first correction. B holds the socket and chooses a second correction from its joint history and the two hands’ contact maps. Train on the provided simulated transitions and measure whether the learned response preserves insertion. The training and evaluation commands use CPU only.

## Evaluate the paper's frozen policies

From the repository root, use a Python environment with NumPy installed. This path performs actual policy inference. The stage-one network chooses a first action, the stage-two network sees only the history reached by that action, and the evaluator scores the selected terminal branch.

```sh
python training/run.py evaluate --run training/reference --json paper-checkpoints.json
```

To score the same weights on a new simulation grid, add `--native-test runs/native-test.npz` and use a different JSON output name. The packaged [checkpoint lock](reference/selected.json) contains the model selection and hashes. Its 22 referenced files are the 18 validation-selected actor checkpoints, two training-fitted sensor transforms and two training/validation records. All 198 actor tensors matched the original selected experiment checkpoints exactly. The bundle occupies 1,513,715 bytes including the lock. It lets a reader evaluate the published policies independently of the numerical variation introduced by training again.

Frozen inference reproduced all six reported ensemble scores on the bundled data and on a newly integrated 11,520-branch grid. The latter also reproduced the 418 expected successes with an independently selected history, the 544 committed hindsight bound and 709 with the aligned history. This was checked on NumPy 1.22.4 and 1.26.4 in separate environments. An isolated NumPy-only run verified that evaluation requires neither PyTorch nor SciPy.

## Train from the repository root

The reference fit used Linux x86-64, CPython 3.8.20, NumPy 1.22.4, SciPy 1.10.1 and PyTorch 2.4.1+cpu. A separate virtual environment isolates these versions from the simulator.

```sh
python3.8 -m venv .venv-training
. .venv-training/bin/activate
python -m pip install --only-binary=:all: --no-deps --index-url https://download.pytorch.org/whl/cpu torch==2.4.1
python -m pip install -r training/requirements.txt

python training/run.py smoke
python training/run.py fit --out runs/contact --workers 3 --controls
python training/run.py evaluate --run runs/contact --json runs/contact/test.json
```

The smoke test performs two actual PPO updates per training regime, selects a checkpoint per regime, scores the resulting policies and removes its temporary outputs. The full command fits three seeds per ensemble from random weights. Omit `--controls` to train only randomized and nominal bilateral PPO. Running with one worker uses less memory. Specify a new output path for each fit. The fit command reads TRAIN and validation arrays. The evaluate command checks the selected weights and sensor transforms against their saved hashes before loading TEST.

## Score a fresh physics integration

Set up the [native simulator](../simulation/README.md) and fit the controller once as above. The first command uses the simulation environment and writes all raw observations and terminal measurements. The second uses the training environment to apply the existing encoders and model checkpoints.

```sh
.venv-sim/bin/python simulation/replay.py test-grid --workers 8 --output runs/native-test.npz --report runs/grid-checks.json
.venv-training/bin/python training/run.py evaluate --run runs/contact --native-test runs/native-test.npz --json runs/contact/native-test.json
```

The scorer matches every native family ID to the fixed 64 draws and four materials, checks material values and common first-action branches, rounds joint measurements, and derives all 11,520 rewards from the five physical guards at both endpoints. It retains draws outside the initial guards. With `--native-test`, the model reads its observation arrays and receives its outcome labels from this newly integrated grid.

An independent simulator installation regenerated all 256 rows. The 17 initially outside the guards remained in the analysis. The six learned ensembles selected identical first and second actions on all 1,280 paths per policy in the native and bundled evaluations. The randomized and nominal primary counts remained 709 and 462. This checks the same assigned draws through a second physics integration; the number of independent nuisance draws remains 64.

## Training and evaluation

Each training regime has 256 initial draws on the same four archived grasps. Randomized training varies part mass and grip friction over eight settings. Its matched nominal control uses the same draws with nominal mass and friction. Both regimes use the same 64 randomized validation draws on two other grasps for checkpoint selection. Each regime fits its own normalization and PCA on TRAIN only.

Each decision has three lateral increments: left, hold and right. Each stage encoder consumes six time samples of B's seven joint angles, seven velocities and one normal-force map from each hand. It keeps up to 16 principal components per input block. A fixed-width 64-component vector has an unused 16-component block. The second network receives the initial and realized intermediate encodings plus its own first action. It cannot access the identity of the partner motion. The optional two-load policy sums each hand's map before fitting its sensor encoder.

Two networks with 64- and 32-unit tanh layers each have policy and value heads. PPO samples the initial draw, one of five partner motions and two actions under its current policy. The transition cache returns the observation reached by the first action and the physical terminal reward reached by the second. Both stages use the terminal reward, a value baseline and batch-normalized advantages. Each seed receives 192 batches of 384 episodes, with four optimizer passes per batch, minibatches of 192, Adam step size 0.0005, PPO clip radius 0.2, value coefficient 0.5, entropy coefficient 0.01 and gradient norm cap 0.5. Each seed's 73,728 sampled episodes revisit the stored physical transitions. The supervised control regresses the three counterfactual second-action outcomes and their per-first-action best continuation averaged over hidden motions. Its label budget is larger than PPO's.

Seeds are 9201, 9202 and 9203. At updates 48, 96, 144 and 192, take the ensemble mean of the categorical probabilities, or the predicted values for supervision. Select a common update by validation success count, then fewer losses against holding, fewer movements and earlier update. The committed PPO chooses one of nine action pairs in its initial observation. The returned `training_summary.json` reports sampled reward statistics, `validation.json` gives all checkpoint scores, and `selected.json` pins the choice and files used for final evaluation.

## Reference results

The primary cohort has 64 independent nuisance draws in two fixed test grasps. Each draw is crossed with three previously untrained material combinations and five partner motions. All 960 paths enter the primary count. The 320 familiar-material paths use the same draws at a fourth material setting present in training.

| Policy | Three interpolated materials, /960 | Familiar material, /320 |
|---|---:|---:|
| Randomized bilateral PPO | 709 | 244 |
| Matched nominal bilateral PPO | 462 | 153 |
| Randomized bilateral supervised | 692 | 239 |
| Randomized joint-only PPO | 614 | 205 |
| Randomized per-hand total-force PPO | 717 | 250 |
| Randomized committed bilateral PPO | 478 | 166 |
| Hold at both stages | 460 | 154 |

The randomized ensemble gained 335 successes and lost 88 relative to matched nominal PPO. Its paired gain was 25.73 percentage points, with a 95% bootstrap interval of 22.29 to 29.37. The calculation resamples the 64 draws within the two fixed grasp strata 10,000 times, keeping the three materials and five futures together. Individual randomized seeds scored 704, 673 and 695; the three nominal seeds each scored 462.

With the randomized policy fixed, replacing the realized history by an independently selected history from the same initial draw, material and chosen first action produced an expected 418 successes. The best precommitted action pair chosen retrospectively for each draw and material could reach 544. Aligned histories gave 709, compared with an upper bound of 862 given the learned first action and optimal hindsight follow-up. The evaluation report includes the complete actual-motion by history-donor table.

The local reference fit reproduced every tensor in the 18 selected paper checkpoints. On an independent GitHub-hosted CPU using the same pinned software versions, training from scratch gave 716/960 for randomized PPO and 462/960 for matched nominal PPO. Another local fit, with Python 3.11.16, NumPy 1.26.4, SciPy 1.15.3 and PyTorch 2.7.1, gave 700/960 and 462/960. The seeds and versions do not fully determine training outcomes across numerical kernels. To evaluate the paper's particular learned policies, use `--run training/reference`. The checked results are in [verification.json](verification.json). The [data dictionary](data/README.md) documents each array and its units.
