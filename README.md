# Contact-Guided Correction Selection After Partner Motion in Simulated Bimanual Insertion

Reproduce experiments on two-stage residual control for **bimanual insertion maintenance**. One simulated Panda holds a pin and makes one of five unannounced lateral motions. The other holds the socket and chooses a correction twice. The experiment asks whether the second choice helps because it uses contact feedback from the partner motion that actually occurred.

[Read the companion research manuscript (PDF)](paper/bimanual_contact_feedback.pdf).

## Run an experiment

**On GitHub:** Open [Run paper experiment](https://github.com/BitAdventurer/bimanual-contact-feedback/actions/workflows/experiments.yml), click **Run workflow**, select an experiment and start it on `main`. Choose `results` to reconstruct the timing comparison, verify the receiving-hand contact intervention, and score the paper's 18 locked actor checkpoints on their original test observations. Choose **`end-to-end`** to regenerate all held-out trajectories in MuJoCo, simulate the contact intervention, train six three-seed policy ensembles, and score the validation-selected models on the newly generated observations. For a quick physics check choose `native-smoke`. Numerical reports appear in the job log or its downloadable artifact. The workflow uses Linux CPU runners. The quick readout took 40 seconds on GitHub. Allow roughly half an hour for the full workflow on a two-worker hosted run; native simulation accounts for most of that time.

**Locally**, from this directory, the shortest path to the paper's timing comparison is:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r analysis/requirements.txt
python analysis/reproduce.py
python mechanism/verify.py
python training/run.py evaluate --run training/reference --json runs/paper-checkpoints.json
```

| Experiment | What is executed | Where to start |
| --- | --- | --- |
| Timing comparison | Pairs every actual motion with all five possible follow-up decisions at the same root and computes paired uncertainty | [analysis](analysis/README.md) |
| Receiving-hand contact cut | Recomputes force contrasts and runs the frozen socket policy on six-sample histories from matched native contact/cut branches. The native runner also regenerates the paired trajectories | [mechanism](mechanism/README.md) |
| Native physical replay | Advances MuJoCo from archived force-gripping states, issues both actions, and checks engagement and grip before and after settling | [simulation](simulation/README.md) |
| Training and controls | Fits fresh CPU PPO and supervised models on cached physical transitions, selects checkpoints on validation and scores a separate test split | [training](training/README.md) |

### Run the simulator or retrain locally

Reference environment: Linux x86-64, CPython 3.8.20. For the commands below, activate a virtual environment made with this version. One simulator worker and one PyTorch thread can run on CPU; the full comparisons parallelize across independent work units. No display or GPU is required.

```bash
python -m pip install -r simulation/requirements.txt
python -m pip install --no-deps robosuite==1.4.1
python simulation/replay.py smoke
mkdir -p runs
python simulation/replay.py test-grid --workers 2 --output runs/native-grid.npz
python -m mechanism.replay --workers 2
```

```bash
python -m pip install --only-binary=:all: --no-deps 'torch==2.4.1+cpu' --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r training/requirements.txt
python training/run.py smoke
python training/run.py fit --out runs/contact --workers 2 --controls
python training/run.py evaluate --run runs/contact --json runs/contact/test.json
# Score the same locked models on your new MuJoCo test trajectories:
python training/run.py evaluate --run runs/contact --native-test runs/native-grid.npz --json runs/contact/native-test.json
```

**One-command reproduction.** After installing both simulation and training dependencies above, run the following from a fresh checkout or the extracted ZIP. This uses the same entry point as GitHub's `end-to-end` option. Choose a new output directory for each run.

```bash
bash scripts/reproduce.sh runs/full 2
```

Open `runs/full/native-test.json` for the trained policy scores and `runs/full/native-contact-cut.json` for the causal contact comparison. The final argument is the number of CPU workers. Use a new output directory for a new fit. Training reads archived simulator transitions, so retraining does not require a prior 11,520-branch simulator run. The native test grid independently checks the physical outcomes used in evaluation.

## Reference results

| Cohort or intervention | Reference measurement |
| --- | --- |
| Held-out materials, 64 matched initial draws × 3 materials × 5 motions | Bilateral PPO matched to its own contact history **709/960**. Independent history **418/960** in expectation. Best fixed two-action sequence per initial draw and material **544/960**. |
| Nominal material, 12 newly acquired grasp resets × 6 nuisance variants × 5 motions | The same frozen policy scored **289/360**, compared with **161.8/360** for independent history and **205/360** for the best fixed sequence per variant. |
| Effect of randomized-material training | **709/960** vs **462/960** for nominal training with matched training size. Relative to nominal training, 335 failures became successes and 88 successes became failures. |
| Receiving-hand contact intervention | The number of opposite partner-motion comparisons in which the socket-local policy chose different follow-up actions fell from **85/96** with contact to **0/96** after the simulated pin–socket contact was disabled. |

“Independent history” is an exact average over the five possible histories, including a possible match. “Best fixed” uses all outcome labels after the fact and bounds an action pair that cannot react to partner motion. The terminal outcome measures engagement and grip over a short simulated maintenance interval. A wrist or total-force signal can carry much of the same information as the spatial maps. The fresh-reset packet supports the timing readout; the native grid replays the archived held-out material configurations. See individual directories for scope and numerical tolerances.

The timing readout uses the paper's frozen choices. Fresh training can select different actions across computers with the same Python and PyTorch versions. In a separate [GitHub CPU training run](https://github.com/BitAdventurer/bimanual-contact-feedback/actions/runs/44360999139) randomized and nominal PPO reached **716/960 and 462/960** on the same archived test paths. Switching the local OpenBLAS kernel to Haswell gave **703/960 and 462/960** on newly simulated paths. An additional Python 3.11/PyTorch 2.7 fit reached 700/960 and 462/960. These are computational replication runs on the same initial draws.

The Panda meshes are supplied by the separately installed robosuite distribution. Source fingerprints for the bundled experiment data are included with each experiment.
