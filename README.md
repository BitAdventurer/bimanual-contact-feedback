# When Spatial Tactile Feedback Improves Follow-Up Corrections in Bimanual Insertion

Chang-Hoon Ji and Tae-Eui Kam · Department of Artificial Intelligence, Korea University, Seoul, Republic of Korea

[Paper (5-page PDF)](paper/bimanual_contact_feedback.pdf) · [Supplement](paper/socket_map_supplement.pdf) · [Matched replays](#matched-replays) · [Reproduce the partner-motion results](#reproduce-the-results)

We study how the arm holding a socket chooses its next correction after contact changes. Two simulated tests restore the same grasp to compare the corrections that would keep a pin inserted. In one, the partner arm moves. In the other, an external probe presses opposite sides of the pin while both arms receive identical commands.

![In one simulated grasp the socket moves right, the pin moves left, and the socket policy uses both hands’ pad-force maps and its joint history to select a leftward follow-up.](paper/media/overview.png)

## Partner motion

Two Panda arms hold a pin and a socket in MuJoCo. The trained policies and video replays in this repository use the following partner-motion protocol. [Experimental setup](simulation/README.md#experimental-setup)

- **First correction.** The socket arm shifts its servo reference left or right, or holds. The pin arm then makes one of five unannounced lateral motions, including a stationary command.
- **Follow-up.** After the pin command, a residual policy trained with proximal policy optimization (PPO) chooses left, hold or right. It reads both hands’ 8 × 8 pad-force maps and the socket arm’s joint history.
- **Matched comparison.** Restoring the same simulated grasp lets us try all nine socket-action pairs for each pin motion. We score engagement, depth, clearance, angle and grip at 0.88 s and again at 1.00 s.

### Partner-motion results

The socket policy maintained or recovered insertion on **709 of 960** material-test paths. Even with hindsight, the best pair of actions fixed before the pin moved succeeded on **544**. The fixed pair can vary across initial conditions and materials but must handle all five pin motions at that condition.

| How the socket arm chooses its follow-up | Held-out materials · 960 paths | New grasps · 360 paths |
| --- | ---: | ---: |
| Hold after the learned first action | 474 (49.4%) | 188 (52.2%) |
| Best action pair fixed before pin motion | 544 (56.7%) | 205 (56.9%) |
| **PPO with feedback from the actual motion** | **709 (73.9%)** | **289 (80.3%)** |
| Paired gain over the best fixed pair, percentage points [95% interval] | +17.19 [14.79, 19.48] | +23.33 [20.83, 26.11] |

For opposite large pin motions after the same first correction, no common follow-up could succeed on both sides in 124 of 192 material-test conditions and 66 of 72 new-grasp conditions, even though each side admitted a successful action. The policy succeeded on both sides in 41 and 31 of those conditions, respectively. This descriptive comparison uses the same frozen test paths.

The material test uses 64 initial draws from two held-out grasps, each tested under three material settings and five motions. The second test uses 12 newly acquired grasps, each with six variants and five motions at nominal material. Intervals resample whole initial draws within each material-test grasp or whole new grasps. [Paired analysis and data](analysis/README.md)

<p align="center"><img src="paper/media/contact-results.png" width="570" alt="Paper figure comparing fixed and feedback-guided insertion, changes in the socket-local decision when contact is cut, and insertion when socket observations are held or replaced."></p>

**What passes through contact?** In a separate test, a policy sensing only the socket hand chose different follow-ups for opposite pin motions in **85/96** comparisons with part contact intact and **0/96** after contact was disabled. Both hands kept their grasps. Panel (b) counts action changes. Panels (a) and (c) count insertions with contact intact. [Contact experiment](mechanism/README.md)

**Socket-map sensitivity (supplement).** Shuffling socket pressure locations at fixed total load reduced material-test successes from 709 to 609. Replacing the spatial distribution with that of the same grasp under a stationary pin gave 699. A separately trained wrist-wrench policy reached 722/960. The simulated rigid pads put 99.97% of the training socket load on four corner cells on average. These interventions test sensitivity to rearranging that input. [Supplementary Figure S1 and paired results](paper/socket_map_supplement.pdf)

## When spatial tactile feedback helps

With both arm commands held fixed across trials, a probe pressed opposite sides of the pin. The socket arm completed the same unloading motion and selected one of five lateral corrections. We compared its six-axis wrist wrench and both hands’ total loads with the same inputs plus an 8 × 8 map of load on each hand.

| Insertions retained | 3 mm probe travel | 2.5 or 3.5 mm probe travel |
| --- | ---: | ---: |
| Residual PPO from scratch, wrench and total loads | 511/576 | 510/576 |
| Residual PPO from scratch, with spatial maps | 548/576 | 554/576 |
| Supervised on all five action outcomes, wrench and total loads | 548/576 | 554/576 |
| Supervised on all five action outcomes, with spatial maps | 548/576 | 554/576 |
| Best available action for each state | 548/576 | 554/576 |

Each column contains 32 independent held-out grasps. Adding maps to PPO recovered 37 and 44 insertions, respectively, while preserving every previous success. In 27 and 28 of those recoveries, the map policy omitted an extra +3 mm lateral move that had lost insertion. With all five action outcomes available for supervision, the wrist wrench and total loads already sufficed at every state where one of the five candidates could succeed. [Paired estimates and grasp-level intervals](analysis/probe_results.json)

<p align="center"><img src="paper/media/probe-learning.png" width="510" alt="Fig. 3. Adding spatial maps increased retention for PPO trained from scratch by 6.42 and 7.64 percentage points in the two cohorts. Supervision on all five candidate outcomes reached the same ceiling with either sensor input."></p>

## Matched replays

All three replays below share the grasp, the material and the first socket correction. Each is the first released material-test path with that outcome. Each film shows the same trial side by side. On the left the socket holds its reference after the pin command. On the right the policy chooses its follow-up. Both views use the same fixed close-up. The one-second simulation plays at quarter speed so the millimetre-scale corrections are visible. Click a preview for its MP4, or [watch both arms acquire the parts](#grasp-acquisition) for a wider view.

| Episode | Pin command | Holding the socket reference | Learned follow-up | Material-test paths |
| --- | --- | --- | --- | ---: |
| [Recovery](#recovery) | Large left | Fails | Left · succeeds | 235 |
| [Success preserved](#success-preserved) | Stationary | Succeeds | Hold · succeeds | 474 |
| [Missed recovery](#missed-recovery) | Small left | Fails | Hold · fails | 251 |

The learned follow-up preserved all 474 material-test successes under hold. On the 360 new-grasp paths, it recovered 101 hold failures, preserved 188 successes, and shared 71 failures with hold. [Replay verification](scripts/verify_gallery.py)

### Recovery

<p align="center"><a href="paper/media/follow-up-comparison.mp4"><img src="paper/media/follow-up-comparison.gif" width="760" alt="Play the two synchronized branches. Holding loses insertion and the learned leftward correction recovers it."></a></p>

The pin moves 4.80 mm left after the socket’s first correction. Holding leaves **−0.057 mm** of clearance at the settled check. PPO shifts the socket 4.74 mm left and finishes with **+1.769 mm**. The task requires at least 0.10 mm at both checks. [Play MP4](paper/media/follow-up-comparison.mp4) · [Branch measurements](paper/media/follow-up-comparison.json)

<details>
<summary>View decision and scoring frames</summary>

<p align="center"><img src="paper/media/recovery-frames.png" width="760" alt="The two branches at the shared decision and both insertion checks. Holding ends at minus 0.057 mm clearance. Shifting left ends at plus 1.769 mm."></p>

The top row holds the reference and the bottom row applies the learned follow-up. Times are in simulation seconds.

</details>

### Success preserved

<p align="center"><a href="paper/media/preserved-success.mp4"><img src="paper/media/preserved-success.gif" width="760" alt="Both socket responses hold with a stationary pin command and preserve insertion."></a></p>

The pin receives a stationary command. The policy also chooses to hold, so the two views coincide and both branches pass. [Play MP4](paper/media/preserved-success.mp4) · [Measurements](paper/media/preserved-success.json)

### Missed recovery

<p align="center"><a href="paper/media/shared-failure.mp4"><img src="paper/media/shared-failure.gif" width="760" alt="Holding fails in both panels after a small leftward pin command."></a></p>

The pin moves 1.97 mm left. The policy holds, just as the reference does, and both branches finish with 0.063 mm of clearance. In a separate replay from the same state, a leftward correction reaches 1.196 mm and passes both checks. [Play MP4](paper/media/shared-failure.mp4) · [Measurements for all three actions](paper/media/shared-failure.json)

### Grasp acquisition

<p align="center"><a href="paper/media/grasp-acquisition.mp4"><img src="paper/media/grasp-acquisition.gif" width="760" alt="The two arms approach, grasp and engage the parts during setup of a new-grasp trial."></a></p>

Both arms approach and grasp the parts before bringing them together. This 6.5-second clip shows the scripted setup of a separate new-grasp trial at 4.8× speed. [Play MP4](paper/media/grasp-acquisition.mp4)

### Contact signals

What could the socket hand sense before choosing? The figure follows the first released grasp–material file in the contact-cut experiment, using the socket-local policy’s own first action (hold). The same pin-left and pin-right commands were replayed with contact intact and disabled. This is a separate grasp from the videos above.

<p align="center"><img src="paper/media/contact-traces.png" width="760" alt="Opposite pin commands separate in transmitted contact force and socket-hand pad readings with contact intact. With contact disabled the transmitted force stays at zero and both commands produce a hold decision."></p>

The upper traces show pin–socket normal force over 320 ms. The lower dots show the difference between the left- and right-command socket pressure maps at the three observation times. With contact intact, the rightward command produced a 2.88 N peak and a rightward follow-up. With contact disabled, both commands produced a hold. The socket-local policy also observes joint motion. [Plot from the recorded signals](mechanism/plot_traces.py) · [Experimental protocol](mechanism/README.md)

<details>
<summary>View the experimental timeline</summary>

![The first socket correction, unannounced pin motion and follow-up in a one-second trial. The contact-cut test preserves both finger grasps.](paper/media/feedback-protocol.png)

</details>

## Repository layout

| Directory | Contents |
| --- | --- |
| [analysis](analysis/README.md) | Frozen policy choices, branch outcomes and paired uncertainty estimates. |
| [mechanism](mechanism/README.md) | Contact-intact and contact-cut force traces, socket-local policy and native replay. |
| [simulation](simulation/README.md) | MuJoCo scene, initial grasps, physical branches and insertion criteria. Panda meshes come from robosuite. |
| [training](training/README.md) | Training transitions, policy controls, frozen checkpoints and validation-based model selection. |
| [paper](paper/) | Manuscript, figures, replay videos and measurements. |

## Reproduce the results

These commands reproduce the partner-motion analyses and evaluate the released checkpoints. The external-probe comparison is available in the [paired results file](analysis/probe_results.json). From the repository root, run the following with Python and NumPy.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r analysis/requirements.txt
python analysis/reproduce.py
python mechanism/verify.py
python training/run.py evaluate --run training/reference --json runs/paper-checkpoints.json
python scripts/verify_gallery.py
```

To regenerate the contact signal figure, install Matplotlib in the same environment and run `python -m mechanism.plot_traces`. The timing readout also scores reassigned histories. Pairing an independent motion history with the same first action yields 418/960 and 161.8/360 expected successes, averaging all five possible histories including a match. [Timing analysis](analysis/README.md)

## Run an experiment

On GitHub, open [Run paper experiment](https://github.com/BitAdventurer/bimanual-contact-feedback/actions/workflows/experiments.yml), select **Run workflow** on `main`, and choose a run.

| Workflow option | What it runs |
| --- | --- |
| `results` | Recompute the timing comparison, contact intervention and scores for the paper’s 18 locked actor checkpoints. A hosted CPU run took about 40 seconds. |
| `native-smoke` | Replay 45 physical branches in MuJoCo. |
| `end-to-end` | Regenerate the held-out material trajectories and contact intervention, retrain six three-seed ensembles on archived training transitions, and score the selected models on the regenerated test trajectories. Allow roughly half an hour with two hosted CPU workers. |

Results appear in the job log and downloadable workflow artifact.

<details>
<summary>Run the physics simulator and retrain locally</summary>

The reference environment is Linux x86-64 with CPython 3.8.20. Use a virtual environment created with that interpreter. These headless experiments run on CPU.

```bash
python -m pip install -r simulation/requirements.txt
python -m pip install --no-deps robosuite==1.4.1
python -m pip install --only-binary=:all: --no-deps 'torch==2.4.1+cpu' --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r training/requirements.txt
bash scripts/reproduce.sh runs/full 2
```

The last argument sets the CPU worker count. Choose a new output directory for each full run. See `runs/full/native-test.json` for the trained policy scores and `runs/full/native-contact-cut.json` for the contact comparison. The individual steps can also be run separately.

```bash
python simulation/replay.py smoke
mkdir -p runs
python simulation/replay.py test-grid --workers 2 --output runs/native-grid.npz
python -m mechanism.replay --workers 2
python training/run.py smoke
python training/run.py fit --out runs/contact --workers 2 --controls
python training/run.py evaluate --run runs/contact --json runs/contact/test.json
python training/run.py evaluate --run runs/contact --native-test runs/native-grid.npz --json runs/contact/native-test.json
```

</details>

### Training replication

Material-randomized PPO scored 709/960 against 462/960 for nominal-only training at equal training size, with 335 gains and 88 losses on matched paths. Fresh fits can vary across CPU implementations. A [GitHub CPU training run](https://github.com/BitAdventurer/bimanual-contact-feedback/actions/runs/44360999139) scored 716/960 for randomized training and 462/960 for nominal training. A local Haswell-kernel run scored 703/960 and 462/960 on regenerated paths. A Python 3.11/PyTorch 2.7 fit scored 700/960 and 462/960. These replications share the same test draws.

## Citation

Chang-Hoon Ji and Tae-Eui Kam. *When Spatial Tactile Feedback Improves Follow-Up Corrections in Bimanual Insertion*. Korea University. [Manuscript](paper/bimanual_contact_feedback.pdf) · [Citation metadata](CITATION.cff) · [Manuscript references (BibTeX)](paper/references.bib)
