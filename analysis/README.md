# Matching a correction to the observed partner motion

Run `python analysis/reproduce.py` from the repository root. The result in `runs/alignment.json` contains the response matrices, per-root scores, paired differences and 95% bootstrap intervals. To plot the paper figure, install Matplotlib and run `python analysis/plot_alignment.py`. Both commands accept `--output`; the plotting command also accepts `--input`.

Each input contains binary outcomes for all 45 physical branches per profile (five possible pin motions × three first actions × three second actions), the two frozen policy choices and their observed success. The program verifies the recorded success for every selected trajectory. It keeps the first action fixed and applies the follow-up from each of the five possible motion histories to each actual motion. A uniformly chosen history can coincide with the correct one with probability 1/5. The fractional score is the exact expectation. The best fixed pair is an outcome-informed upper bound for a pair chosen separately per profile before partner motion. The best informed second action is an upper bound given the learned first choice.

The `opposite_large_motions` field pairs the large leftward and rightward motions at the policy’s first action. It counts conditions in which each motion can succeed but no single follow-up succeeds on both, and reports how often the policy succeeds on both sides. This is a retrospective diagnostic within the same draws and resets.

The material cohort has 64 nuisance draws on two archived grasp layouts, matched across three interpolated materials. The fresh cohort has 12 acquired grasp resets, each with six variants at nominal material. All five motions stay together. The bootstrap resamples nuisance draws within layout on the material cohort and whole resets on the fresh cohort using 10,000 draws (seed 261119). The analysis is retrospective and conditions on the frozen trained ensemble. The source archive and prediction-table fingerprints are recorded in `sources.json`.

## External-probe comparison

The [probe study summary](probe_results.json) records insertion counts, paired gains and losses, and whole-grasp confidence intervals for the external-contact experiment in Table I and Fig. 3. The two primary cohorts are `fresh` and `transfer`. The follow-up `transfer_crossover` reuses the transfer grasps at the complementary travel. The `reproduce.py` command above reanalyzes the separate partner-motion experiment.
