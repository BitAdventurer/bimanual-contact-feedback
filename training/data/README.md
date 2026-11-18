# Simulation transition data

Four NumPy archives contain 832 draw–material rows and 37,440 recorded terminal leaves. Files retain the original simulation row order. Each file contains all physical outcomes for three first and three second socket movements across five hidden pin movements. Each training regimen has one material per initial draw. The randomized and nominal archives use the same nuisance draws. Both regimens are evaluated on the same randomized validation cohort.

| File | Rows | Distinct draws | Purpose |
|---|---:|---:|---|
| `train.npz` | 256 | 256 | Four grasp configurations and eight material settings, balanced across draws |
| `nominal_train.npz` | 256 | 256 | Same physical nuisance draws, nominal materials |
| `validation.npz` | 64 | 64 | Two independent grasp configurations, common randomized checkpoint selection |
| `test.npz` | 256 | 64 | Two further grasp configurations, every draw crossed with four materials |

The compressed archives total 9,656,781 bytes. The single-split SHA-256 fingerprints and hashes of their original simulation archives are recorded in [manifest.json](manifest.json). The reduction preserves the contact maps and binary outcomes exactly and packs the published joint-input rounding into signed 16-bit integers. Only the four splits used by this workflow are included. The input data rights are described in [RIGHTS.md](RIGHTS.md).

## Axes and units

The first axis indexes a draw–material row. The five partner-motion indices correspond to large left, small left, stationary, small right and large right. The three action indices for arm B correspond to left, hold and right. Those indices are used to route the physical simulation branches.

| Array | Shape after the row axis | Meaning |
|---|---|---|
| `root_joint_bins` | (6,14) | B joint bins at six initial times; multiply the first seven by 0.001 rad and the last seven by 0.01 rad/s |
| `root_a_map`, `root_b_map` | (6,8,8) | A pin-hand and B socket-hand normal forces, N per cell |
| `mid_joint_bins` | (5,3,6,14) | Intermediate B joint measurements indexed by realized partner motion and B's chosen first action |
| `mid_a_map`, `mid_b_map` | (5,3,6,8,8) | Intermediate hand-force maps at three times during the first correction and three during the partner motion |
| `success` | (5,3,3) | Physical terminal success given partner motion, first action and second action |
| `paired_root`, `layout`, `condition` | scalar | Nuisance draw, archived grasp configuration and material setting for matching or reporting |

Each hand map sums the two simulated finger-pad grids. Each pad uses a 16 mm × 16 mm square with contact-point normal forces distributed bilinearly to grid cells. These are synthesized normal loads at rigid contacts. Material IDs use thousandths of the nominal mass and hundredths of the nominal pad–handle friction. The material at `m1375_f080`, for example, has mass scale 1.375 and friction scale 0.80.

Success requires full shaft engagement, depth at least 25 mm, lateral clearance of at least 0.1 mm, angle error below 0.10 rad and load above 0.01 N on all four pads. All conditions must hold both after B's follow-up and at the end of the final hold. Hidden geometry, arm A's chosen motion, material parameters, metadata and terminal labels are excluded from actor observations. The supplied archives contain the evaluated binary leaves. For the underlying physical procedure, see the repository's simulation workflow.
