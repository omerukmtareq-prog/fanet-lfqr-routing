# Tuning-split selection

Rule: highest mean tuning PDR over S1, S1_L02, S2, S3 x 2 replicates; ties within 0.1 pp -> lower censored delay.

## lfqr: selected `lfqr_i1_a0.02_plus_o0` (mean PDR 87.39 %, censored delay 30.19 steps)

| rank | label | mean PDR % | censored delay |
|---|---|---|---|
| 1 | lfqr_i1_a0.02_plus_o0 | 87.39 | 30.19 |
| 2 | lfqr_i1_a0.05_plus_o0 | 86.93 | 32.23 |
| 3 | lfqr_i0_a0.05_plus_o0 | 86.60 | 33.45 |
| 4 | lfqr_i1_a0.05_plus_o1 | 85.79 | 32.28 |
| 5 | lfqr_i1_a0.02_plus_o1 | 85.66 | 31.43 |
| 6 | lfqr_i0_a0.02_plus_o0 | 85.55 | 34.67 |
| 7 | lfqr_i1_a0.02_prune_o1 | 85.50 | 30.83 |
| 8 | lfqr_i1_a0.02_none_o1 | 85.41 | 31.97 |
| 9 | lfqr_i1_a0.1_prune_o1 | 85.38 | 31.53 |
| 10 | lfqr_i1_a0.1_none_o1 | 85.32 | 32.04 |

## heur: selected `heur_q0.25_e0.2_l1.0_s0.5_none` (mean PDR 88.11 %, censored delay 28.09 steps)

| rank | label | mean PDR % | censored delay |
|---|---|---|---|
| 1 | heur_q0.25_e0.2_l1.0_s0.5_none | 88.11 | 28.09 |
| 2 | heur_q0.25_e0.0_l1.0_s0.5_none | 88.03 | 28.10 |
| 3 | heur_q0.5_e0.2_l1.0_s0.5_none | 87.84 | 27.95 |
| 4 | heur_q1.0_e0.2_l1.0_s0.5_none | 87.80 | 27.90 |
| 5 | heur_q0.5_e0.0_l1.0_s0.5_none | 87.80 | 28.22 |
| 6 | heur_q1.0_e0.0_l1.0_s0.5_none | 87.66 | 28.27 |
| 7 | heur_q0.25_e0.0_l1.0_s0.2_none | 87.26 | 28.80 |
| 8 | heur_q0.25_e0.2_l1.0_s0.2_none | 87.08 | 28.64 |
| 9 | heur_q0.5_e0.0_l1.0_s0.2_none | 87.04 | 28.78 |
| 10 | heur_q0.5_e0.2_l1.0_s0.2_none | 86.90 | 28.75 |

## heur_plus: selected `heur_q0.5_e0.2_l1.0_s0.5_plus` (mean PDR 90.01 %, censored delay 27.41 steps)

| rank | label | mean PDR % | censored delay |
|---|---|---|---|
| 1 | heur_q0.5_e0.2_l1.0_s0.5_plus | 90.01 | 27.41 |
| 2 | heur_q0.25_e0.0_l1.0_s0.5_plus | 89.80 | 27.41 |
| 3 | heur_q0.5_e0.0_l1.0_s0.5_plus | 89.64 | 27.89 |
| 4 | heur_q0.25_e0.2_l1.0_s0.5_plus | 89.53 | 28.08 |
| 5 | heur_q1.0_e0.2_l1.0_s0.5_plus | 89.08 | 28.61 |
| 6 | heur_q0.25_e0.0_l1.0_s0.2_plus | 88.97 | 28.35 |
| 7 | heur_q1.0_e0.0_l1.0_s0.5_plus | 88.95 | 28.23 |
| 8 | heur_q0.5_e0.0_l1.0_s0.2_plus | 88.89 | 28.73 |
| 9 | heur_q1.0_e0.0_l1.0_s0.2_plus | 88.75 | 29.11 |
| 10 | heur_q0.5_e0.2_l1.0_s0.2_plus | 88.71 | 29.02 |

Reference `qmr`: mean PDR 82.61 %, censored delay 34.70
Reference `gpsr`: mean PDR 66.73 %, censored delay 54.75
Reference `geo_greedy`: mean PDR 69.77 %, censored delay 51.60
Reference `random_quadrant`: mean PDR 32.68 %, censored delay 78.13
Reference `lfa_q_pilot01`: mean PDR 82.74 %, censored delay 37.24
