# fanet_redesign — redesigned FANET routing simulator (Omar UKM, redesign 2026-09-13)

Release line: pilot03 (package version 0.3.0-pilot03). Pure Python 3.10+, numpy
only. No installation step: run from the directory containing `fanet_redesign/`.

```
python -m fanet_redesign test                       # correctness suite (26 tests, ≈ 12 s)
python -m fanet_redesign smoke --out smoke_out      # every method end to end, 12 nodes, with resume check
python -m fanet_redesign hash                       # SHA-256 of the package sources
python -m fanet_redesign schema                     # checkpoint schema
python -m fanet_redesign wire --nodes 100           # published wire-format bit budget

# train (learners) + frozen evaluation; rerun the same command to resume
python -m fanet_redesign run --method lfa_q --scenario S1 --replicate 0 \
    --train 5 --eval 2 --eval-split tune --out runs/S1/lfa_q/rep_00
python -m fanet_redesign run --method geo_greedy --scenario S1 --replicate 0 \
    --eval 2 --eval-split tune --out runs/S1/geo_greedy/rep_00
# design variants are config files; --label names the variant in summaries
python -m fanet_redesign run --method lfa_q --scenario S1 --replicate 0 --train 5 --eval 2 \
    --config configs/variants/lfa_q_shared_hinit.json --label lfa_q_shared_hinit \
    --out runs/S1/lfa_q_shared_hinit/rep_00
# numeric overrides: --set load=0.02 --set alpha=0.1 (values are JSON; strings need
# JSON quotes, so prefer a --config file for string settings)
python -m fanet_redesign summarize --root runs      # per (scenario, label): replicate-level PDR and means
```

Methods: geo_greedy, geo_greedy_fresh, random_quadrant, cost_heuristic,
tabular_q_hist, lfa_q (the proposed learner; LFQR = lfa_q with the selected
configuration), gpsr (greedy + perimeter recovery, Karp & Kung 2000), qmr
(Liu et al. 2020, transcribed from the authors' qmr.c). See design/PILOT03_CHANGES.md.
Protocol options usable by any method: queue_policy (fifo|slack),
prune_stale_dest, drop_infeasible. Transfer evaluation: --eval-set key=json. lfa_q variants
(settings `weight_sharing` = per_node | shared, `init` = zero | heuristic) are
candidate designs for the tuning phase; the defaults are the pilot01 design.

Scenarios: S1 (load 0.01), S1_L02 (load 0.02, prespecified calibration
condition), S2 (S1 + 20 % node failure), S3 (load 0.03), S1_hist_load and
S3_hist_load (historical loads, retained as stress tests).

Trace keys: split × scenario × (replicate · 1000 + episode). Training and
evaluation traces are distinct per replicate and paired across methods within
a replicate (`config.trace_key`, `run.job_plan`).

Run directory: manifest.json (settings, plan, code hash, label — checked on
resume), checkpoint.pkl (schema 1, episode granularity), records.json
(METRICS_SCHEMA.md, schema 2), DONE. All writes are atomic (temp file +
rename); a stray `*.tmp` is ignored.

Parallel execution (any Linux host, e.g. a RunPod CPU pod): see
`runpod/README_RUNPOD.md`. `runpod/make_jobs.py <phase> <out>` writes the
declared job list; `runpod/run_jobs.sh <jobs> <parallel>` runs it with one
process per core and is safe to re-run (resume).

Resource expectations (cloud host, single core, pilot02): 100-node S1 episode
≈ 2.5–3.5 s for non-learning methods, ≈ 5–8 s for exploring learners; memory
< 200 MB per process. Higher loads are slower (more frames per step).
