"""Replicate-level aggregation of evaluation episodes (run on the execution host).

  python runpod/aggregate.py RUNS_ROOT OUT_PREFIX

Writes OUT_PREFIX_replicates.csv (one row per condition x label x replicate:
pooled PDR and packet-weighted / episode-averaged metrics, loss shares) and
OUT_PREFIX_training.csv (learner training curves: pdr, epsilon, learner cost).
Also OUT_PREFIX_index.json with run counts, DONE status and code hashes.
Condition = the directory two levels above rep_XX (see make_jobs.py layout).
"""
import csv, json, sys, collections
from pathlib import Path

root, prefix = Path(sys.argv[1]), sys.argv[2]
LOSS = ["expired", "no_route", "buffer_overflow", "source_overflow", "failed_source", "failed_queue",
        "dedup_loss", "dest_unreachable", "infeasible", "horizon", "ineligible_relay"]
reps, train_rows, index = [], [], collections.Counter()
hashes = collections.Counter()
for rp in sorted(root.rglob("records.json")):
    d = rp.parent
    man = json.loads((d / "manifest.json").read_text())
    done = (d / "DONE").exists()
    cond, label = d.parent.parent.name, d.parent.name
    index[(cond, label, done)] += 1
    hashes[man["code_sha256"]] += 1
    rec = json.loads(rp.read_text())
    ev = [r for r in rec if r["phase"] == "eval"]
    for r in rec:
        if r["phase"] == "train":
            train_rows.append([cond, label, man["replicate"], r["episode"], round(r["pdr"], 5),
                               round(r.get("epsilon") or 0, 4), None if r.get("learner_mean_cost") is None else round(r["learner_mean_cost"], 3),
                               round(r["elapsed_seconds"], 2)])
    if not ev:
        continue
    gen = sum(r["generated"] for r in ev); dl = sum(r["delivered"] for r in ev)
    def wmean(key, w="delivered"):
        num = sum((r[key] or 0) * r[w] for r in ev if r[key] is not None); den = sum(r[w] for r in ev if r[key] is not None)
        return num / den if den else None
    def mean(key):
        v = [r[key] for r in ev if r.get(key) is not None]
        return sum(v) / len(v) if v else None
    loss = collections.Counter()
    for r in ev:
        loss.update(r["losses"])
    row = dict(condition=cond, label=label, method=man["method"], scenario=man["scenario"], replicate=man["replicate"],
               done=int(done), n_eval=len(ev), nodes=man.get("eval_settings", man["settings"])["nodes"],
               generated=gen, delivered=dl, pdr=dl / gen if gen else None,
               delay=wmean("delay_mean"), delay_p95=mean("delay_p95"), delay_cens=wmean("delay_censored_mean", "generated"),
               hops=wmean("hops_mean"), attempts_per_delivered=(sum(r["attempts_total"] for r in ev) / dl) if dl else None,
               e_total_per_deliv=(sum(r["energy_total_j"] for r in ev) / dl) if dl else None,
               e_active_per_deliv=(sum(r["energy_active_j"] for r in ev) / dl) if dl else None,
               lifetime=mean("lifetime_projection_s"), load_balance=mean("load_balance_max_over_mean"),
               ctrl_overhead=(sum(r["control_bits"] for r in ev) / (dl * 4000)) if dl else None,
               info_overhead=mean("information_overhead_ratio"),
               decision_us=mean("decision_us_mean"), memory_bytes=mean("policy_memory_bytes"),
               elapsed_eval_s=sum(r["elapsed_seconds"] for r in ev), elapsed_total_s=sum(r["elapsed_seconds"] for r in rec))
    for k in LOSS:
        row["loss_" + k] = loss.get(k, 0) / gen if gen else None
    reps.append(row)
cols = list(reps[0].keys()) if reps else []
with open(prefix + "_replicates.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(cols)
    for r in reps:
        w.writerow([("" if r[c] is None else (round(r[c], 6) if isinstance(r[c], float) else r[c])) for c in cols])
with open(prefix + "_training.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["condition", "label", "replicate", "episode", "pdr", "epsilon", "learner_cost", "elapsed_s"])
    w.writerows(train_rows)
json.dump({"runs": {f"{c}|{l}|done={d}": n for (c, l, d), n in sorted(index.items())}, "code_sha256": dict(hashes)},
          open(prefix + "_index.json", "w"), indent=1)
print(len(reps), "replicate rows;", len(train_rows), "training rows;", dict(hashes))
