"""Selection rule for the tuning phase (declared before the final split is used).

  python runpod/select_configs.py TUNE_REPLICATES_CSV OUT_DIR

For each family the configuration with the highest tuning-split PDR averaged
over the four reference scenarios (S1, S1_L02, S2, S3) and both tuning
replicates is selected; ties within 0.1 percentage points are broken by the
lower mean censored delay, then by the lexicographically smaller label.
Families: lfqr (labels lfqr_*), heur (heur_*_none), heur_plus (heur_*_plus).
Writes OUT_DIR/selected.json (settings dicts) and OUT_DIR/SELECTION.md.
"""
import csv, json, os, sys, collections

src, out = sys.argv[1], sys.argv[2]
rows = list(csv.DictReader(open(src)))
agg = collections.defaultdict(list); cens = collections.defaultdict(list)
for r in rows:
    if r["condition"] in ("S1", "S1_L02", "S2", "S3") and r["pdr"]:
        agg[r["label"]].append(float(r["pdr"])); cens[r["label"]].append(float(r["delay_cens"]))
score = {k: (sum(v) / len(v), sum(cens[k]) / len(cens[k]), len(v)) for k, v in agg.items()}
cfgdir = os.path.join(os.path.dirname(src), "_configs") if os.path.isdir(os.path.join(os.path.dirname(src), "_configs")) else None


def pick(pred):
    cand = sorted((k for k in score if pred(k) and score[k][2] == 8), key=lambda k: (-score[k][0], score[k][1], k))
    if not cand:
        raise SystemExit("no complete candidates")
    best = cand[0]
    close = [k for k in cand if score[best][0] - score[k][0] <= 0.001]
    best = sorted(close, key=lambda k: (score[k][1], k))[0]
    return best, cand


def settings_of(label, cfg_root):
    with open(os.path.join(cfg_root, label + ".json")) as f:
        return json.load(f)


cfg_root = sys.argv[3] if len(sys.argv) > 3 else None
res, md = {}, ["# Tuning-split selection", "", "Rule: highest mean tuning PDR over S1, S1_L02, S2, S3 x 2 replicates; ties within 0.1 pp -> lower censored delay.", ""]
for fam, pred in (("lfqr", lambda k: k.startswith("lfqr_")), ("heur", lambda k: k.startswith("heur_") and k.endswith("_none")),
                  ("heur_plus", lambda k: k.startswith("heur_") and k.endswith("_plus"))):
    best, cand = pick(pred)
    res[fam] = settings_of(best, cfg_root) if cfg_root else {"label": best}
    res[fam + "_label"] = best
    md.append(f"## {fam}: selected `{best}` (mean PDR {100 * score[best][0]:.2f} %, censored delay {score[best][1]:.2f} steps)")
    md.append("")
    md.append("| rank | label | mean PDR % | censored delay |")
    md.append("|---|---|---|---|")
    for i, k in enumerate(cand[:10]):
        md.append(f"| {i + 1} | {k} | {100 * score[k][0]:.2f} | {score[k][1]:.2f} |")
    md.append("")
for k in ("qmr", "gpsr", "geo_greedy", "random_quadrant", "lfa_q_pilot01"):
    if k in score:
        md.append(f"Reference `{k}`: mean PDR {100 * score[k][0]:.2f} %, censored delay {score[k][1]:.2f}")
os.makedirs(out, exist_ok=True)
json.dump({k: v for k, v in res.items() if not k.endswith("_label")} | {"labels": {k: v for k, v in res.items() if k.endswith("_label")}},
          open(os.path.join(out, "selected.json"), "w"), indent=1, sort_keys=True)
open(os.path.join(out, "SELECTION.md"), "w").write("\n".join(md) + "\n")
print("\n".join(md))
