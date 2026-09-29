"""Declared job matrices (one command per line, no quotes, safe for xargs).

  python runpod/make_jobs.py PHASE OUT_ROOT [--selected DIR] > jobs.txt

Phases
  phase0      pilot02 pilot (kept for provenance)
  tune        pilot03 tuning on the TUNING split (LFQR grid, heuristic grid, QMR/GPSR sanity)
  final       locked FINAL split, 10 replicates, reference scenarios (needs --selected)
  ablation    FINAL split, 5 replicates, one-component-removed variants of the selected LFQR
  sensitivity FINAL split, 5 replicates, alternative operating conditions (base S1_L02)
  scaling     FINAL split, 3 replicates, N = 100..1000 at fixed density
  convergence TUNING split, 3 replicates, LFQR with 80 training traces
Generated variant configs are written under OUT_ROOT/_configs/ and referenced by path,
so every job line is self-contained. Output layout: OUT_ROOT/<condition>/<label>/rep_XX.
"""
import itertools, json, math, os, sys

BASE_NL = ["geo_greedy", "gpsr", "random_quadrant"]


def cfgfile(out, name, d):
    path = os.path.join(out, "_configs", name + ".json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(d, f, sort_keys=True)
    return path


def cmd(method, sc, rep, train, ev, split, out, label, cond=None, config=None, sets=(), eval_sets=()):
    c = (f"python -m fanet_redesign run --method {method} --scenario {sc} --replicate {rep} "
         f"--train {train} --eval {ev} --eval-split {split} --quiet --label {label}")
    if config:
        c += f" --config {config}"
    for s_ in sets:
        c += f" --set {s_}"
    for s_ in eval_sets:
        c += f" --eval-set {s_}"
    c += f" --out {out}/{cond or sc}/{label}/rep_{rep:02d}"
    return c


OPTS = {"none": {}, "prune": {"prune_stale_dest": True},
        "plus": {"prune_stale_dest": True, "queue_policy": "slack", "drop_infeasible": True}}
LFQR_BASE = {"weight_sharing": "shared", "init": "heuristic"}
REF_SCEN = ("S1", "S1_L02", "S2", "S3")


def tune(out):
    jobs = []
    for inter, alpha, opt, online in itertools.product((False, True), (0.02, 0.05, 0.1), OPTS, (False, True)):
        lab = f"lfqr_i{int(inter)}_a{alpha}_{opt}_o{int(online)}"
        d = dict(LFQR_BASE, feature_interactions=inter, alpha=alpha, online_adaptation=online, **OPTS[opt])
        p = cfgfile(out, lab, d)
        for sc in REF_SCEN:
            for rep in (0, 1):
                jobs.append(cmd("lfa_q", sc, rep, 20, 2, "tune", out, lab, config=p))
    for wq, we, wl, ws, opt in itertools.product((0.25, 0.5, 1.0), (0.0, 0.2), (0.25, 0.5, 1.0), (0.0, 0.2, 0.5), ("none", "plus")):
        lab = f"heur_q{wq}_e{we}_l{wl}_s{ws}_{opt}"
        d = dict(w_queue=wq, w_energy=we, w_link=wl, w_stability=ws, **OPTS[opt])
        p = cfgfile(out, lab, d)
        for sc in REF_SCEN:
            for rep in (0, 1):
                jobs.append(cmd("cost_heuristic", sc, rep, 0, 2, "tune", out, lab, config=p))
    ref = cfgfile(out, "lfa_q_pilot01", {})
    for sc in REF_SCEN:
        for rep in (0, 1):
            jobs.append(cmd("lfa_q", sc, rep, 20, 2, "tune", out, "lfa_q_pilot01", config=ref))
            jobs.append(cmd("qmr", sc, rep, 20, 2, "tune", out, "qmr"))
            for m in BASE_NL:
                jobs.append(cmd(m, sc, rep, 0, 2, "tune", out, m))
    jobs.sort(key=lambda c: ("--train 20" not in c,))
    return jobs


def _selected(sel):
    with open(os.path.join(sel, "selected.json")) as f:
        return json.load(f)


def final(out, sel):
    S = _selected(sel)
    lf = cfgfile(out, "lfqr", S["lfqr"]); hp = cfgfile(out, "heur_plus", S["heur_plus"]); hh = cfgfile(out, "cost_heuristic", S["heur"])
    jobs = []
    for sc in REF_SCEN:
        for rep in range(10):
            jobs.append(cmd("lfa_q", sc, rep, 40, 5, "final", out, "lfqr", config=lf))
            jobs.append(cmd("qmr", sc, rep, 40, 5, "final", out, "qmr"))
            jobs.append(cmd("tabular_q_hist", sc, rep, 40, 5, "final", out, "tabular_q_hist"))
            jobs.append(cmd("cost_heuristic", sc, rep, 0, 5, "final", out, "cost_heuristic", config=hh))
            jobs.append(cmd("cost_heuristic", sc, rep, 0, 5, "final", out, "heur_plus", config=hp))
            for m in BASE_NL:
                jobs.append(cmd(m, sc, rep, 0, 5, "final", out, m))
    for rep in range(3):   # historical-load stress test
        sc = "S1_hist_load"
        jobs.append(cmd("lfa_q", sc, rep, 20, 2, "final", out, "lfqr", config=lf))
        jobs.append(cmd("qmr", sc, rep, 20, 2, "final", out, "qmr"))
        jobs.append(cmd("cost_heuristic", sc, rep, 0, 2, "final", out, "cost_heuristic", config=hh))
        for m in ("geo_greedy", "gpsr"):
            jobs.append(cmd(m, sc, rep, 0, 2, "final", out, m))
    jobs.sort(key=lambda c: ("--train 40" not in c, "S1_hist_load" not in c))
    return jobs


def ablation(out, sel):
    S = _selected(sel)["lfqr"]
    variants = {
        "no_prune": {"prune_stale_dest": False},
        "fifo_queue": {"queue_policy": "fifo", "drop_infeasible": False},
        "no_interactions": {"feature_interactions": False} if S.get("feature_interactions") else {"feature_interactions": True},
        "per_node_weights": {"weight_sharing": "per_node"},
        "zero_init": {"init": "zero"},
        "no_drop_notification": {"use_drop_notification": False},
        "no_bootstrap": {"use_bootstrap": False},
        "geometry_features": {"feature_mask": "geometry"},
        "no_queue_feature": {"feature_mask": "no_queue"},
        "no_link_features": {"feature_mask": "no_link"},
        "toggle_online": {"online_adaptation": not S.get("online_adaptation", False)},
        "oracle_location": {"location_service": "oracle"},
    }
    jobs = []
    for name, ch in variants.items():
        p = cfgfile(out, "abl_" + name, dict(S, **ch))
        for sc in REF_SCEN:
            for rep in range(5):
                jobs.append(cmd("lfa_q", sc, rep, 40, 5, "final", out, "abl_" + name, config=p))
    return jobs


def tx_power_for_range(r):
    # nominal range = 10 ** ((P - 40 + 95 - 10) / 22)  (config.Settings.nominal_range)
    return round(22 * math.log10(r) - 45, 3)


SENS = {
    "nodes_50": {"nodes": 50}, "nodes_200": {"nodes": 200},
    "area_500": {"area": 500.0}, "area_1000": {"area": 1000.0}, "area_2500": {"area": 2500.0},
    "speed_x0.5": {"speed_scale": 0.5}, "speed_x2": {"speed_scale": 2.0},
    "load_0.005": {"load": 0.005}, "load_0.04": {"load": 0.04},
    "range_200m": {"tx_power_dbm": tx_power_for_range(200)}, "range_450m": {"tx_power_dbm": tx_power_for_range(450)},
    "buffer_10": {"buffer": 10}, "buffer_100": {"buffer": 100},
    "fail_0.1": {"failure_ratio": 0.1}, "fail_0.2": {"failure_ratio": 0.2}, "fail_0.4": {"failure_ratio": 0.4},
    "fail_spatial": {"failure_ratio": 0.2, "failure_pattern": "spatial"},
    "fail_early": {"failure_ratio": 0.2, "failure_step": 75}, "fail_late": {"failure_ratio": 0.2, "failure_step": 225},
}


def sensitivity(out, sel):
    S = _selected(sel)
    jobs = []
    for cond, ch in SENS.items():
        lf = cfgfile(out, f"lfqr__{cond}", dict(S["lfqr"], **ch))
        hh = cfgfile(out, f"heur__{cond}", dict(S["heur"], **ch))
        base = cfgfile(out, f"base__{cond}", dict(ch))
        for rep in range(5):
            jobs.append(cmd("lfa_q", "S1_L02", rep, 20, 3, "final", out, "lfqr", cond, config=lf))
            jobs.append(cmd("qmr", "S1_L02", rep, 20, 3, "final", out, "qmr", cond, config=base))
            jobs.append(cmd("cost_heuristic", "S1_L02", rep, 0, 3, "final", out, "cost_heuristic", cond, config=hh))
            for m in ("geo_greedy", "gpsr"):
                jobs.append(cmd(m, "S1_L02", rep, 0, 3, "final", out, m, cond, config=base))
    jobs.sort(key=lambda c: ("--train 20" not in c,))
    return jobs


def scaling(out, sel):
    S = _selected(sel)
    jobs = []
    for n in (100, 250, 500, 1000):
        area = round(1500.0 * math.sqrt(n / 100.0), 1)
        cond = f"N{n}"
        ev = [f"nodes={n}", f"area={area}", "steps=100"]
        lf = cfgfile(out, "lfqr_S1", S["lfqr"])
        hh = cfgfile(out, f"heur_N{n}", dict(S["heur"], nodes=n, area=area, steps=100))
        base = cfgfile(out, f"base_N{n}", dict(nodes=n, area=area, steps=100))
        for rep in range(3):
            # LFQR: shared weights trained at N=100 (S1) and transferred frozen to N nodes
            jobs.append(cmd("lfa_q", "S1", rep, 20, 2, "final", out, "lfqr_transfer", cond, config=lf, eval_sets=ev))
            jobs.append(cmd("qmr", "S1", rep, 5, 2, "final", out, "qmr", cond, config=base))
            jobs.append(cmd("cost_heuristic", "S1", rep, 0, 2, "final", out, "cost_heuristic", cond, config=hh))
            for m in ("geo_greedy", "gpsr"):
                jobs.append(cmd(m, "S1", rep, 0, 2, "final", out, m, cond, config=base))
    jobs.sort(key=lambda c: ("N1000" not in c, "--train 20" not in c))
    return jobs


def convergence(out, sel):
    S = _selected(sel)
    lf = cfgfile(out, "lfqr_T80", S["lfqr"])
    return [cmd("lfa_q", sc, rep, 80, 2, "tune", out, "lfqr_T80", config=lf) for sc in REF_SCEN for rep in range(3)]


def phase0(out):
    from itertools import product  # kept for provenance of r2026.09.21-pilot02
    V = {"lfa_q_shared": {"weight_sharing": "shared"}, "lfa_q_hinit": {"init": "heuristic"},
         "lfa_q_shared_hinit": {"weight_sharing": "shared", "init": "heuristic"}}
    jobs = []
    for sc, rep in product(("S1", "S1_L02", "S2", "S3"), (0, 1)):
        for m in ("geo_greedy", "geo_greedy_fresh", "random_quadrant", "cost_heuristic"):
            jobs.append(cmd(m, sc, rep, 0, 2, "tune", out, m))
        for m in ("tabular_q_hist", "lfa_q"):
            jobs.append(cmd(m, sc, rep, 5, 2, "tune", out, m))
        for lab, d in V.items():
            jobs.append(cmd("lfa_q", sc, rep, 5, 2, "tune", out, lab, config=cfgfile(out, lab, d)))
    return jobs


PHASES = {"phase0": phase0, "tune": tune, "final": final, "ablation": ablation,
          "sensitivity": sensitivity, "scaling": scaling, "convergence": convergence}

if __name__ == "__main__":
    ph, out = sys.argv[1], sys.argv[2]
    sel = sys.argv[sys.argv.index("--selected") + 1] if "--selected" in sys.argv else None
    fn = PHASES[ph]
    jobs = fn(out, sel) if ph in ("final", "ablation", "sensitivity", "scaling", "convergence") else fn(out)
    print("\n".join(jobs))
