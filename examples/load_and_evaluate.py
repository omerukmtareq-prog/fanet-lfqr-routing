"""Load a trained LFQR model and re-run its frozen evaluation episodes.

    python examples/load_and_evaluate.py models/lfqr_models.json final/S1/lfqr/rep_00
    python examples/load_and_evaluate.py models/lfqr_models.json --all-fast   # all S1 / S1_L02 / scaling models

The script rebuilds the Settings from the scenario + selected configuration (+ evaluation
overrides), checks the settings hash stored with the model, loads theta into a frozen lfa_q
policy and replays exactly the evaluation episodes of the campaign (same trace keys, channel
seeds and policy seeds). The pooled PDR must equal the stored eval_pdr.
"""
import json, os, sys, time
for _d in ("code", "simulator"):
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", _d))
import numpy as np
from fanet_redesign.run import load_settings, job_plan
from fanet_redesign.policies import make_policy
from fanet_redesign.traces import make_trace
from fanet_redesign.sim import simulate
from fanet_redesign.config import Settings


def build_policy(model, cfg):
    pol = make_policy("lfa_q", cfg)
    pol.load_state_dict({"theta": np.array(model["theta"], float), "epsilon": 0.0, "updates": model["updates"],
                         "updates_ack": 0, "updates_dn": 0, "updates_drop": 0})
    return pol


def evaluate(bundle, model, verbose=True):
    cfg = Settings.from_dict(model["settings"])
    ecfg = Settings.from_dict(model["eval_settings"])
    assert cfg.sha256() == model["settings_sha256"] and ecfg.sha256() == model["eval_settings_sha256"]
    pol = build_policy(model, cfg)
    plan = job_plan("lfa_q", model["scenario"], model["replicate"], model["train_episodes"],
                    model["eval_episodes"], model["eval_split"])
    gen = dlv = 0
    for k, ep in enumerate(plan):
        if ep["phase"] != "eval":
            continue
        pol.cfg = ecfg
        row = simulate(ecfg, pol, make_trace(ecfg, ep["trace"]), 5_000_000 + ep["trace"], False,
                       7_000_000 + model["replicate"] * 100_000 + k, episode_index=ep["index"],
                       total_training=model["train_episodes"])
        gen += row["generated"]; dlv += row["delivered"]
    pdr = dlv / gen if gen else float("nan")
    ok = abs(pdr - model["eval_pdr"]) < 1e-12
    if verbose:
        print(f"{model['run']:40s} PDR {pdr:.6f} stored {model['eval_pdr']:.6f} {'OK' if ok else 'MISMATCH'}", flush=True)
    return ok


if __name__ == "__main__":
    bundle = json.load(open(sys.argv[1]))
    models = {m["run"]: m for m in bundle["models"]}
    sel = sys.argv[2:] or ["final/S1/lfqr/rep_00"]
    if sel == ["--all-fast"]:
        sel = [r for r in models if "/S1/" in r or "/S1_L02/" in r or r.startswith("scaling")]
    elif sel == ["--all"]:
        sel = list(models)
    t = time.time()
    res = [evaluate(bundle, models[r]) for r in sel]
    print(f"{sum(res)}/{len(res)} reproduced in {time.time() - t:.0f} s")
    sys.exit(0 if all(res) else 1)
