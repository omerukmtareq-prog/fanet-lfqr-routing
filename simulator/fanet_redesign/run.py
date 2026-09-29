"""Command-line entry points: run (train+evaluate, resumable), summarize, test, smoke.

Run layout (one directory per job):
  <out>/manifest.json     settings, method, split ids, code hash (checked on resume)
  <out>/checkpoint.pkl    CHECKPOINT_SCHEMA (see checkpoint_schema())
  <out>/records.json      list of per-episode metric dicts (METRICS schema)
  <out>/DONE              written when all episodes are complete
All writes are atomic (temp file + rename).
"""
from __future__ import annotations
import argparse, hashlib, json, os, pickle, platform, sys, time
from pathlib import Path
import numpy as np
from . import __version__
from .config import Settings, SCENARIOS, trace_key, replicate_index
from .traces import make_trace
from .policies import make_policy, METHODS
from .sim import simulate

CHECKPOINT_SCHEMA = 1
METRICS_SCHEMA = 2
PACKAGE_DIR = Path(__file__).resolve().parent


def code_sha256() -> str:
    h = hashlib.sha256()
    for f in sorted(PACKAGE_DIR.glob("*.py")):
        h.update(f.name.encode()); h.update(f.read_bytes())
    return h.hexdigest()


def atomic_write_bytes(path: Path, data: bytes):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data); os.replace(tmp, path)


def atomic_write_text(path: Path, text: str):
    atomic_write_bytes(path, text.encode())


def checkpoint_schema() -> dict:
    return {
        "schema": CHECKPOINT_SCHEMA,
        "fields": {
            "schema": "int, CHECKPOINT_SCHEMA",
            "policy_state": "policy.state_dict(): name plus learnable state (theta or Q rows), epsilon, update counters",
            "next_episode": "int, index of the next episode to run (episode granularity; no mid-episode resume)",
            "records": "list of per-episode metric dicts already completed",
            "rng_note": "policy RNG is re-derived from (policy_seed, episode) so no generator state is stored",
            "code_sha256": "hash of the package sources that wrote the checkpoint",
            "written_utc": "ISO timestamp",
        },
    }


def job_plan(method: str, scenario: str, replicate: int, train: int, evaluate: int, eval_split: str):
    """Episodes: training on split 'train' (learners only), then frozen evaluation on eval_split.

    pilot02 (Codex receipt 0003, item 2): evaluation traces are distinct per
    replicate (index = replicate * 1000 + k) and paired across methods within a
    replicate; the channel realisation is keyed by the trace, so it is distinct
    across replicates too. pilot01 evaluated every replicate on the same keys."""
    plan = []
    if METHODS[method].learns:
        for k in range(train):
            plan.append(dict(phase="train", split="train", index=k, trace=trace_key("train", scenario, replicate_index(replicate, k))))
    for k in range(evaluate):
        plan.append(dict(phase="eval", split=eval_split, index=k, trace=trace_key(eval_split, scenario, replicate_index(replicate, k))))
    return plan


def run_job(cfg: Settings, method: str, scenario: str, replicate: int, train: int, evaluate: int,
            eval_split: str, out: Path, quiet: bool = False, label: str | None = None,
            eval_cfg: Settings | None = None):
    """eval_cfg (optional) changes the operating condition of the evaluation
    episodes only: a policy trained under cfg is evaluated, frozen, under
    eval_cfg (transfer / sensitivity). Default: eval_cfg = cfg."""
    out.mkdir(parents=True, exist_ok=True)
    eval_cfg = eval_cfg or cfg
    plan = job_plan(method, scenario, replicate, train, evaluate, eval_split)
    manifest = dict(package_version=__version__, code_sha256=code_sha256(), settings=cfg.to_dict(), settings_sha256=cfg.sha256(),
                    method=method, scenario=scenario, replicate=replicate, train=train, evaluate=evaluate,
                    eval_split=eval_split, plan=plan, python=platform.python_version(), numpy=np.__version__,
                    checkpoint_schema=CHECKPOINT_SCHEMA, metrics_schema=METRICS_SCHEMA, label=label or method,
                    eval_settings=eval_cfg.to_dict(), eval_settings_sha256=eval_cfg.sha256())
    mp = out / "manifest.json"
    if mp.exists():
        old = json.loads(mp.read_text())
        for k in ("settings_sha256", "method", "scenario", "replicate", "train", "evaluate", "eval_split", "eval_settings_sha256"):
            if old.get(k) != manifest[k]:
                raise RuntimeError(f"run directory manifest differs in {k}; refusing to mix runs")
        if old.get("code_sha256") != manifest["code_sha256"]:
            raise RuntimeError("run directory was written by different package sources; use a new run directory")
    else:
        atomic_write_text(mp, json.dumps(manifest, indent=1))
    policy = make_policy(method, cfg)
    ck = out / "checkpoint.pkl"
    if ck.exists():
        state = pickle.loads(ck.read_bytes())
        if state.get("schema") != CHECKPOINT_SCHEMA:
            raise RuntimeError("checkpoint schema mismatch")
        policy.load_state_dict(state["policy_state"])
        records, start = state["records"], state["next_episode"]
    else:
        records, start = [], 0
    if (out / "DONE").exists() and start >= len(plan):
        return records
    policy_seed_base = 7_000_000 + replicate * 100_000
    channel_seed_base = 5_000_000
    for k in range(start, len(plan)):
        ep = plan[k]
        training = ep["phase"] == "train"
        ecfg = cfg if training else eval_cfg
        trace = make_trace(ecfg, ep["trace"])
        policy.cfg = ecfg
        # channel realisation is keyed by the trace, identical for every method
        channel_seed = channel_seed_base + ep["trace"]
        policy_seed = policy_seed_base + k
        row = simulate(ecfg, policy, trace, channel_seed, training, policy_seed,
                       episode_index=ep["index"], total_training=train)
        row.update(episode=k, phase=ep["phase"], split=ep["split"], index=ep["index"], replicate=replicate,
                   scenario=scenario, policy_seed=policy_seed, metrics_schema=METRICS_SCHEMA)
        records.append(row)
        state = dict(schema=CHECKPOINT_SCHEMA, policy_state=policy.state_dict(), next_episode=k + 1, records=records,
                     code_sha256=manifest["code_sha256"], written_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        atomic_write_bytes(ck, pickle.dumps(state, protocol=4))
        atomic_write_text(out / "records.json", json.dumps(records, indent=1, default=float))
        if not quiet:
            print(json.dumps({x: row[x] for x in ["method", "phase", "index", "pdr", "delay_mean", "elapsed_seconds"]}), flush=True)
    atomic_write_text(out / "DONE", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    return records


def load_settings(scenario: str, config: str | None, overrides: list[str]) -> Settings:
    d = dict(SCENARIOS[scenario]) if scenario in SCENARIOS else {}
    if config:
        d.update(json.loads(Path(config).read_text()))
    for ov in overrides:
        k, v = ov.split("=", 1)
        d[k] = json.loads(v)
    return Settings.from_dict(d)


def summarize(root: Path) -> dict:
    rows = []
    for rp in sorted(root.rglob("records.json")):
        m = json.loads((rp.parent / "manifest.json").read_text()) if (rp.parent / "manifest.json").exists() else {}
        for r in json.loads(rp.read_text()):
            if r.get("phase") == "eval":
                rows.append(dict(run=str(rp.parent.relative_to(root)), done=(rp.parent / "DONE").exists(), **{k: r.get(k) for k in
                            ["method", "scenario", "replicate", "index", "pdr", "delay_mean", "delay_censored_mean", "energy_active_per_delivered_j",
                             "energy_total_per_delivered_j", "lifetime_projection_s", "control_overhead_ratio", "information_overhead_ratio",
                             "attempts_per_delivered", "generated", "delivered", "trace_key"]},
                            label=m.get("label") or rp.parent.parent.name))
    groups = {}
    fields = ("delay_mean", "delay_censored_mean", "energy_active_per_delivered_j", "energy_total_per_delivered_j",
              "lifetime_projection_s", "control_overhead_ratio", "information_overhead_ratio")
    for r in rows:
        key = (r["scenario"], r["label"])
        g = groups.setdefault(key, dict(scenario=r["scenario"], label=r["label"], method=r["method"], per_rep={}, vals={f: [] for f in fields}))
        pr = g["per_rep"].setdefault(r["replicate"], [0, 0])
        pr[0] += r["generated"]; pr[1] += r["delivered"]
        for f in fields:
            if r[f] is not None:
                g["vals"][f].append(r[f])
    table = []
    for g in groups.values():
        rep_pdr = {k: (v[1] / v[0] if v[0] else None) for k, v in sorted(g["per_rep"].items())}
        pdrs = [x for x in rep_pdr.values() if x is not None]
        row = dict(scenario=g["scenario"], label=g["label"], method=g["method"], replicates=len(rep_pdr),
                   eval_episodes=len(g["vals"]["delay_censored_mean"]),
                   pdr_rep_mean=float(np.mean(pdrs)) if pdrs else None,
                   pdr_rep_sd=float(np.std(pdrs, ddof=1)) if len(pdrs) > 1 else None,
                   pdr_by_replicate=rep_pdr)
        for f in fields:
            row[f] = float(np.mean(g["vals"][f])) if g["vals"][f] else None
        table.append(row)
    table.sort(key=lambda r: (r["scenario"], r["label"]))
    return dict(rows=rows, table=table)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="fanet_redesign")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="train (learners) and evaluate one job; rerun the same command to resume")
    r.add_argument("--method", required=True, choices=sorted(METHODS))
    r.add_argument("--scenario", default="S1")
    r.add_argument("--replicate", type=int, default=0)
    r.add_argument("--train", type=int, default=5)
    r.add_argument("--eval", type=int, default=2)
    r.add_argument("--eval-split", default="tune", choices=["tune", "final"])
    r.add_argument("--config", help="JSON file of Settings overrides")
    r.add_argument("--set", action="append", default=[], help="override key=json, e.g. --set load=0.02")
    r.add_argument("--out", required=True)
    r.add_argument("--quiet", action="store_true")
    r.add_argument("--label", help="variant label used by summarize (default: method name)")
    r.add_argument("--eval-set", action="append", default=[], help="override key=json for evaluation episodes only (transfer)")
    s = sub.add_parser("summarize"); s.add_argument("--root", required=True); s.add_argument("--json", action="store_true")
    sub.add_parser("test", help="run the correctness test suite")
    sm = sub.add_parser("smoke", help="tiny end-to-end run of every method"); sm.add_argument("--out", required=True)
    sub.add_parser("schema", help="print checkpoint and metrics schema")
    sub.add_parser("hash", help="print the package source hash")
    w = sub.add_parser("wire", help="print the wire-format bit budget"); w.add_argument("--nodes", type=int, default=100)
    a = ap.parse_args(argv)
    if a.cmd == "run":
        cfg = load_settings(a.scenario, a.config, a.set)
        ecfg = load_settings(a.scenario, a.config, a.set + a.eval_set) if a.eval_set else None
        run_job(cfg, a.method, a.scenario, a.replicate, a.train, a.eval, a.eval_split, Path(a.out), a.quiet, a.label, ecfg)
    elif a.cmd == "summarize":
        res = summarize(Path(a.root))
        if a.json:
            print(json.dumps(res, indent=1))
        else:
            for row in res["table"]:
                print(json.dumps(row))
    elif a.cmd == "test":
        from .tests import run_all
        sys.exit(run_all())
    elif a.cmd == "smoke":
        from .tests import smoke
        smoke(Path(a.out))
    elif a.cmd == "schema":
        from .sim import Simulator
        print(json.dumps(dict(checkpoint=checkpoint_schema(), metrics_schema=METRICS_SCHEMA,
                              metrics_fields="see METRICS_SCHEMA.md in the release"), indent=1))
    elif a.cmd == "hash":
        print(code_sha256())
    elif a.cmd == "wire":
        from .wire import budget_table
        cfg = Settings(nodes=a.nodes)
        print(json.dumps(budget_table(cfg.nodes, cfg.max_visited, cfg.payload_bits), indent=1))


if __name__ == "__main__":
    main()
