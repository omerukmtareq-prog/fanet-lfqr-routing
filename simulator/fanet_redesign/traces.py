"""Exogenous, keyed inputs: mobility, traffic, failures.

Everything here depends only on (settings, key). It is generated before the
episode and never consumes policy or channel randomness.
"""
from __future__ import annotations
import hashlib, json
import numpy as np
from .config import Settings


class Trace:
    __slots__ = ("pos", "vel", "traffic", "failed", "sha256", "key", "horizon")

    def __init__(self, pos, vel, traffic, failed, key, sha):
        self.pos, self.vel, self.traffic, self.failed = pos, vel, traffic, failed
        self.key, self.sha256, self.horizon = key, sha, pos.shape[0]


def make_trace(cfg: Settings, key: int) -> Trace:
    n = cfg.nodes
    horizon = cfg.steps + cfg.ttl + 2
    rng = np.random.default_rng(np.random.SeedSequence([key, 11]))
    pos = np.empty((horizon, n, 2))
    vel = np.zeros((horizon, n, 2))
    pos[0] = rng.uniform(cfg.margin, cfg.area - cfg.margin, (n, 2))
    groups = np.arange(n) % 3
    lo = np.choose(groups, [66.7, 33.4, 0.0]) * cfg.speed_scale
    hi = np.choose(groups, [100.0, 66.6, 33.3]) * cfg.speed_scale
    speeds = rng.uniform(lo, hi)
    heading = rng.uniform(0, 2 * np.pi, n)
    for t in range(1, horizon):
        if t % cfg.move_period == 0:
            heading = rng.uniform(0, 2 * np.pi, n)
        v = np.stack((np.cos(heading), np.sin(heading)), 1) * speeds[:, None]
        nxt = pos[t - 1] + v * 0.01
        pos[t] = np.clip(nxt, cfg.margin, cfg.area - cfg.margin)
        vel[t] = (pos[t] - pos[t - 1]) / 0.01
    traffic_rng = np.random.default_rng(np.random.SeedSequence([key, 22]))
    traffic = []
    pid = 0
    for t in range(cfg.steps):
        events = []
        for src, count in enumerate(traffic_rng.poisson(cfg.load, n)):
            for _ in range(int(count)):
                dest = int(traffic_rng.integers(n - 1))
                dest += dest >= src
                events.append((pid, src, dest))
                pid += 1
        traffic.append(events)
    fail_rng = np.random.default_rng(np.random.SeedSequence([key, 33]))
    if cfg.failure_pattern == "random":
        order = fail_rng.permutation(n)
    elif cfg.failure_pattern == "spatial":
        ref = pos[min(cfg.failure_step, horizon - 1)]
        order = np.argsort(np.sum((ref - cfg.area / 2) ** 2, axis=1))
    else:
        raise ValueError("Unknown failure pattern")
    failed = np.sort(order[: int(n * cfg.failure_ratio)]).astype(int)
    h = hashlib.sha256()
    h.update(pos.tobytes()); h.update(json.dumps(traffic).encode()); h.update(failed.tobytes())
    return Trace(pos, vel, traffic, failed, key, h.hexdigest())
