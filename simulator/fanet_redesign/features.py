"""Per-candidate feature vector, computed only from locally held information
(neighbour table entry, own position/velocity, packet header)."""
from __future__ import annotations
import numpy as np

N_FEATURES = 10
F_BIAS, F_PROGRESS, F_LINKLEN, F_LINKQ, F_QUEUE, F_ENERGY, F_STABILITY, F_AGE, F_TTL, F_ISDEST = range(N_FEATURES)
NAMES = ["bias", "progress", "link_len", "link_q", "queue", "energy", "stability", "beacon_age", "ttl_left", "is_dest"]

MASKS = {
    "full": np.ones(N_FEATURES),
    "no_queue": np.array([1, 1, 1, 1, 0, 1, 1, 1, 1, 1], float),
    "no_energy": np.array([1, 1, 1, 1, 1, 0, 1, 1, 1, 1], float),
    "no_link": np.array([1, 1, 1, 0, 1, 1, 0, 1, 1, 1], float),
    "geometry": np.array([1, 1, 1, 0, 0, 0, 0, 0, 1, 1], float),
}


def stability_steps(pi, vi, pj, vj, r_nom, cap=100.0):
    """Steps until the relative distance exceeds r_nom under constant velocity."""
    dp = pj - pi; dv = (vj - vi) * 0.01   # m per step
    a = float(dv @ dv); b = 2.0 * float(dp @ dv); c = float(dp @ dp) - r_nom * r_nom
    if a < 1e-12:
        return cap if c <= 0 else 0.0
    disc = b * b - 4 * a * c
    if disc < 0:
        return 0.0
    tt = (-b + np.sqrt(disc)) / (2 * a)
    return float(min(max(tt, 0.0), cap))


def candidate_features(cfg, pos_i, vel_i, entries, cands, linkq, packet, t, r_nom, mask):
    """entries: dict j -> NeighborEntry; returns (len(cands) x N_FEATURES) array."""
    k = len(cands)
    out = np.zeros((k, N_FEATURES))
    if k == 0:
        return out
    d_i = float(np.linalg.norm(pos_i - packet.dst_pos))
    for r, j in enumerate(cands):
        e = entries[j]
        d_j = float(np.linalg.norm(e.pos - packet.dst_pos))
        out[r, F_BIAS] = 1.0
        out[r, F_PROGRESS] = float(np.clip((d_i - d_j) / r_nom, -1, 1))
        out[r, F_LINKLEN] = min(float(np.linalg.norm(e.pos - pos_i)) / r_nom, 2.0)
        missed = max(0.0, (t - e.last_seen) / cfg.beacon_period - 1.0)
        out[r, F_LINKQ] = linkq.get(j, 0.8) * (0.5 ** missed)
        out[r, F_QUEUE] = e.queue
        out[r, F_ENERGY] = e.energy
        out[r, F_STABILITY] = stability_steps(pos_i, vel_i, e.pos, e.vel, r_nom) / 100.0
        out[r, F_AGE] = (t - e.last_seen) / cfg.neighbor_expiry
        out[r, F_TTL] = max(packet.expires - t, 0) / cfg.ttl
        out[r, F_ISDEST] = 1.0 if j == packet.dst else 0.0
    return out * mask
