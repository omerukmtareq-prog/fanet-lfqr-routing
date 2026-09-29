"""Keyed wireless channel: log-distance path loss, lognormal shadowing,
SINR threshold reception with random-offset overlap interference.

All randomness is keyed by (channel_seed, epoch) and (channel_seed, step) and is
drawn for every node/link regardless of which nodes transmit, so methods with
different action counts see identical channel realisations.
"""
from __future__ import annotations
import dataclasses
import numpy as np
from .config import Settings, STEP_SECONDS

DATA, BEACON, ACK, DN = "data", "beacon", "ack", "dn"


@dataclasses.dataclass
class Frame:
    sender: int
    receiver: int          # -1 for broadcast
    kind: str
    bits: int
    payload: object = None  # packet or beacon record
    offset: float = 0.0
    duration: float = 0.0


def dbm_to_mw(x):
    return 10 ** (np.asarray(x, dtype=float) / 10)


class Channel:
    def __init__(self, cfg: Settings, channel_seed: int):
        self.cfg = cfg
        self.seed = int(channel_seed)
        self._shadow_epoch = -1
        self._shadow = None
        self.noise_mw = float(dbm_to_mw(cfg.noise_dbm))
        self.beta = float(dbm_to_mw(cfg.sinr_threshold_db))
        self.tx_mw = float(dbm_to_mw(cfg.tx_power_dbm))
        self.cs_mw = float(dbm_to_mw(cfg.cs_threshold_dbm))
        self.beacon_phase = np.random.default_rng(np.random.SeedSequence([self.seed, 7])).integers(0, cfg.beacon_period, cfg.nodes)

    def shadow(self, t: int) -> np.ndarray:
        epoch = t // self.cfg.shadow_epoch
        if epoch != self._shadow_epoch:
            n = self.cfg.nodes
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, 5, epoch]))
            m = rng.standard_normal((n, n)) * self.cfg.shadow_sigma_db
            m = np.triu(m, 1); m = m + m.T   # symmetric, zero diagonal
            self._shadow, self._shadow_epoch = m, epoch
        return self._shadow

    def offsets(self, t: int, subslot: int) -> np.ndarray:
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, 9, t, subslot]))
        return rng.uniform(0.0, 1.0, self.cfg.nodes)

    def rx_power_mw(self, pos: np.ndarray, senders: np.ndarray, t: int) -> np.ndarray:
        """(F x N) received power in mW from each frame's sender at every node."""
        cfg = self.cfg
        d = np.linalg.norm(pos[senders][:, None, :] - pos[None, :, :], axis=2)
        d = np.maximum(d, 1.0)
        pl = cfg.pl_d0_db + 10 * cfg.pl_exponent * np.log10(d) + self.shadow(t)[senders]
        p = dbm_to_mw(cfg.tx_power_dbm - pl)
        p[d > cfg.prune_distance] = 0.0
        p[np.arange(len(senders)), senders] = 0.0
        return p

    def resolve(self, frames: list[Frame], pos: np.ndarray, alive: np.ndarray, t: int, subslot: int):
        """Decide which frames are transmitted (carrier-sense deferral) and which
        nodes decode them. Returns (decoded F x N bool, deferred F bool, tx_energy_j N, rx_energy_j N).
        Energy follows the receiver-state model of radio_time(); self.last_sensed
        (F x N) records which nodes detected each transmitted frame."""
        cfg = self.cfg
        n = cfg.nodes
        if not frames:
            self.last_sensed = np.zeros((0, n), dtype=bool)
            return np.zeros((0, n), dtype=bool), np.zeros(0, dtype=bool), np.zeros(n), np.zeros(n)
        senders = np.array([f.sender for f in frames])
        dur = np.array([cfg.airtime_steps(f.bits) for f in frames], dtype=float)
        if subslot == 0:
            win0, win1 = 0.0, cfg.data_subslot
        else:
            win0, win1 = cfg.data_subslot, 1.0
        u = self.offsets(t, subslot)[senders]
        start = win0 + u * np.maximum(win1 - win0 - dur, 0.0)
        end = start + dur
        for f, s, e in zip(frames, start, end):
            f.offset, f.duration = float(s), float(e - s)
        p = self.rx_power_mw(pos, senders, t)
        # carrier sense without random backoff: in offset order, a frame whose
        # sender senses an ongoing transmission (>= cs_threshold) waits until
        # the sensed frames end; if it no longer fits in the window it is
        # deferred to the next step. Nodes that cannot sense each other
        # (hidden terminals) may still overlap and interfere.
        F = len(frames)
        deferred = np.zeros(F, dtype=bool)
        order = np.argsort(start, kind="stable")
        scheduled = []
        for f in order:
            s0 = start[f]
            for _ in range(F + 1):
                conflicts = [g for g in scheduled if (senders[g] == senders[f] or p[g, senders[f]] >= self.cs_mw) and start[g] < s0 + dur[f] and end[g] > s0]
                if not conflicts:
                    break
                s0 = max(end[g] for g in conflicts)
            if s0 + dur[f] > win1 + 1e-12:
                deferred[f] = True
            else:
                start[f], end[f] = s0, s0 + dur[f]
                scheduled.append(f)
        for f, s_, e_ in zip(frames, start, end):
            f.offset, f.duration = float(s_), float(e_ - s_)
        # overlap matrix (F x F) among transmitted frames
        overlap = (start[:, None] < end[None, :]) & (start[None, :] < end[:, None])
        np.fill_diagonal(overlap, False)
        overlap[deferred, :] = False; overlap[:, deferred] = False
        if cfg.interference:
            interf = overlap.astype(float) @ p
        else:
            interf = np.zeros_like(p)
        sinr = p / (self.noise_mw + interf)
        decoded = sinr >= self.beta
        # half duplex: a node transmitting an overlapping frame cannot receive
        for g in range(len(frames)):
            busy_frames = overlap[g]
            decoded[busy_frames, senders[g]] = False
        decoded[:, ~alive] = False
        decoded[np.arange(len(frames)), senders] = False
        decoded[deferred, :] = False
        sent = ~deferred
        tx_t, rx_t = self.radio_time(senders, start, end, sent, p, alive)
        tx_e = tx_t * cfg.p_tx_w
        rx_e = rx_t * cfg.p_rx_w
        self.last_sensed = self._sensed
        return decoded, deferred, tx_e, rx_e

    def radio_time(self, senders, start, end, sent, p, alive):
        """Declared receiver-state model (pilot02, Codex receipt 0003 item 1).

        Per node and sub-slot the radio is in exactly one state at any instant:
        TX while it transmits one of its own frames; otherwise RX while at
        least one other transmitted frame arrives at >= cs_threshold (it is
        detecting/receiving, whether or not the frame is finally decoded, and
        whether or not it is addressed to it); otherwise IDLE. Overlapping
        frames are counted once (interval union) and TX takes precedence over
        RX (half duplex). Dead nodes are in no state. Returns seconds."""
        cfg = self.cfg
        n = cfg.nodes
        tx_t = np.zeros(n); rx_t = np.zeros(n)
        idx = np.nonzero(sent)[0]
        sensed = (p[idx] >= self.cs_mw) & alive[None, :]
        sensed[np.arange(len(idx)), senders[idx]] = False
        self._sensed = np.zeros((len(senders), n), dtype=bool); self._sensed[idx] = sensed
        s_idx, e_idx = start[idx], end[idx]
        for node in range(n):
            if not alive[node]:
                continue
            own = idx[senders[idx] == node]
            heard = np.nonzero(sensed[:, node])[0]
            if len(own) == 0 and len(heard) == 0:
                continue
            own_iv = [(start[g], end[g]) for g in own]
            heard_iv = [(s_idx[h], e_idx[h]) for h in heard]
            pts = sorted({x for iv in own_iv + heard_iv for x in iv})
            for a, b in zip(pts[:-1], pts[1:]):
                mid = 0.5 * (a + b)
                if any(s0 <= mid < e0 for s0, e0 in own_iv):
                    tx_t[node] += b - a
                elif any(s0 <= mid < e0 for s0, e0 in heard_iv):
                    rx_t[node] += b - a
        return tx_t * STEP_SECONDS, rx_t * STEP_SECONDS
