"""Forwarding policies. All share the same candidate set and features
(computed by the simulator) and differ only in selection and learning.

Interface used by the simulator (all ids are ints, pid = packet id):
  candidate_mode           'all' or 'quadrant'
  uses_value               True if ACKs must carry a piggybacked value
  begin_episode(training, episode_index, total_training_episodes, rng)
  choose(i, packet, cands, feats, t, training) -> j
  value(j, packet, cands, feats) -> float | None     (bootstrap value j reports on ACK)
  on_forward(i, packet, j, feat, t)                 (a decision was executed)
  on_ack(i, packet, j, delivered, cost, value, e_j, refused=False)
                                                    (delayed feedback arrived; refused = j's buffer was full)
  on_dn(i, packet, j)                               (drop notification from j)
  on_local_drop(i, packet)                          (i itself dropped the packet)
  state_dict() / load_state_dict(d) / memory_bytes()
"""
from __future__ import annotations
import numpy as np
from .config import Settings
from .features import N_FEATURES, F_PROGRESS, F_LINKQ, F_QUEUE, F_ENERGY, F_STABILITY, F_ISDEST, F_LINKLEN
from .config import STEP_SECONDS


class Policy:
    candidate_mode = "all"
    uses_value = False
    learns = False
    name = "base"

    def __init__(self, cfg: Settings):
        self.cfg = cfg
        self.n = cfg.nodes
        self.rng = np.random.default_rng(0)
        self.training = False
        self.epsilon = 0.0
        self.updates = 0

    def begin_episode(self, training, episode_index, total_training, rng):
        self.training, self.rng = training, rng

    def choose(self, i, packet, cands, feats, t, training):
        raise NotImplementedError

    def value(self, j, packet, cands, feats):
        return None

    def on_forward(self, i, packet, j, feat, t): pass
    def on_ack(self, i, packet, j, delivered, cost, value, e_j, refused=False, t=0): pass
    def on_attempt_fail(self, i, packet, j, t): pass
    def on_step_end(self, t): pass
    def on_dn(self, i, packet, j): pass
    def on_local_drop(self, i, packet): pass
    def state_dict(self): return {"name": self.name}
    def load_state_dict(self, d): pass
    def memory_bytes(self): return 0

    def _tie_break(self, cands, score, minimize=True):
        best = score.min() if minimize else score.max()
        tied = [c for c, s in zip(cands, score) if s == best]
        return int(tied[0]) if len(tied) == 1 else int(self.rng.choice(tied))


class GeoGreedy(Policy):
    name = "geo_greedy"
    def choose(self, i, packet, cands, feats, t, training):
        prog = feats[:, F_PROGRESS]
        if packet.dst in cands:
            return packet.dst
        if prog.max() <= 0:
            return None  # local minimum: no recovery
        return self._tie_break(cands, prog, minimize=False)


class GeoGreedyFresh(GeoGreedy):
    """Greedy on progress, near-ties (within 0.05) resolved by link quality."""
    name = "geo_greedy_fresh"
    def choose(self, i, packet, cands, feats, t, training):
        if packet.dst in cands:
            return packet.dst
        prog = feats[:, F_PROGRESS]
        if prog.max() <= 0:
            return None
        near = prog >= prog.max() - 0.05
        score = np.where(near, feats[:, F_LINKQ], -1.0)
        return self._tie_break(cands, score, minimize=False)


class RandomQuadrant(Policy):
    name = "random_quadrant"
    candidate_mode = "quadrant"
    def choose(self, i, packet, cands, feats, t, training):
        if packet.dst in cands:
            return packet.dst
        return int(self.rng.choice(cands))


class CostHeuristic(Policy):
    """Non-learning control using the same information as the learner."""
    name = "cost_heuristic"
    def choose(self, i, packet, cands, feats, t, training):
        if packet.dst in cands:
            return packet.dst
        c = self.cfg
        score = (c.w_progress * (1 - feats[:, F_PROGRESS]) + c.w_queue * feats[:, F_QUEUE]
                 + c.w_energy * (1 - feats[:, F_ENERGY]) + c.w_link * (1 - feats[:, F_LINKQ])
                 + c.w_stability * (1 - feats[:, F_STABILITY]))
        return self._tie_break(cands, score, minimize=True)


class TabularHist(Policy):
    """Historical corrected destination-ID tabular learner, on the new model.
    Rewards +100 / -1 / -50, alpha 0.01, per-decision epsilon decay, quadrant
    candidates, bootstrap value carried on the ACK. Drop notifications are
    ignored (the historical method had none); local drops give -50."""
    name = "tabular_q_hist"
    candidate_mode = "quadrant"
    uses_value = True
    learns = True

    def __init__(self, cfg):
        super().__init__(cfg)
        self.q = [dict() for _ in range(self.n)]
        self.eps = np.full(self.n, cfg.hist_eps_start)
        self.pending = {}

    def row(self, i, d):
        r = self.q[i].get(d)
        if r is None:
            r = self.q[i][d] = np.zeros(self.n)
        return r

    def choose(self, i, packet, cands, feats, t, training):
        d = packet.dst
        if d in cands:
            return d
        eps = self.eps[i] if training else 0.0
        if training:
            self.eps[i] = max(self.cfg.hist_eps_min, eps * (1 - self.cfg.hist_eps_decay))
        if self.rng.random() < eps:
            return int(self.rng.choice(cands))
        row = self.q[i].get(d)
        vals = np.zeros(len(cands)) if row is None else row[cands]
        return self._tie_break(cands, vals, minimize=False)

    def value(self, j, packet, cands, feats):
        d = packet.dst
        if j == d:
            return 0.0
        if not cands:
            return 0.0
        row = self.q[j].get(d)
        return 0.0 if row is None else float(row[cands].max())

    def on_forward(self, i, packet, j, feat, t):
        self.pending[(i, packet.pid)] = j

    def on_ack(self, i, packet, j, delivered, cost, value, e_j, refused=False, t=0):
        pid, d = packet.pid, packet.dst
        if not self.training:
            self.pending.pop((i, pid), None); return
        c = self.cfg
        if refused:      # pilot01 bootstrapped from the refusal's c_drop value (sign error); pilot02 treats it as a drop
            target = c.hist_reward_drop
        else:
            target = c.hist_reward_delivery if delivered else c.hist_reward_hop + c.hist_gamma * (value or 0.0)
        row = self.row(i, d)
        row[j] += c.hist_alpha * (target - row[j]); self.updates += 1
        self.pending.pop((i, pid), None)

    def on_local_drop(self, i, packet):
        pid, d = packet.pid, packet.dst
        j = self.pending.pop((i, pid), None)
        if self.training and j is not None:
            row = self.row(i, d)
            row[j] += self.cfg.hist_alpha * (self.cfg.hist_reward_drop - row[j]); self.updates += 1

    def state_dict(self):
        return {"name": self.name, "q": [{int(k): v.copy() for k, v in q.items()} for q in self.q],
                "eps": self.eps.copy(), "updates": self.updates}

    def load_state_dict(self, d):
        self.q = [{int(k): np.array(v) for k, v in q.items()} for q in d["q"]]
        self.eps = np.array(d["eps"]); self.updates = int(d["updates"])

    def memory_bytes(self):
        return sum(v.nbytes for q in self.q for v in q.values())


def heuristic_theta(cfg: Settings) -> np.ndarray:
    """theta with theta.phi = init_scale * cost_heuristic score (same ordering).
    score = w_p(1-p) + w_q u + w_e(1-e) + w_l(1-q) + w_s(1-s)."""
    th = np.zeros(N_FEATURES)
    k = cfg.init_scale
    th[0] = k * (cfg.w_progress + cfg.w_energy + cfg.w_link + cfg.w_stability)
    th[F_PROGRESS] = -k * cfg.w_progress
    th[F_QUEUE] = k * cfg.w_queue
    th[F_ENERGY] = -k * cfg.w_energy
    th[F_LINKQ] = -k * cfg.w_link
    th[F_STABILITY] = -k * cfg.w_stability
    return th


class LinearQ(Policy):
    """Proposed method: per-node linear cost-to-go Q(phi) = theta_i . phi,
    semi-gradient one-step update with delayed ACK feedback and optional drop
    notifications. Cost framing: lower is better."""
    name = "lfa_q"
    uses_value = True
    learns = True

    def __init__(self, cfg):
        super().__init__(cfg)
        if cfg.weight_sharing not in ("per_node", "shared"):
            raise ValueError(f"unknown weight_sharing {cfg.weight_sharing}")
        rows = 1 if cfg.weight_sharing == "shared" else self.n
        self.nf = N_FEATURES + (3 if cfg.feature_interactions else 0)
        self.theta = np.zeros((rows, self.nf))
        if cfg.init == "heuristic":
            self.theta[:, :N_FEATURES] = heuristic_theta(cfg)
        elif cfg.init != "zero":
            raise ValueError(f"unknown init {cfg.init}")
        self.pending = {}   # (i,pid) -> dict(feat, acked, dn)
        self.updates_ack = 0; self.updates_dn = 0; self.updates_drop = 0
        self.epsilon = cfg.eps_start
        self.cost_sum = 0.0; self.cost_n = 0

    def begin_episode(self, training, k, total_training, rng):
        super().begin_episode(training, k, total_training, rng)
        c = self.cfg
        if training and total_training > 1:
            self.epsilon = c.eps_start * (c.eps_end / c.eps_start) ** (k / (total_training - 1))
        elif training:
            self.epsilon = c.eps_start
        else:
            self.epsilon = 0.0

    def row(self, i):
        return 0 if self.theta.shape[0] == 1 else i

    def phi(self, feats):
        """Feature map. With interactions: progress*link_q, queue*link_q, progress*queue."""
        if not self.cfg.feature_interactions:
            return feats
        f = np.atleast_2d(feats)
        x = np.column_stack([f, f[:, F_PROGRESS] * f[:, F_LINKQ], f[:, F_QUEUE] * f[:, F_LINKQ], f[:, F_PROGRESS] * f[:, F_QUEUE]])
        return x if feats.ndim == 2 else x[0]

    def qvals(self, i, feats):
        return self.phi(feats) @ self.theta[self.row(i)]

    def choose(self, i, packet, cands, feats, t, training):
        if packet.dst in cands:
            return packet.dst
        if training and self.rng.random() < self.epsilon:
            return int(self.rng.choice(cands))
        return self._tie_break(cands, self.qvals(i, feats), minimize=True)

    def value(self, j, packet, cands, feats):
        if j == packet.dst:
            return 0.0
        if not cands:
            return float(self.cfg.c_drop)
        if not self.cfg.use_bootstrap:
            return 0.0
        return float(self.qvals(j, feats).min())

    def on_forward(self, i, packet, j, feat, t):
        pid = packet.pid
        self.pending[(i, pid)] = {"feat": self.phi(np.asarray(feat)).copy(), "j": j, "acked": False, "dn": False, "t": t}

    def _update(self, i, feat, target):
        if not (self.training or self.cfg.online_adaptation):
            return
        r = self.row(i)
        err = target - float(feat @ self.theta[r])
        self.theta[r] += self.cfg.alpha * err * feat
        self.updates += 1

    def on_ack(self, i, packet, j, delivered, cost, value, e_j, refused=False, t=0):
        pid = packet.pid
        p = self.pending.get((i, pid))
        if p is None or p["acked"]:
            return
        p["acked"] = True
        c = self.cfg
        if refused:
            target = cost + float(c.c_drop)          # terminal: the relay refused the packet
        else:
            target = cost + c.lambda_lifetime * (1.0 - e_j) + (0.0 if delivered else c.gamma * (value if value is not None else 0.0))
        self._update(i, p["feat"], target); self.updates_ack += 1
        self.cost_sum += target; self.cost_n += 1
        if delivered or refused or not c.use_drop_notification:
            self.pending.pop((i, pid), None)

    def on_dn(self, i, packet, j):
        pid = packet.pid
        p = self.pending.pop((i, pid), None)
        if p is None or p["dn"] or not self.cfg.use_drop_notification:
            return
        p["dn"] = True
        self._update(i, p["feat"], float(self.cfg.c_drop)); self.updates_dn += 1

    def on_local_drop(self, i, packet):
        pid = packet.pid
        p = self.pending.pop((i, pid), None)
        if p is None or p["acked"]:
            return  # an acked forward that later expired elsewhere is handled by DN
        self._update(i, p["feat"], float(self.cfg.c_drop)); self.updates_drop += 1

    def expire_pending(self, t):
        ttl = self.cfg.ttl
        for k in [k for k, p in self.pending.items() if t - p["t"] > ttl]:
            self.pending.pop(k)

    def state_dict(self):
        return {"name": self.name, "theta": self.theta.copy(), "epsilon": self.epsilon,
                "updates": self.updates, "updates_ack": self.updates_ack,
                "updates_dn": self.updates_dn, "updates_drop": self.updates_drop}

    def load_state_dict(self, d):
        self.theta = np.array(d["theta"], dtype=float); self.epsilon = float(d["epsilon"])
        self.updates = int(d["updates"]); self.updates_ack = int(d["updates_ack"])
        self.updates_dn = int(d["updates_dn"]); self.updates_drop = int(d["updates_drop"])

    def memory_bytes(self):
        return int(self.theta.nbytes)


class GPSR(Policy):
    """Greedy Perimeter Stateless Routing (Karp & Kung, MobiCom 2000): greedy
    forwarding on beacon positions; at a local maximum the packet enters
    perimeter mode and is routed by the right-hand rule on the Gabriel-graph
    planarisation of the node's neighbour table, with face changes where an
    edge crosses the segment Lp-D closer to D than Lf, returning to greedy
    mode as soon as a node is closer to D than Lp. A packet that would reuse
    its first perimeter edge e0 is undeliverable. Perimeter state (mode, Lp,
    Lf, e0) travels in the data header and is charged (wire.gpsr_header_bits).
    Adaptations: 2-D positions from beacons/location service of this
    simulator; the common retry/ACK MAC of the simulator; no route caching."""
    name = "gpsr"
    candidate_mode = "gpsr"

    def _gabriel(self, i, t):
        tab = self.sim.nbr[i]; me = self.sim.trace.pos[t][i]
        ids = [j for j in tab if j != i and t - tab[j].last_seen <= self.cfg.neighbor_expiry]
        pos = {j: tab[j].pos for j in ids}
        keep = []
        for v in ids:
            m = 0.5 * (me + pos[v]); r2 = float(np.sum((me - pos[v]) ** 2)) / 4.0
            if all(float(np.sum((pos[w] - m) ** 2)) >= r2 for w in ids if w != v):
                keep.append(v)
        return keep, pos, me

    @staticmethod
    def _ccw_first(me, ref_vec, cand, pos, exclude_zero=True):
        """First candidate counter-clockwise from the reference direction."""
        a0 = np.arctan2(ref_vec[1], ref_vec[0])
        best, bj = None, None
        for j in cand:
            v = pos[j] - me
            ang = (np.arctan2(v[1], v[0]) - a0) % (2 * np.pi)
            if exclude_zero and ang < 1e-9:
                ang = 2 * np.pi
            if best is None or ang < best or (ang == best and j < bj):
                best, bj = ang, j
        return bj

    @staticmethod
    def _cross(p1, p2, q1, q2):
        """Intersection point of segments p1-p2 and q1-q2, or None."""
        d1 = p2 - p1; d2 = q2 - q1
        den = d1[0] * d2[1] - d1[1] * d2[0]
        if abs(den) < 1e-12:
            return None
        w = q1 - p1
        s_ = (w[0] * d2[1] - w[1] * d2[0]) / den
        u_ = (w[0] * d1[1] - w[1] * d1[0]) / den
        if 1e-9 < s_ < 1 - 1e-9 and 0 <= u_ <= 1:
            return p1 + s_ * d1
        return None

    def choose(self, i, packet, cands, feats, t, training):
        if packet.dst in cands:
            packet.gpsr = ()
            return packet.dst
        if not cands:
            return None
        tab = self.sim.nbr[i]; me = self.sim.trace.pos[t][i]; D = np.asarray(packet.dst_pos, float)
        dme = float(np.linalg.norm(me - D))
        st = packet.gpsr
        mode = st[0] if st else 0
        if mode == 1 and dme < float(np.linalg.norm(st[1] - D)):
            mode = 0
        if mode == 0:
            dists = [float(np.linalg.norm(tab[j].pos - D)) for j in cands]
            k = int(np.argmin(dists))
            if dists[k] < dme:
                packet.gpsr = ()
                return int(cands[k])
            planar, pos, me = self._gabriel(i, t)
            planar = [j for j in planar if j in cands]
            if not planar:
                return None
            nxt = self._ccw_first(me, D - me, planar, pos, exclude_zero=False)
            packet.gpsr = (1, me.copy(), me.copy(), (i, nxt))
            return int(nxt)
        _, Lp, Lf, e0 = st
        planar, pos, me = self._gabriel(i, t)
        planar = [j for j in planar if j in cands]
        if not planar:
            return None
        prev = packet.prev_hop
        ref = (pos[prev] - me) if prev in pos else (D - me)
        nxt = self._ccw_first(me, ref, planar, pos)
        for _ in range(len(planar)):
            x = self._cross(me, pos[nxt], Lp, D)
            if x is not None and float(np.linalg.norm(x - D)) < float(np.linalg.norm(Lf - D)):
                Lf = x
                nxt = self._ccw_first(me, pos[nxt] - me, planar, pos)
                continue
            break
        if (i, nxt) == tuple(e0):
            return None
        packet.gpsr = (1, Lp, Lf, e0)
        return int(nxt)


class QMR(Policy):
    """QMR (Liu, Wang, He et al., Computer Communications 150 (2020) 304-316),
    transcribed from the authors' published implementation (qmr.c,
    github.com/AIWN-ICT/QMR). Per-neighbour one-hop delay from a 5-sample
    window with EWMA factor B=0.5 plus the neighbour's reported queue wait;
    adaptive learning rate a = 1-exp(-|delay-avg|/std) (0.3 when std=0);
    discount r_j = 1-(added+deleted)/(old+added) neighbour churn per 100 ms;
    reward 1 at the destination, else W*exp(-delay/maxdelay)+(1-W)*energy_j
    with W=0.8; no-ACK / no-route penalty -1; candidate filter by required
    velocity dist/remaining-deadline versus the neighbour's actual velocity
    (progress to D of its predicted position divided by its delay); choice
    of max Q*tq with tq = (1-d/R)*link quality; fallback to the fastest
    neighbour; epsilon=0.1 exploration during learning; Q keeps updating in
    the test phase as in qmr.c. Adaptations for this simulator: the paper
    has one sink, here Q is kept per (destination, neighbour); max-Q, r and
    queue wait travel on the ACK (charged 32+8+16 bits) instead of HELLO;
    remaining TTL plays the deadline; link quality is the simulator's
    beacon/ACK success estimate; the common beacon period applies."""
    name = "qmr"
    uses_value = True
    learns = True
    ack_extra_bits = 8 + 16

    def __init__(self, cfg):
        super().__init__(cfg)
        self.Q = [dict() for _ in range(self.n)]           # (d, j) -> q
        self.nb = [dict() for _ in range(self.n)]          # j -> stats
        self.remote = [dict() for _ in range(self.n)]      # (d, j) -> (maxq_j, r_j)
        self.r = np.zeros(self.n)
        self.maxdelay = np.full(self.n, 1e-9)
        self.prev_set = [set() for _ in range(self.n)]
        self.fails = {}
        self.t0_delay = max(cfg.airtime_steps(cfg.payload_bits), 1e-3)

    def _stats(self, i, j):
        st = self.nb[i].get(j)
        if st is None:
            st = self.nb[i][j] = {"win": [self.t0_delay], "delay": self.t0_delay, "macd": self.t0_delay, "a": 0.3}
            self.maxdelay[i] = max(self.maxdelay[i], self.t0_delay)
        return st

    def _q(self, i, d, j):
        return self.Q[i].get((d, j), 0.5)

    def _learning(self):
        return self.training or self.cfg.qmr_online

    def choose(self, i, packet, cands, feats, t, training):
        if packet.dst in cands:
            return packet.dst
        c = self.cfg; sim = self.sim; tab = sim.nbr[i]
        me = sim.trace.pos[t][i]; D = np.asarray(packet.dst_pos, float)
        dist = float(np.linalg.norm(me - D))
        req = dist / (max(packet.expires - t, 1) * STEP_SECONDS)
        R = sim.r_nom
        best_q, best_j, best_sp = None, None, None
        max_sp, max_j = 0.0, None
        for r_, j in enumerate(cands):
            e = tab[j]; st = self._stats(i, j)
            pred = e.pos + e.vel * (t - e.last_seen + st["macd"]) * STEP_SECONDS
            dd = float(np.linalg.norm(me - pred))
            if dd > R:
                continue
            tq = (1.0 - dd / R) * float(feats[r_, F_LINKQ])
            sp = (dist - float(np.linalg.norm(pred - D))) / (st["delay"] * STEP_SECONDS)
            if sp >= req:
                v = self._q(i, packet.dst, j) * tq
                if best_q is None or v > best_q or (v == best_q and sp > best_sp):
                    best_q, best_j, best_sp = v, j, sp
            if sp > max_sp:
                max_sp, max_j = sp, j
        choice = best_j if best_j is not None else (max_j if max_sp > 0 else None)
        if training and self.rng.random() < c.qmr_eps:
            choice = int(self.rng.choice(cands))
        return None if choice is None else int(choice)

    def value(self, j, packet, cands, feats):
        wait = float(self.sim.qwait[j])
        if j == packet.dst:
            return (0.0, float(self.r[j]), wait)
        tab = self.sim.nbr[j]
        mq = max([self._q(j, packet.dst, k) for k in tab] or [0.0])
        return (float(mq), float(self.r[j]), wait)

    def _update(self, i, d, j, reward):
        st = self._stats(i, j)
        mq, rj = self.remote[i].get((d, j), (0.5, 0.0))
        q = self._q(i, d, j)
        self.Q[i][(d, j)] = (1 - st["a"]) * q + st["a"] * (reward + rj * mq)
        self.updates += 1

    def on_ack(self, i, packet, j, delivered, cost, value, e_j, refused=False, t=0):
        if not self._learning():
            return
        c = self.cfg; d = packet.dst
        self.fails.pop((i, packet.pid, j), None)
        if not isinstance(value, tuple):
            value = None                      # refusal ACKs carry no QMR fields
        if value is not None:
            self.remote[i][(d, j)] = (value[0], value[1])
        if refused:
            self._update(i, d, j, -1.0); return
        st = self._stats(i, j)
        mac = float(t - packet.hop_t0 + 1) if packet.hop_t0 >= 0 else 1.0
        win = st["win"]
        delay = mac if not win else (1 - c.qmr_ewma) * (sum(win) / len(win)) + c.qmr_ewma * mac
        st["macd"] = delay
        delay += value[2] if value is not None else 0.0
        st["delay"] = delay
        self.maxdelay[i] = max(self.maxdelay[i], delay)
        win.append(delay)
        if len(win) > c.qmr_window:
            win.pop(0)
        avg = sum(win) / len(win)
        sd = (sum((w - avg) ** 2 for w in win) / len(win)) ** 0.5
        st["a"] = 0.3 if sd == 0 else min(1.0, 1.0 - np.exp(-abs(delay - avg) / sd))
        reward = 1.0 if (delivered or j == d) else c.qmr_w * np.exp(-delay / self.maxdelay[i]) + (1 - c.qmr_w) * e_j
        self._update(i, d, j, float(reward))

    def on_attempt_fail(self, i, packet, j, t):
        if not self._learning():
            return
        k = (i, packet.pid, j)
        self.fails[k] = self.fails.get(k, 0) + 1
        if self.fails[k] >= self.cfg.qmr_not_ack:
            self.fails.pop(k, None)
            self._update(i, packet.dst, j, -1.0)

    def on_dn(self, i, packet, j):
        if self._learning():
            self._update(i, packet.dst, j, -1.0)

    def on_step_end(self, t):
        if t % self.cfg.qmr_r_update:
            return
        sim = self.sim; exp_ = self.cfg.neighbor_expiry
        for i in range(self.n):
            cur = {j for j, e in sim.nbr[i].items() if t - e.last_seen <= exp_}
            prev = self.prev_set[i]
            add = len(cur - prev); dele = len(prev - cur); old = len(prev)
            self.r[i] = 0.0 if old + add == 0 else 1.0 - (add + dele) / (old + add)
            self.prev_set[i] = cur
        if len(self.fails) > 50000:
            self.fails.clear()

    def begin_episode(self, training, k, total_training, rng):
        super().begin_episode(training, k, total_training, rng)
        self.prev_set = [set() for _ in range(self.n)]; self.r[:] = 0.0; self.fails = {}

    def state_dict(self):
        import copy
        return {"name": self.name, "Q": copy.deepcopy(self.Q), "nb": copy.deepcopy(self.nb),
                "remote": copy.deepcopy(self.remote), "maxdelay": self.maxdelay.copy(), "updates": self.updates}

    def load_state_dict(self, d):
        self.Q = d["Q"]; self.nb = d["nb"]; self.remote = d["remote"]
        self.maxdelay = np.array(d["maxdelay"]); self.updates = int(d["updates"])

    def memory_bytes(self):
        return 8 * sum(len(q) for q in self.Q) + 8 * 8 * sum(len(x) for x in self.nb)


METHODS = {c.name: c for c in (GeoGreedy, GeoGreedyFresh, RandomQuadrant, CostHeuristic, TabularHist, LinearQ, GPSR, QMR)}


def make_policy(name: str, cfg: Settings) -> Policy:
    if name not in METHODS:
        raise ValueError(f"unknown method {name}; choose from {sorted(METHODS)}")
    return METHODS[name](cfg)
