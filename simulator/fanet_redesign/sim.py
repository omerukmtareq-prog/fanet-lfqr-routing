"""Redesigned FANET routing simulator (v1, pilot02).

Packet-level, step-driven, with a keyed physical channel, lossy beacons,
ACK-based link feedback, drop notifications and exactly-once accounting.
See design/MODEL_AND_METHOD.md.

pilot02 separation of concerns (Codex receipt 0003, item 3): protocol code
(nodes' forwarding, ACK, DN and dedup behaviour) acts only on node-local state
(own queue, own duplicate cache `seen`, own neighbour table, received frames).
Unique-packet accounting lives in `Observer`, which the protocol only notifies;
no protocol branch reads observer state. tests.py verifies this by running the
simulator with an observer whose internal state is scrambled after every call
and checking that every transmitted frame is unchanged.
"""
from __future__ import annotations
import collections, dataclasses, time
import numpy as np
from .config import Settings, STEP_SECONDS
from .channel import Channel, Frame, DATA, BEACON, ACK, DN
from .features import candidate_features, MASKS
from .traces import Trace
from . import wire


@dataclasses.dataclass
class Packet:
    pid: int
    src: int
    dst: int
    dst_pos: np.ndarray
    generated: int
    expires: int
    visited: tuple = ()
    hops: int = 0
    prev_hop: int = -1
    t_enq: int = 0
    attempts: int = 0
    target: int = -1
    target_attempts: int = 0
    dst_pos_step: int = 0
    hop_t0: int = -1          # step of the first transmission attempt at the current hop
    gpsr: tuple = ()          # GPSR perimeter state (mode, Lp, Lf, e0) carried in the header


@dataclasses.dataclass
class NeighborEntry:
    pos: np.ndarray
    vel: np.ndarray
    queue: float
    energy: float
    relay_ok: bool
    last_seen: int


class Observer:
    """Passive evaluator accounting (write-only from the protocol's view).

    A packet is terminal when (a) its destination first receives it
    (delivered) or (b) the last live copy anywhere disappears without delivery
    (lost, with the reason of that last removal). Removals of copies while
    another copy lives, or after the packet is terminal, are counted as stale
    copy removals and never create a second terminal event."""

    def __init__(self):
        self.generated = 0
        self.terminal = set(); self.delivered_pids = set()
        self.loss = collections.Counter(); self.delays = []; self.hops = []
        self.copies = collections.Counter()
        self.stale_discards = 0; self.duplicate_deliveries = 0

    def on_generate(self, p):
        self.generated += 1

    def on_reject_at_source(self, p, reason):
        self._finish(p, reason)

    def on_copy_added(self, p):
        self.copies[p.pid] += 1

    def on_copy_removed(self, p, reason, drop):
        self.copies[p.pid] -= 1
        if self.copies[p.pid] < 0:
            raise AssertionError(f"negative copy count for packet {p.pid}")
        if self.copies[p.pid] == 0 and p.pid not in self.terminal:
            self._finish(p, reason)
        elif drop:
            self.stale_discards += 1

    def on_delivery(self, p, t):
        if p.pid in self.terminal:
            # only reachable if the destination's own cache was pruned
            self.duplicate_deliveries += 1
            return
        self.terminal.add(p.pid); self.delivered_pids.add(p.pid)
        self.delays.append(t - p.generated); self.hops.append(p.hops + 1)

    def _finish(self, p, reason):
        if p.pid in self.terminal:
            raise AssertionError(f"duplicate terminal event for packet {p.pid}: {reason}")
        self.terminal.add(p.pid)
        self.loss[reason] += 1


class Simulator:
    def __init__(self, cfg: Settings, policy, trace: Trace, channel_seed: int, training: bool, policy_rng):
        self.cfg, self.policy, self.trace, self.training = cfg, policy, trace, training
        self.channel = Channel(cfg, channel_seed)
        self.rng = policy_rng
        n = cfg.nodes
        self.n = n
        self.alive = np.ones(n, dtype=bool)
        self.relay_ok = np.ones(n, dtype=bool) if cfg.all_relay_ok else (np.arange(n) % 4 != 0)
        self.queues = [collections.deque() for _ in range(n)]
        self.nbr = [dict() for _ in range(n)]
        self.linkq = [dict() for _ in range(n)]
        self.seen = [dict() for _ in range(n)]     # node-local duplicate cache: pid -> step received
        self.dn_outbox = [[] for _ in range(n)]     # (prev_hop, packet)
        self.e_tx = np.zeros(n); self.e_rx = np.zeros(n); self.e_idle = np.zeros(n)
        self.alive_steps = np.zeros(n)
        self.r_nom = cfg.nominal_range
        self.mask = MASKS[cfg.feature_mask]
        self.obs = Observer()
        # frame sizes (wire.py)
        self.bits_beacon = wire.beacon_bits(n)
        self.bits_ack = wire.ack_bits(n, policy.uses_value, getattr(policy, "ack_extra_bits", 0))
        self.bits_dn = wire.dn_bits(n)
        self.bits_loc_frame = wire.location_frame_bits(n)
        if cfg.location_service not in ("periodic", "oracle"):
            raise ValueError(f"unknown location service {cfg.location_service}")
        self.loc_tx_s = np.zeros(n); self.loc_rx_s = np.zeros(n); self.location_bits = 0
        self.loc_seen = np.zeros(n, dtype=int)     # last refresh epoch at which each node's record was issued
        self.qwait = np.zeros(n)                   # node-local EWMA of own queueing delay (steps)
        if cfg.queue_policy not in ("fifo", "slack"):
            raise ValueError(f"unknown queue policy {cfg.queue_policy}")
        policy.sim = self                          # policies may read node-local state of the deciding node only
        # protocol statistics (not used for decisions)
        self.attempts_total = 0; self.decisions = 0; self.decision_ns = 0
        self.frames_bits = collections.Counter(); self.frames_count = collections.Counter()
        self.frames_decoded = collections.Counter(); self.header_bits = 0
        self.max_queue = 0; self.failure_applied = False
        self.duplicate_receptions = 0; self.refusals = 0; self.deferrals = collections.Counter()
        self.frame_log = None   # tests set a list to record every transmitted frame

    # ------------------------------------------------------------- helpers
    def schedule_dn(self, j: int, p: Packet):
        if p.prev_hop >= 0:
            self.dn_outbox[j].append((p.prev_hop, p))

    def candidates(self, i: int, p: Packet, t: int):
        cfg = self.cfg
        table = self.nbr[i]
        for j in [j for j, e in table.items() if t - e.last_seen > cfg.neighbor_expiry]:
            del table[j]
        if self.policy.candidate_mode == "gpsr":
            # GPSR needs the complete (planarisable) neighbour set, including earlier hops
            return sorted(j for j, e in table.items() if j != i and (e.relay_ok or j == p.dst or cfg.feasibility != "pre"))
        visited = set(p.visited); visited.add(i)
        if p.prev_hop >= 0: visited.add(p.prev_hop)
        cands = []
        for j, e in table.items():
            if j in visited:
                continue
            if cfg.feasibility == "pre" and not (e.relay_ok or j == p.dst):
                continue
            cands.append(j)
        if p.dst in cands:
            return [p.dst]
        if self.policy.candidate_mode == "quadrant" and cands:
            pos_i = self.trace.pos[t][i]
            chosen = {}
            for j in cands:
                dx, dy = table[j].pos - pos_i
                key = (dx >= 0, dy >= 0); dist = dx * dx + dy * dy
                if key not in chosen or (dist, -j) > chosen[key][0]:
                    chosen[key] = ((dist, -j), j)
            cands = [chosen[k][1] for k in sorted(chosen)]
        return sorted(cands)

    def local_drop(self, i: int, p: Packet, reason: str, t: int):
        """Node i drops its own copy (already removed from its queue). The
        node's reaction is purely local: its policy is told, and unless the
        node failed or the episode ended it notifies its previous hop (DN)
        whether or not other copies exist elsewhere (it cannot know)."""
        self.obs.on_copy_removed(p, reason, drop=True)
        self.policy.on_local_drop(i, p)
        if reason not in ("failed_queue", "horizon", "dest_unreachable"):
            self.schedule_dn(i, p)

    def location_of(self, dst: int, t: int):
        """Destination position known to every node through the location
        service: the position at the last refresh epoch (periodic) or the true
        position at generation time (oracle control, pilot01 behaviour)."""
        if self.cfg.location_service == "periodic":
            e = int(self.loc_seen[dst])           # a failed node stops refreshing its record
            return self.trace.pos[e][dst], e
        return self.trace.pos[t][dst], t

    def dest_stale(self, dst: int, t: int) -> bool:
        """Location-record age exceeds one refresh period: the destination
        missed a refresh (failed or disconnected from the location service)."""
        if self.cfg.location_service != "periodic":
            return False
        return t - int(self.loc_seen[dst]) > self.cfg.location_period

    def est_hops(self, i: int, p: Packet, t: int) -> int:
        if p.dst in self.nbr[i]:
            return 1
        d = float(np.linalg.norm(self.trace.pos[t][i] - p.dst_pos))
        return max(1, int(np.ceil(d / self.r_nom)))

    def pick_index(self, i: int, q, t: int) -> int:
        """Queue discipline: FIFO, or least slack (TTL left minus estimated hops)."""
        if self.cfg.queue_policy == "fifo" or len(q) == 1:
            return 0
        best, bi = None, 0
        for k, p in enumerate(q):
            sl = (p.expires - t) - self.est_hops(i, p, t)
            if best is None or sl < best:
                best, bi = sl, k
        return bi

    def location_epoch(self, t: int):
        """Analytic cost of one location-table refresh (declared abstraction):
        every alive node broadcasts one aggregated table (one record per node);
        every alive node within nominal range of the sender receives it. Bits,
        TX time and RX time are charged; contention/interference of these
        frames is not simulated. The busy time is spread uniformly over the
        epoch's steps."""
        cfg = self.cfg
        per = cfg.location_period
        alive = self.alive
        pos = self.trace.pos[t]
        dur = self.bits_loc_frame / cfg.bit_rate            # seconds
        d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=2)
        hear = (d <= self.r_nom) & alive[:, None] & alive[None, :]
        np.fill_diagonal(hear, False)
        self.loc_tx_s = np.where(alive, dur, 0.0) / per
        self.loc_rx_s = hear.sum(axis=0) * dur / per
        self.loc_seen[alive] = t
        self.location_bits += int(alive.sum()) * self.bits_loc_frame

    # ------------------------------------------------------------- one step
    def step(self, t: int):
        cfg, pol = self.cfg, self.policy
        pos, vel = self.trace.pos[t], self.trace.vel[t]
        if cfg.failure_ratio and t == cfg.failure_step and not self.failure_applied:
            for j in self.trace.failed:
                self.alive[int(j)] = False
                while self.queues[int(j)]:
                    self.local_drop(int(j), self.queues[int(j)].popleft(), "failed_queue", t)
                self.dn_outbox[int(j)].clear()
            self.failure_applied = True
        if cfg.location_service == "periodic" and t % cfg.location_period == 0:
            self.location_epoch(t)
        frames = []
        # beacons
        for i in np.nonzero(self.alive)[0]:
            if (t + self.channel.beacon_phase[i]) % cfg.beacon_period == 0:
                rec = NeighborEntry(pos[i].copy(), vel[i].copy(), len(self.queues[i]) / cfg.buffer,
                                    self.residual(i), bool(self.relay_ok[i]), t)
                frames.append(Frame(int(i), -1, BEACON, self.bits_beacon, rec))
        # drop notifications scheduled earlier
        for j in range(self.n):
            if self.alive[j] and self.dn_outbox[j] and cfg.use_drop_notification:
                prev, p = self.dn_outbox[j].pop(0)
                # no liveness oracle: a DN to a failed node is sent and simply not decoded
                frames.append(Frame(j, prev, DN, self.bits_dn, p))
            elif not cfg.use_drop_notification:
                self.dn_outbox[j].clear()
        # data: one attempt per node
        for i in range(self.n):
            if not self.alive[i]:
                continue
            q = self.queues[i]
            while q:
                k = self.pick_index(i, q, t)
                p = q[k]
                if t >= p.expires:
                    del q[k]; self.local_drop(i, p, "expired", t); continue
                if cfg.location_service == "periodic":
                    p.dst_pos, p.dst_pos_step = self.location_of(p.dst, t)
                if cfg.prune_stale_dest and self.dest_stale(p.dst, t):
                    del q[k]; self.local_drop(i, p, "dest_unreachable", t); continue
                if cfg.drop_infeasible and (p.expires - t) < self.est_hops(i, p, t):
                    del q[k]; self.local_drop(i, p, "infeasible", t); continue
                cands = self.candidates(i, p, t)
                if not cands:
                    del q[k]; self.local_drop(i, p, "no_route", t); continue
                feats = candidate_features(cfg, pos[i], vel[i], self.nbr[i], cands, self.linkq[i], p, t, self.r_nom, self.mask)
                if p.target in cands and 0 < p.target_attempts < cfg.retry_pin:
                    j = p.target   # retry pinned to the same next hop
                else:
                    t0 = time.perf_counter_ns()
                    j = pol.choose(i, p, cands, feats, t, self.training)
                    self.decision_ns += time.perf_counter_ns() - t0; self.decisions += 1
                    p.target, p.target_attempts = (j if j is not None else -1), 0
                if j is None:
                    del q[k]; self.local_drop(i, p, "no_route", t); continue
                if j not in cands:
                    raise AssertionError(f"policy {pol.name} chose {j} outside candidates {cands}")
                if cfg.feasibility == "post" and j != p.dst and not self.nbr[i][j].relay_ok:
                    del q[k]; self.local_drop(i, p, "ineligible_relay", t); continue
                bits = wire.data_frame_bits(cfg.payload_bits, self.n, cfg.max_visited, len(p.visited))
                if pol.candidate_mode == "gpsr":
                    bits += wire.gpsr_header_bits(self.n)
                frames.append(Frame(i, j, DATA, bits, (p, feats[cands.index(j)])))
                break
        decoded, deferred, tx_e, rx_e = self.channel.resolve(frames, pos, self.alive, t, 0)
        self.e_tx += tx_e; self.e_rx += rx_e
        busy = tx_e / cfg.p_tx_w + rx_e / cfg.p_rx_w
        acks = []
        for f, row, dfr in zip(frames, decoded, deferred):
            if dfr:
                self.deferrals[f.kind] += 1
                if f.kind == DN:
                    self.dn_outbox[f.sender].insert(0, (f.receiver, f.payload))
                continue
            self.frames_bits[f.kind] += f.bits; self.frames_count[f.kind] += 1
            if self.frame_log is not None:
                pl = f.payload[0] if f.kind == DATA else f.payload
                self.frame_log.append((t, f.kind, f.sender, f.receiver, getattr(pl, "pid", -1), f.bits, round(f.offset, 9)))
            if f.kind == DATA:
                p, feat = f.payload
                self.header_bits += f.bits - cfg.payload_bits - wire.MAC_BITS
                if p.hop_t0 < 0:
                    p.hop_t0 = t
                    self.qwait[f.sender] = 0.8 * self.qwait[f.sender] + 0.2 * (t - p.t_enq)
                p.attempts += 1; p.target_attempts += 1; self.attempts_total += 1
                pol.on_forward(f.sender, p, f.receiver, feat, t)
                f.payload = p
            if f.kind == BEACON:
                for r in np.nonzero(row)[0]:
                    r = int(r)
                    prev = self.nbr[r].get(f.sender)
                    if prev is not None:
                        # beacon periods missed since the last one heard count as failures
                        missed = max(0, round((t - prev.last_seen) / cfg.beacon_period) - 1)
                        for _ in range(min(missed, 3)):
                            self.link_feedback(r, f.sender, False)
                    self.link_feedback(r, f.sender, True)
                    self.nbr[r][f.sender] = f.payload
                self.frames_decoded[BEACON] += int(row.sum())
            elif f.kind == DN:
                if row[f.receiver]:
                    self.frames_decoded[DN] += 1
                    pol.on_dn(f.receiver, f.payload, f.sender)
            else:
                i, j, p = f.sender, f.receiver, f.payload
                if not row[j]:
                    self.link_feedback(i, j, False)
                    pol.on_attempt_fail(i, p, j, t)
                    continue
                self.frames_decoded[DATA] += 1
                delivered, refused, value = self.receive(i, j, p, t)
                acks.append((Frame(j, i, ACK, self.bits_ack, (p, delivered, refused, value)), j))
        ack_frames = [a for a, _ in acks]
        decoded2, deferred2, tx_e, rx_e = self.channel.resolve(ack_frames, pos, self.alive, t, 1)
        self.e_tx += tx_e; self.e_rx += rx_e
        busy = busy + tx_e / cfg.p_tx_w + rx_e / cfg.p_rx_w
        for f, row, dfr in zip(ack_frames, decoded2, deferred2):
            i, j = f.receiver, f.sender
            p, delivered, refused, value = f.payload
            if dfr:
                self.deferrals[ACK] += 1; self.link_feedback(i, j, False); pol.on_attempt_fail(i, p, j, t); continue
            self.frames_bits[ACK] += f.bits; self.frames_count[ACK] += 1
            if self.frame_log is not None:
                self.frame_log.append((t, ACK, j, i, p.pid, f.bits, round(f.offset, 9)))
            if not row[i]:
                self.link_feedback(i, j, False); pol.on_attempt_fail(i, p, j, t); continue
            self.frames_decoded[ACK] += 1
            self.link_feedback(i, j, True)
            q = self.queues[i]
            k = next((k for k, x in enumerate(q) if x.pid == p.pid), None)
            if k is not None:
                del q[k]
                # observer: if this was the last copy and the receiver did not
                # take it (refused, or acknowledged from its duplicate cache
                # without holding a copy) the packet is lost
                self.obs.on_copy_removed(p, "buffer_overflow" if refused else "dedup_loss", drop=refused)
            cost = (t - p.t_enq) + cfg.lambda_energy * p.attempts
            e_j = self.nbr[i][j].energy if j in self.nbr[i] else 1.0
            pol.on_ack(i, p, j, delivered, float(cost), value, float(e_j), refused=refused, t=t)
        # queue TTL sweep and bookkeeping
        for i in range(self.n):
            if not self.alive[i]:
                continue
            q = self.queues[i]
            if q:
                keep = collections.deque()
                for p in q:
                    if t >= p.expires:
                        self.local_drop(i, p, "expired", t)
                    else:
                        keep.append(p)
                self.queues[i] = q = keep
            self.max_queue = max(self.max_queue, len(q))
            if t % 50 == 0:
                s = self.seen[i]
                for pid in [pid for pid, ts in s.items() if t - ts[0] > 2 * cfg.ttl]:
                    del s[pid]
        # traffic
        if t < cfg.steps:
            for pid, src, dst in self.trace.traffic[t]:
                dpos, dstep = self.location_of(dst, t)
                p = Packet(pid, src, dst, np.array(dpos, dtype=float), t, t + cfg.ttl, (), 0, -1, t, 0, dst_pos_step=dstep)
                self.obs.on_generate(p)
                if not self.alive[src]:
                    self.obs.on_reject_at_source(p, "failed_source")
                elif cfg.prune_stale_dest and self.dest_stale(dst, t):
                    self.obs.on_reject_at_source(p, "dest_unreachable")
                elif len(self.queues[src]) >= cfg.buffer:
                    self.obs.on_reject_at_source(p, "source_overflow")
                else:
                    self.queues[src].append(p); self.seen[src][pid] = (t, -1, 0); self.obs.on_copy_added(p)
        # radio-state energy for this step: TX/RX from both sub-slots plus the
        # analytic location-service share; the remainder of the step is IDLE
        loc_tx = np.where(self.alive, self.loc_tx_s, 0.0); loc_rx = np.where(self.alive, self.loc_rx_s, 0.0)
        self.e_tx += loc_tx * cfg.p_tx_w; self.e_rx += loc_rx * cfg.p_rx_w
        busy = busy + loc_tx + loc_rx
        idle = np.maximum(STEP_SECONDS - busy, 0.0) * cfg.p_idle_w
        self.e_idle += np.where(self.alive, idle, 0.0)
        self.alive_steps += self.alive
        if hasattr(pol, "expire_pending"):
            pol.expire_pending(t)
        pol.on_step_end(t)

    def link_feedback(self, i, j, ok):
        lq = self.linkq[i]
        lq[j] = 0.8 * lq.get(j, 0.8) + 0.2 * (1.0 if ok else 0.0)

    def residual(self, i):
        used = self.e_tx[i] + self.e_rx[i] + self.e_idle[i]
        return float(max(0.0, 1.0 - used / self.cfg.energy_budget_j))

    def receive(self, i: int, j: int, p: Packet, t: int):
        """Node j decodes a data frame from i. Uses only j's own state.
        Returns (delivered, refused, value_for_ack)."""
        cfg, pol = self.cfg, self.policy
        if j == p.dst:
            if p.pid in self.seen[j]:
                self.duplicate_receptions += 1          # destination's own dedup cache
            else:
                self.seen[j][p.pid] = (t, i, p.hops)
                self.obs.on_delivery(p, t)
            return True, False, (0.0 if pol.uses_value else None)
        # j works on its own copy; i keeps its object untouched until the ACK
        pj = dataclasses.replace(p, visited=(p.visited + (i,))[-cfg.max_visited:], hops=p.hops + 1,
                                 prev_hop=i, t_enq=t, attempts=0, target=-1, target_attempts=0, hop_t0=-1)
        prior = self.seen[j].get(p.pid)
        if prior is not None and (self.policy.candidate_mode != "gpsr" or (prior[1] == i and prior[2] == p.hops)
                                  or any(x.pid == p.pid for x in self.queues[j])):
            # duplicate of the same hop transmission (our earlier ACK was lost): re-acknowledge, do not enqueue.
            # A later arrival over a different hop (e.g. a GPSR perimeter revisit) is a new reception.
            self.duplicate_receptions += 1
            cands = self.candidates(j, pj, t)
            return False, False, self.ack_value(j, pj, cands, t)
        if len(self.queues[j]) >= cfg.buffer:
            # refuse without caching the pid, so a later retry can still be accepted
            self.refusals += 1
            return False, True, (float(cfg.c_drop) if pol.uses_value else None)
        self.seen[j][p.pid] = (t, i, p.hops)
        self.queues[j].append(pj); self.obs.on_copy_added(pj)
        cands = self.candidates(j, pj, t)
        return False, False, self.ack_value(j, pj, cands, t)

    def ack_value(self, j, p, cands, t):
        pol = self.policy
        if not pol.uses_value:
            return None
        pos, vel = self.trace.pos[t], self.trace.vel[t]
        feats = candidate_features(self.cfg, pos[j], vel[j], self.nbr[j], cands, self.linkq[j], p, t, self.r_nom, self.mask)
        return pol.value(j, p, cands, feats)

    # ------------------------------------------------------------- episode
    def run(self):
        cfg = self.cfg
        started = time.perf_counter()
        horizon = cfg.steps + cfg.ttl + 1
        for t in range(horizon):
            self.step(t)
            if t >= cfg.steps and not any(self.queues):
                break
        for i, q in enumerate(self.queues):
            while q:
                self.local_drop(i, q.popleft(), "horizon", horizon)
        o = self.obs
        losses = sum(o.loss.values())
        assert o.generated == len(o.delays) + losses == len(o.terminal), (o.generated, len(o.delays), losses)
        assert all(c == 0 for c in o.copies.values()), "live copies left after the horizon"
        return self.summary(time.perf_counter() - started, t)

    def summary(self, elapsed, last_t):
        cfg, o = self.cfg, self.obs
        n_del = len(o.delays)
        active = self.e_tx + self.e_rx
        total = active + self.e_idle
        seconds = (last_t + 1) * STEP_SECONDS
        alive_s = np.maximum(self.alive_steps, 1) * STEP_SECONDS
        rate_total = total / alive_s          # W, per node over its own alive time
        rate_active = active / alive_s
        payload_bits = n_del * cfg.payload_bits
        control_bits = self.frames_bits[BEACON] + self.frames_bits[ACK] + self.frames_bits[DN] + self.location_bits
        info_bits = control_bits + self.header_bits + wire.MAC_BITS * self.frames_count[DATA]
        delays = np.array(o.delays, dtype=float)
        censored = np.concatenate([delays, np.full(o.generated - n_del, float(cfg.ttl))]) if o.generated else delays
        pol = self.policy
        return dict(
            method=pol.name, training=self.training, trace_key=self.trace.key, trace_sha256=self.trace.sha256,
            channel_seed=self.channel.seed, steps_run=int(last_t + 1), seconds_run=float(seconds),
            generated=int(o.generated), delivered=int(n_del), pdr=n_del / o.generated if o.generated else 0.0,
            delay_mean=float(delays.mean()) if n_del else None, delay_p95=float(np.quantile(delays, 0.95)) if n_del else None,
            delay_censored_mean=float(censored.mean()) if o.generated else None,
            hops_mean=float(np.mean(o.hops)) if n_del else None,
            attempts_per_delivered=self.attempts_total / n_del if n_del else None, attempts_total=int(self.attempts_total),
            losses=dict(o.loss),
            energy_tx_j=float(self.e_tx.sum()), energy_rx_j=float(self.e_rx.sum()), energy_idle_j=float(self.e_idle.sum()),
            energy_active_j=float(active.sum()), energy_total_j=float(total.sum()),
            energy_total_per_offered_j=float(total.sum() / o.generated) if o.generated else None,
            energy_total_per_delivered_j=float(total.sum() / n_del) if n_del else None,
            energy_active_per_offered_j=float(active.sum() / o.generated) if o.generated else None,
            energy_active_per_delivered_j=float(active.sum() / n_del) if n_del else None,
            # first-node-depletion EXTRAPOLATION from a short horizon, not a measured lifetime
            lifetime_projection_s=float(cfg.energy_budget_j / rate_total.max()) if rate_total.max() > 0 else None,
            lifetime_active_projection_s=float(cfg.energy_budget_j / rate_active.max()) if rate_active.max() > 0 else None,
            load_balance_max_over_mean=float(rate_active.max() / rate_active.mean()) if rate_active.mean() > 0 else None,
            control_bits=int(control_bits), location_bits=int(self.location_bits), data_header_bits=int(self.header_bits),
            payload_bits_delivered=int(payload_bits),
            control_overhead_ratio=control_bits / payload_bits if payload_bits else None,
            information_overhead_ratio=info_bits / payload_bits if payload_bits else None,
            frames=dict(self.frames_count), frames_decoded=dict(self.frames_decoded),
            frame_bits=dict(self.frames_bits),
            decisions=int(self.decisions), decision_us_mean=self.decision_ns / self.decisions / 1e3 if self.decisions else None,
            max_queue=int(self.max_queue), epsilon=float(getattr(pol, "epsilon", 0.0)),
            stale_discards=int(o.stale_discards), duplicate_receptions=int(self.duplicate_receptions),
            duplicate_deliveries=int(o.duplicate_deliveries), refusals=int(self.refusals),
            deferrals=dict(self.deferrals),
            policy_updates=int(getattr(pol, "updates", 0)), policy_memory_bytes=int(pol.memory_bytes()),
            learner_mean_cost=(pol.cost_sum / pol.cost_n) if getattr(pol, "cost_n", 0) else None,
            location_service=cfg.location_service,
            elapsed_seconds=float(elapsed), settings_sha256=cfg.sha256(),
        )


def simulate(cfg: Settings, policy, trace: Trace, channel_seed: int, training: bool, policy_seed: int,
             episode_index: int = 0, total_training: int = 1):
    rng = np.random.default_rng(np.random.SeedSequence([policy_seed]))
    policy.begin_episode(training, episode_index, total_training, rng)
    if hasattr(policy, "cost_sum"):
        policy.cost_sum = 0.0; policy.cost_n = 0
    if hasattr(policy, "pending"):
        policy.pending.clear()
    sim = Simulator(cfg, policy, trace, channel_seed, training, rng)
    return sim.run()
