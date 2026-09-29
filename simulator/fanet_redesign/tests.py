"""Correctness tests (numpy only). Run: python -m fanet_redesign test

Each test returns nothing and raises on failure. Tests cover the contract
list: packet conservation, exactly-once delivery/drop accounting, bounded
queues and TTL, terminal credit semantics, feedback loss/delay,
transmission/energy units, valid actions, deterministic replay, frozen
evaluation, split separation, checkpoint resume and interrupted writes,
plus tiny networks with known behaviour and disconnected routes.
"""
from __future__ import annotations
import json, os, pickle, shutil, tempfile, time, traceback
from pathlib import Path
import numpy as np
from .config import Settings, STEP_SECONDS, trace_key
from .traces import make_trace, Trace
from .channel import Channel
from .policies import make_policy, METHODS, LinearQ, CostHeuristic, Policy
from .sim import Simulator, simulate, Packet, Observer
from . import run as R
from . import wire

SMALL = Settings(nodes=25, area=700.0, steps=40, ttl=30, load=0.05)


def static_trace(cfg: Settings, positions, key=424242, packets=()):
    """Trace with fixed positions and an explicit packet list [(t, src, dst)]."""
    n = cfg.nodes
    horizon = cfg.steps + cfg.ttl + 2
    pos = np.tile(np.asarray(positions, float)[None], (horizon, 1, 1))
    vel = np.zeros_like(pos)
    traffic = [[] for _ in range(cfg.steps)]
    pid = 0
    for t, s, d in packets:
        traffic[t].append((pid, s, d)); pid += 1
    return Trace(pos, vel, traffic, np.array([], int), key, "static")


def sim_for(cfg, method, trace, training=False, seed=1):
    pol = make_policy(method, cfg)
    rng = np.random.default_rng(seed); pol.begin_episode(training, 0, 1, rng)
    return Simulator(cfg, pol, trace, 5, training, rng), pol


# --------------------------------------------------------------------- tests
def test_line_network_delivers():
    """3 static nodes in a line, 200 m apart, no shadowing: greedy delivers in 2 hops."""
    # beacons need to be heard first: the packet is offered at t=25 (beacon period 10)
    cfg = Settings(nodes=3, area=600, steps=30, ttl=30, shadow_sigma_db=0.0, load=0.0)
    tr = static_trace(cfg, [[50, 300], [250, 300], [450, 300]], packets=[(25, 0, 2)])
    s, pol = sim_for(cfg, "geo_greedy", tr)
    r = s.run()
    assert r["generated"] == 1 and r["delivered"] == 1, r
    assert r["hops_mean"] == 2.0, r["hops_mean"]
    assert r["losses"] == {}, r["losses"]
    assert 2 <= r["delay_mean"] <= 4, r["delay_mean"]


def test_disconnected_route_is_no_route():
    cfg = Settings(nodes=3, area=3000, steps=30, ttl=30, shadow_sigma_db=0.0, load=0.0)
    tr = static_trace(cfg, [[50, 50], [250, 50], [2500, 2500]], packets=[(25, 0, 2)])
    for m in METHODS:
        s, _ = sim_for(cfg, m, tr)
        r = s.run()
        assert r["generated"] == 1 and r["delivered"] == 0, (m, r)
        assert set(r["losses"]) <= {"no_route", "expired"}, (m, r["losses"])
        assert sum(r["losses"].values()) == 1, (m, r)


def test_packet_conservation_and_exactly_once():
    for m in METHODS:
        for key in (trace_key("train", "S1", 0), trace_key("train", "S2", 1)):
            cfg = SMALL if "S1" in str(key) else Settings(**{**SMALL.to_dict(), "failure_ratio": 0.2, "failure_step": 20})
            tr = make_trace(cfg, key)
            r = simulate(cfg, make_policy(m, cfg), tr, 5, True, 3)
            assert r["generated"] == r["delivered"] + sum(r["losses"].values()), (m, r)
            assert r["generated"] == sum(len(e) for e in tr.traffic)


def test_bounded_queue_and_ttl():
    cfg = Settings(**{**SMALL.to_dict(), "buffer": 4, "load": 0.5, "ttl": 12})
    tr = make_trace(cfg, trace_key("train", "S3", 0))
    s, _ = sim_for(cfg, "random_quadrant", tr, training=True)
    horizon = cfg.steps + cfg.ttl + 1
    for t in range(horizon):
        s.step(t)
        for q in s.queues:
            assert len(q) <= cfg.buffer
            for p in q:
                assert p.expires > t, "expired packet left in queue"
    assert s.max_queue <= cfg.buffer
    assert s.obs.loss.get("buffer_overflow", 0) + s.obs.loss.get("source_overflow", 0) + s.refusals > 0, "overflow never exercised"
    assert s.obs.loss.get("expired", 0) > 0, "expiry never exercised"


def test_valid_actions_only():
    cfg = SMALL
    tr = make_trace(cfg, trace_key("train", "S1", 2))
    for m in METHODS:
        s, pol = sim_for(cfg, m, tr, training=True)
        orig = pol.choose
        def wrapped(i, packet, cands, feats, t, training, orig=orig):
            j = orig(i, packet, cands, feats, t, training)
            assert j is None or j in cands, (m, j, cands)
            assert all(c != i for c in cands)
            if pol.candidate_mode != "gpsr":     # GPSR perimeter mode may revisit nodes by design
                assert all(c not in packet.visited for c in cands)
            return j
        pol.choose = wrapped
        s.run()


def test_deterministic_replay():
    cfg = SMALL
    tr = make_trace(cfg, trace_key("train", "S1", 3))
    for m in ("geo_greedy", "lfa_q", "tabular_q_hist"):
        a = simulate(cfg, make_policy(m, cfg), tr, 5, True, 11)
        b = simulate(cfg, make_policy(m, cfg), tr, 5, True, 11)
        for k in ("pdr", "losses", "delivered", "energy_active_j", "attempts_total", "frames"):
            assert a[k] == b[k], (m, k, a[k], b[k])
    # a different policy seed changes tie-breaks but never the exogenous inputs
    c = simulate(cfg, make_policy("random_quadrant", cfg), tr, 5, True, 12)
    assert c["generated"] == a["generated"] and c["trace_sha256"] == a["trace_sha256"]


def test_channel_is_keyed_not_sequential():
    """The channel realisation must not depend on how many frames were sent."""
    cfg = SMALL
    ch = Channel(cfg, 77)
    s1 = ch.shadow(35).copy(); o1 = ch.offsets(35, 0).copy()
    ch2 = Channel(cfg, 77)
    _ = ch2.shadow(3); _ = ch2.offsets(4, 0); _ = ch2.offsets(35, 1)
    assert np.array_equal(s1, ch2.shadow(35)) and np.array_equal(o1, ch2.offsets(35, 0))
    assert np.allclose(s1, s1.T) and np.all(np.diag(s1) == 0)


def test_frozen_evaluation_does_not_learn():
    cfg = SMALL
    pol = make_policy("lfa_q", cfg)
    simulate(cfg, pol, make_trace(cfg, trace_key("train", "S1", 0)), 5, True, 1, 0, 2)
    theta = pol.theta.copy(); n_upd = pol.updates
    r = simulate(cfg, pol, make_trace(cfg, trace_key("tune", "S1", 0)), 5, False, 2)
    assert np.array_equal(theta, pol.theta) and pol.updates == n_upd and r["epsilon"] == 0.0
    pol2 = make_policy("tabular_q_hist", cfg)
    simulate(cfg, pol2, make_trace(cfg, trace_key("train", "S1", 0)), 5, True, 1)
    snap = pickle.dumps(pol2.state_dict()["q"])
    simulate(cfg, pol2, make_trace(cfg, trace_key("tune", "S1", 0)), 5, False, 2)
    assert pickle.dumps(pol2.state_dict()["q"]) == snap


def test_split_separation():
    keys = {trace_key(s, sc, i) for s in ("train", "tune", "final") for sc in ("S1", "S2", "S3") for i in range(50)}
    assert len(keys) == 3 * 3 * 50
    a = make_trace(SMALL, trace_key("train", "S1", 0)); b = make_trace(SMALL, trace_key("final", "S1", 0))
    assert a.sha256 != b.sha256 and not np.array_equal(a.pos[0], b.pos[0])
    cfg2 = Settings(**{**SMALL.to_dict(), "load": 0.06})
    c = make_trace(cfg2, trace_key("train", "S1", 0))
    assert np.array_equal(a.pos, c.pos), "mobility must not change when only traffic load changes"


def test_terminal_credit_semantics():
    """One ACK update and at most one DN update per forwarding decision; duplicates ignored."""
    cfg = Settings(nodes=4, alpha=0.5, lambda_energy=0.0, c_drop=100.0)
    pol = LinearQ(cfg); pol.begin_episode(True, 0, 1, np.random.default_rng(0))
    p = Packet(1, 0, 3, np.zeros(2), 0, 100)
    feat = np.zeros(10); feat[0] = 1.0
    pol.on_forward(0, p, 1, feat, 0)
    pol.on_ack(0, p, 1, False, cost=3.0, value=7.0, e_j=1.0)     # target 10 -> theta0 = 5
    assert pol.theta[0, 0] == 5.0 and pol.updates_ack == 1
    pol.on_ack(0, p, 1, False, cost=3.0, value=7.0, e_j=1.0)     # duplicate ACK ignored
    assert pol.theta[0, 0] == 5.0 and pol.updates_ack == 1
    pol.on_dn(0, p, 1)                                            # correction: target 100 -> theta0 = 52.5
    assert pol.theta[0, 0] == 52.5 and pol.updates_dn == 1
    pol.on_dn(0, p, 1)                                            # duplicate DN ignored
    assert pol.theta[0, 0] == 52.5 and pol.updates_dn == 1
    pol.on_local_drop(0, p)                                       # nothing pending: no update
    assert pol.updates == 2
    # delivered ACK is terminal: no bootstrap, pending removed
    pol.on_forward(0, p, 3, feat, 5); pol.on_ack(0, p, 3, True, cost=2.0, value=50.0, e_j=1.0)
    assert (0, 1) not in pol.pending and abs(pol.theta[0, 0] - (52.5 + 0.5 * (2.0 - 52.5))) < 1e-9
    # a forward that was never acknowledged and is then dropped locally gets c_drop
    q = Packet(2, 0, 3, np.zeros(2), 0, 100)
    pol.on_forward(0, q, 1, feat, 6); before = pol.theta[0, 0]; pol.on_local_drop(0, q)
    assert pol.updates_drop == 1 and abs(pol.theta[0, 0] - (before + 0.5 * (100.0 - before))) < 1e-9
    # frozen policy never updates
    pol.begin_episode(False, 0, 1, np.random.default_rng(0)); th = pol.theta.copy()
    pol.on_forward(0, q, 1, feat, 7); pol.on_ack(0, q, 1, False, 1.0, 1.0, 1.0); pol.on_dn(0, q, 1)
    assert np.array_equal(th, pol.theta)


def test_feedback_loss_and_delay():
    """A lost ACK causes a retransmission; the receiver de-duplicates and re-acknowledges."""
    cfg = Settings(nodes=2, area=600, steps=30, ttl=40, shadow_sigma_db=0.0, load=0.0)
    tr = static_trace(cfg, [[100, 100], [300, 100]], packets=[(25, 0, 1)])
    s, pol = sim_for(cfg, "lfa_q", tr, training=True)
    ch = s.channel; orig = ch.resolve; state = {"ack_calls": 0}
    def resolve(frames, pos, alive, t, subslot, orig=orig):
        dec, dfr, tx, rx = orig(frames, pos, alive, t, subslot)
        if subslot == 1 and frames:
            state["ack_calls"] += 1
            if state["ack_calls"] == 1:
                dec[:] = False      # first ACK lost
        return dec, dfr, tx, rx
    ch.resolve = resolve
    r = s.run()
    assert r["delivered"] == 1 and r["losses"] == {}, r
    assert r["attempts_total"] == 2 and r["duplicate_receptions"] == 1, (r["attempts_total"], r["duplicate_receptions"])
    assert r["frames"]["ack"] == 2
    assert pol.updates_ack == 1, "exactly one ACK-based update after the retry"


def test_transmission_and_energy_units():
    """One data frame + one ACK between two static nodes: energy matches the
    declared radio-state model and frame sizes match wire.py."""
    cfg = Settings(nodes=2, area=600, steps=30, ttl=40, shadow_sigma_db=0.0, load=0.0, beacon_period=10,
                   location_service="oracle")
    tr = static_trace(cfg, [[100, 100], [300, 100]], packets=[(25, 0, 1)])
    s, _ = sim_for(cfg, "geo_greedy", tr)
    r = s.run()
    steps = r["steps_run"]
    n_beacons = r["frames"]["beacon"]
    data_bits = wire.data_frame_bits(cfg.payload_bits, 2, cfg.max_visited, 0)
    assert r["frame_bits"]["data"] == data_bits and r["frame_bits"]["ack"] == wire.ack_bits(2, False)
    assert r["frame_bits"]["beacon"] == n_beacons * wire.beacon_bits(2)
    t_data = cfg.airtime_steps(data_bits) * STEP_SECONDS
    t_ack = cfg.airtime_steps(wire.ack_bits(2, False)) * STEP_SECONDS
    t_beacon = cfg.airtime_steps(wire.beacon_bits(2)) * STEP_SECONDS
    # every frame is sensed (and decoded) by the other node: its airtime is RX time there
    assert r["frames_decoded"]["beacon"] == n_beacons
    tx_expected = cfg.p_tx_w * (t_data + t_ack + n_beacons * t_beacon)
    rx_expected = cfg.p_rx_w * (t_data + t_ack + n_beacons * t_beacon)
    assert abs(r["energy_active_j"] - (tx_expected + rx_expected)) < 1e-9, (r["energy_active_j"], tx_expected + rx_expected)
    busy = 2 * (t_data + t_ack + n_beacons * t_beacon)
    idle_expected = cfg.p_idle_w * (2 * steps * STEP_SECONDS - busy)
    assert abs(r["energy_total_j"] - (tx_expected + rx_expected + idle_expected)) < 1e-9
    assert abs(t_data - data_bits / 11e6) < 1e-12 and abs(cfg.nominal_range - 10 ** ((10 - 40 + 95 - 10) / 22)) < 1e-9
    assert r["control_bits"] == n_beacons * wire.beacon_bits(2) + wire.ack_bits(2, False) and r["location_bits"] == 0


def test_receiver_state_model_counts_failed_reception_once():
    """RX is charged while any sensed frame is on air, decoded or not, counted
    once when frames overlap; TX takes precedence; dead nodes are charged nothing."""
    cfg = Settings(nodes=4, shadow_sigma_db=0.0)
    ch = Channel(cfg, 1)
    senders = np.array([0, 1, 3])
    start = np.array([0.10, 0.15, 0.30]); end = np.array([0.20, 0.25, 0.40])
    sent = np.array([True, True, True])
    p = np.zeros((3, 4)); p[:, :] = 1.0          # every frame sensed everywhere (>= cs threshold)
    alive = np.array([True, True, True, False])
    tx, rx = ch.radio_time(senders, start, end, sent, p, alive)
    S = STEP_SECONDS
    # node 2 hears [0.10,0.25] (two overlapping, collided frames: counted once) and node 3's frame (dead sender still on air here)
    assert abs(rx[2] - (0.15 + 0.10) * S) < 1e-12, rx[2]
    # node 0 transmits [0.10,0.20]; it hears node 1 only in [0.20,0.25] and node 3's frame [0.30,0.40]
    assert abs(tx[0] - 0.10 * S) < 1e-12 and abs(rx[0] - (0.05 + 0.10) * S) < 1e-12, (tx[0], rx[0])
    assert tx[3] == 0 and rx[3] == 0, "dead node charged"
    # below carrier-sense threshold nothing is charged
    tx2, rx2 = ch.radio_time(senders, start, end, sent, np.zeros((3, 4)), alive)
    assert rx2.sum() == 0 and abs(tx2[1] - 0.10 * S) < 1e-12


def test_lifetime_uses_total_radio_energy():
    """Two disconnected nodes, no payload (Codex diagnostic): the projection
    must reflect the idle draw (about 500 J / 1.15 W), not active energy only."""
    cfg = Settings(nodes=2, area=3000, steps=30, ttl=30, shadow_sigma_db=0.0, load=0.0, location_service="oracle")
    tr = static_trace(cfg, [[50, 50], [2900, 2900]])
    s, _ = sim_for(cfg, "geo_greedy", tr); r = s.run()
    assert 400 < r["lifetime_projection_s"] < 435, r["lifetime_projection_s"]
    assert r["lifetime_active_projection_s"] > 1e5


def test_evaluation_traces_distinct_across_replicates():
    """Codex receipt 0003 item 2: eval traces differ between replicates, are
    paired across methods within a replicate, and no key collides across
    splits, scenarios, replicates or episodes."""
    from .config import SCENARIO_CODE
    seen = {}
    for sc in SCENARIO_CODE:
        for rep in range(10):
            plans = {m: R.job_plan(m, sc, rep, 50, 5, "final") for m in ("geo_greedy", "lfa_q", "cost_heuristic")}
            ev = [e["trace"] for e in plans["lfa_q"] if e["phase"] == "eval"]
            assert ev == [e["trace"] for e in plans["geo_greedy"]] == [e["trace"] for e in plans["cost_heuristic"]]
            for split in ("tune", "final"):
                for e in R.job_plan("lfa_q", sc, rep, 50, 5, split):
                    tag = (sc, rep, e["split"], e["index"])
                    if e["trace"] in seen:
                        assert seen[e["trace"]] == tag, ("key collision", e["trace"], seen[e["trace"]], tag)
                    seen[e["trace"]] = tag
    assert len({R.job_plan("geo_greedy", "S1", r, 0, 5, "final")[0]["trace"] for r in range(10)}) == 10
    for bad in (lambda: trace_key("train", "S9", 0), lambda: trace_key("train", "S1", 100_000)):
        try:
            bad(); raise AssertionError("invalid trace key accepted")
        except ValueError:
            pass


class _ChaosObserver(Observer):
    """Scrambles its own state after every notification. If any protocol
    decision read observer state, the frame log would change."""
    def __init__(self, seed=0):
        super().__init__(); self._r = np.random.default_rng(seed)
    def _scramble(self):
        self.terminal = set(int(x) for x in self._r.integers(0, 50, 10))
        self.copies = __import__("collections").Counter({int(k): int(self._r.integers(-3, 4)) for k in range(50)})
    def on_generate(self, p): self.generated += 1; self._scramble()
    def on_reject_at_source(self, p, reason): self._scramble()
    def on_copy_added(self, p): self._scramble()
    def on_copy_removed(self, p, reason, drop): self._scramble()
    def on_delivery(self, p, t): self._scramble()


def test_observer_state_cannot_change_protocol_actions():
    """Codex receipt 0003 item 3: remote/global observer state alone cannot
    change any node's transmissions (data, ACK, DN, beacon), for every method,
    including failures, lost ACKs and duplicate copies."""
    cfg = Settings(**{**SMALL.to_dict(), "failure_ratio": 0.2, "failure_step": 20, "load": 0.12, "buffer": 6})
    tr = make_trace(cfg, trace_key("train", "S2", 4))
    for m in METHODS:
        logs = []
        for chaos in (False, True):
            pol = make_policy(m, cfg)
            rng = np.random.default_rng(3); pol.begin_episode(True, 0, 2, rng)
            s = Simulator(cfg, pol, tr, 5, True, rng)
            s.frame_log = []
            if chaos:
                s.obs = _ChaosObserver()
            for t in range(cfg.steps + cfg.ttl + 1):
                s.step(t)
            logs.append(s.frame_log)
        assert logs[0] == logs[1], f"{m}: observer state changed protocol behaviour"
        kinds = {k for _, k, *_ in logs[0]}
        assert {"data", "ack", "beacon"} <= kinds, (m, kinds)


class _SwitchPolicy(Policy):
    """Test policy: first choice relay 1; after the pinned retries to 1 fail, relay 2."""
    name = "switch_test"
    def choose(self, i, packet, cands, feats, t, training):
        if packet.dst in cands:
            return packet.dst
        if i == 0:
            return 2 if packet.target == 1 and 2 in cands else (1 if 1 in cands else cands[0])
        return cands[0]


def test_lost_acks_create_duplicate_path_counted_once():
    """Node 1 receives the packet but all its ACKs to node 0 are lost; node 0
    re-decides to node 2. Two copies travel to the destination, which
    de-duplicates with its own cache: one delivery, no loss, conservation holds,
    and relay 1 behaves identically whether or not the packet was already delivered."""
    cfg = Settings(nodes=4, area=700, steps=30, ttl=40, shadow_sigma_db=0.0, load=0.0, location_service="oracle")
    tr = static_trace(cfg, [[50, 300], [300, 200], [300, 400], [550, 300]], packets=[(25, 0, 3)])
    pol = _SwitchPolicy(cfg); rng = np.random.default_rng(0); pol.begin_episode(False, 0, 1, rng)
    s = Simulator(cfg, pol, tr, 5, False, rng); s.frame_log = []
    orig = s.channel.resolve
    def resolve(frames, pos, alive, t, subslot):
        dec, dfr, tx, rx = orig(frames, pos, alive, t, subslot)
        for k, f in enumerate(frames):
            if f.kind == "ack" and f.sender == 1 and f.receiver == 0:
                dec[k, :] = False
        return dec, dfr, tx, rx
    s.channel.resolve = resolve
    r = s.run()
    assert r["generated"] == 1 and r["delivered"] == 1 and r["losses"] == {}, r
    data_to = [(snd, rcv) for _, k, snd, rcv, *_ in s.frame_log if k == "data"]
    assert (0, 1) in data_to and (0, 2) in data_to and (1, 3) in data_to and (2, 3) in data_to, data_to
    assert r["duplicate_receptions"] >= 1 and r["duplicate_deliveries"] == 0


def test_refused_packet_is_not_cached():
    cfg = Settings(nodes=3, buffer=1, location_service="oracle")
    tr = static_trace(cfg, [[0, 0], [100, 0], [200, 0]])
    s, _ = sim_for(cfg, "geo_greedy", tr)
    blocker = Packet(99, 1, 2, np.zeros(2), 0, 50); s.queues[1].append(blocker); s.obs.on_copy_added(blocker)
    p = Packet(1, 0, 2, np.zeros(2), 0, 50)
    delivered, refused, _ = s.receive(0, 1, p, 3)
    assert refused and not delivered and 1 not in s.seen[1]
    s.queues[1].clear()
    delivered, refused, _ = s.receive(0, 1, p, 4)
    assert not refused and 1 in s.seen[1] and len(s.queues[1]) == 1


def test_location_service_accounting():
    cfg = Settings(**{**SMALL.to_dict(), "location_period": 10})
    tr = make_trace(cfg, trace_key("train", "S1", 5))
    r_per = simulate(cfg, make_policy("geo_greedy", cfg), tr, 5, False, 1)
    epochs = len(range(0, r_per["steps_run"], cfg.location_period))
    assert r_per["location_bits"] == epochs * cfg.nodes * wire.location_frame_bits(cfg.nodes)
    cfg_o = Settings(**{**cfg.to_dict(), "location_service": "oracle"})
    r_or = simulate(cfg_o, make_policy("geo_greedy", cfg_o), tr, 5, False, 1)
    assert r_or["location_bits"] == 0 and r_per["energy_active_j"] > r_or["energy_active_j"]
    s, _ = sim_for(cfg, "geo_greedy", tr)
    for t in range(28):
        s.step(t)
    pos, step = s.location_of(3, 27)
    assert step == 20 and np.array_equal(pos, tr.pos[20][3])
    # a failed node stops refreshing its record; with pruning, packets to it are dropped as dest_unreachable
    cfg_f = Settings(**{**SMALL.to_dict(), "location_period": 10, "failure_ratio": 0.2, "failure_step": 15, "prune_stale_dest": True})
    tr_f = make_trace(cfg_f, trace_key("train", "S2", 3))
    s2, _ = sim_for(cfg_f, "geo_greedy", tr_f)
    r = s2.run()
    dead = int(tr_f.failed[0])
    assert int(s2.loc_seen[dead]) == 10 and s2.dest_stale(dead, 21) and not s2.dest_stale(dead, 20)
    assert r["losses"].get("dest_unreachable", 0) > 0 and r["generated"] == r["delivered"] + sum(r["losses"].values())


def test_heuristic_init_and_weight_sharing():
    cfg = Settings(nodes=5, init="heuristic", weight_sharing="shared")
    lq = LinearQ(cfg); ch = CostHeuristic(cfg)
    assert lq.theta.shape == (1, 10)
    rng = np.random.default_rng(0)
    for _ in range(200):
        k = int(rng.integers(2, 7))
        feats = rng.uniform(0, 1, (k, 10)); feats[:, 0] = 1.0; feats[:, 9] = 0.0
        pk = Packet(0, 0, 4, np.zeros(2), 0, 50)
        cands = list(range(10, 10 + k))
        lq.begin_episode(False, 0, 1, np.random.default_rng(1)); ch.begin_episode(False, 0, 1, np.random.default_rng(1))
        assert lq.choose(0, pk, cands, feats, 0, False) == ch.choose(0, pk, cands, feats, 0, False)
    lq.begin_episode(True, 0, 1, np.random.default_rng(0))
    f = np.zeros(10); f[0] = 1.0; before = lq.theta.copy()
    lq.on_forward(3, pk, 1, f, 0); lq.on_ack(3, pk, 1, False, 5.0, 5.0, 1.0)
    assert not np.array_equal(before, lq.theta) and lq.theta.shape == (1, 10)


def test_energy_scales_with_distance_only_through_loss_not_power():
    """Transmit power is fixed; energy per frame is independent of distance (declared model)."""
    cfg = Settings(nodes=2, area=600, steps=30, ttl=40, shadow_sigma_db=0.0, load=0.0)
    e = []
    for d in (100.0, 250.0):
        tr = static_trace(cfg, [[100, 100], [100 + d, 100]], packets=[(25, 0, 1)])
        s, _ = sim_for(cfg, "geo_greedy", tr); r = s.run(); e.append(r["energy_active_j"])
    assert abs(e[0] - e[1]) < 1e-9


def test_checkpoint_resume_and_interrupted_write():
    cfg = SMALL
    with tempfile.TemporaryDirectory() as tmp:
        full = Path(tmp) / "full"; part = Path(tmp) / "part"
        rec_full = R.run_job(cfg, "lfa_q", "S1", 0, 3, 2, "tune", full, quiet=True)
        # interrupted run: stop after two episodes by raising inside simulate
        calls = {"n": 0}
        orig = R.simulate
        def boom(*a, **k):
            calls["n"] += 1
            if calls["n"] == 3:
                raise KeyboardInterrupt
            return orig(*a, **k)
        R.simulate = boom
        try:
            R.run_job(cfg, "lfa_q", "S1", 0, 3, 2, "tune", part, quiet=True)
        except KeyboardInterrupt:
            pass
        finally:
            R.simulate = orig
        assert not (part / "DONE").exists() and len(json.loads((part / "records.json").read_text())) == 2
        # a stray partial checkpoint write must not be picked up
        (part / "checkpoint.pkl.tmp").write_bytes(b"garbage")
        rec_part = R.run_job(cfg, "lfa_q", "S1", 0, 3, 2, "tune", part, quiet=True)
        assert (part / "DONE").exists() and len(rec_part) == 5
        strip = lambda rows: [{k: v for k, v in r.items() if k not in ("elapsed_seconds", "decision_us_mean")} for r in rows]
        assert json.dumps(strip(rec_full), sort_keys=True, default=float) == json.dumps(strip(rec_part), sort_keys=True, default=float), "resume must reproduce the uninterrupted run"
        # completed job is not rerun
        again = R.run_job(cfg, "lfa_q", "S1", 0, 3, 2, "tune", part, quiet=True)
        assert json.dumps(again, default=float) == json.dumps(rec_part, default=float)
        # mixing runs is refused
        try:
            R.run_job(cfg, "geo_greedy", "S1", 0, 3, 2, "tune", part, quiet=True)
            raise AssertionError("manifest mismatch not detected")
        except RuntimeError:
            pass


def test_failure_scenario_applies_once():
    cfg = Settings(**{**SMALL.to_dict(), "failure_ratio": 0.2, "failure_step": 20})
    tr = make_trace(cfg, trace_key("train", "S2", 0))
    s, _ = sim_for(cfg, "geo_greedy", tr)
    r = s.run()
    assert s.failure_applied and (~s.alive).sum() == int(0.2 * cfg.nodes)
    assert r["losses"].get("failed_source", 0) > 0


def test_gpsr_perimeter_recovers_from_void():
    """S cannot make greedy progress (its only neighbour A is farther from D);
    geographic greedy drops the packet, GPSR recovers by the right-hand rule."""
    cfg = Settings(nodes=5, area=900, steps=30, ttl=40, shadow_sigma_db=0.0, load=0.0, location_service="oracle")
    pts = [[100, 300], [150, 550], [400, 600], [650, 550], [700, 300]]
    tr = static_trace(cfg, pts, packets=[(25, 0, 4)])
    r_g = sim_for(cfg, "geo_greedy", tr)[0].run()
    r_p = sim_for(cfg, "gpsr", tr)[0].run()
    assert r_g["delivered"] == 0 and r_g["losses"] == {"no_route": 1}, r_g["losses"]
    assert r_p["delivered"] == 1 and r_p["hops_mean"] == 4, (r_p["losses"], r_p["hops_mean"])
    assert r_p["data_header_bits"] > 0


def test_qmr_learns_and_frozen_flag():
    cfg = SMALL
    pol = make_policy("qmr", cfg)
    simulate(cfg, pol, make_trace(cfg, trace_key("train", "S1", 0)), 5, True, 1, 0, 2)
    assert pol.updates > 0 and len(pol.Q[0]) + sum(len(q) for q in pol.Q) > 0
    cfg_off = Settings(**{**SMALL.to_dict(), "qmr_online": False})
    pol2 = make_policy("qmr", cfg_off)
    simulate(cfg_off, pol2, make_trace(cfg_off, trace_key("train", "S1", 0)), 5, True, 1, 0, 2)
    n = pol2.updates
    simulate(cfg_off, pol2, make_trace(cfg_off, trace_key("tune", "S1", 0)), 5, False, 2)
    assert pol2.updates == n
    assert wire.ack_bits(cfg.nodes, True, pol.ack_extra_bits) == wire.ack_bits(cfg.nodes, True) + 24


def test_slack_queue_and_infeasible_drop_conserve():
    cfg = Settings(**{**SMALL.to_dict(), "queue_policy": "slack", "drop_infeasible": True, "load": 0.3, "ttl": 12, "buffer": 6})
    for m in ("lfa_q", "cost_heuristic"):
        r = simulate(cfg, make_policy(m, cfg), make_trace(cfg, trace_key("train", "S3", 1)), 5, True, 2)
        assert r["generated"] == r["delivered"] + sum(r["losses"].values())


TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_")]


def run_all() -> int:
    failed = 0
    started = time.time()
    for t in TESTS:
        t0 = time.time()
        try:
            t()
            print(f"PASS {t.__name__} ({time.time() - t0:.1f}s)", flush=True)
        except Exception:
            failed += 1
            print(f"FAIL {t.__name__}\n{traceback.format_exc()}", flush=True)
    print(f"{len(TESTS) - failed}/{len(TESTS)} passed in {time.time() - started:.1f}s")
    return 1 if failed else 0


def smoke(out: Path):
    """Tiny end-to-end run of every method with resume check; writes a JSON report."""
    cfg = Settings(nodes=12, area=500.0, steps=30, ttl=20, load=0.05)
    report = {"settings": cfg.to_dict(), "methods": {}}
    for m in sorted(METHODS):
        d = out / m
        if d.exists():
            shutil.rmtree(d)
        t0 = time.time()
        rec = R.run_job(cfg, m, "S1", 0, 2, 1, "tune", d, quiet=True)
        rec2 = R.run_job(cfg, m, "S1", 0, 2, 1, "tune", d, quiet=True)
        assert len(rec) == len(rec2)
        report["methods"][m] = dict(episodes=len(rec), pdr_eval=[r["pdr"] for r in rec if r["phase"] == "eval"], seconds=time.time() - t0,
                                    conservation_ok=all(r["generated"] == r["delivered"] + sum(r["losses"].values()) for r in rec))
    (out / "smoke_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report["methods"], indent=1))
