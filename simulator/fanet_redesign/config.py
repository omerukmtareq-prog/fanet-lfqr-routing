"""Frozen settings for the redesigned FANET routing simulator.

Every field is a declared modelling choice; see design/MODEL_AND_METHOD.md for
sources and rationale. Settings are immutable once a run starts and are
hashed into the run manifest.
"""
from __future__ import annotations
import dataclasses, hashlib, json
from typing import Any

STEP_SECONDS = 0.01  # 10 ms simulation step (historical TIME_UNIT)


@dataclasses.dataclass(frozen=True)
class Settings:
    # --- topology / mobility ---
    nodes: int = 100
    area: float = 1500.0
    steps: int = 300                # traffic-generation steps per episode
    ttl: int = 100                  # packet lifetime in steps
    load: float = 0.01              # Poisson packets per node per step (redesigned S1)
    speed_scale: float = 1.0
    margin: float = 20.0
    move_period: int = 10           # heading change every 10 steps
    failure_ratio: float = 0.0
    failure_step: int = 150
    failure_pattern: str = "random"  # random | spatial
    # --- queues ---
    buffer: int = 50
    max_visited: int = 16
    # --- frames (bits) ---
    payload_bits: int = 4000        # header/control sizes come from wire.py (pilot02)
    bit_rate: float = 11.0e6
    # --- radio ---
    tx_power_dbm: float = 10.0
    pl_d0_db: float = 40.0
    pl_exponent: float = 2.2
    shadow_sigma_db: float = 3.0
    shadow_epoch: int = 10
    noise_dbm: float = -95.0
    sinr_threshold_db: float = 10.0
    interference: bool = True
    cs_threshold_dbm: float = -92.0   # carrier-sense threshold (deferral)
    retry_pin: int = 3               # attempts to the same next hop before re-deciding
    data_subslot: float = 0.8       # fraction of the step for data/beacon/DN frames
    prune_distance: float = 800.0   # beyond this, received power is treated as zero
    # --- energy (W) ---
    p_tx_w: float = 1.65
    p_rx_w: float = 1.40
    p_idle_w: float = 1.15
    energy_budget_j: float = 500.0  # declared radio budget for lifetime projection
    # --- neighbour discovery ---
    beacon_period: int = 10
    neighbor_expiry: int = 20
    all_relay_ok: bool = True       # False = id-mod-4 test rule only; NOT a clustering protocol
    # --- location service (pilot02) ---
    location_service: str = "periodic"  # periodic | oracle (oracle = labelled control, true position free of cost)
    location_period: int = 50           # steps between network-wide location-table refreshes
    # --- queue / admission options (pilot03; any method may enable them; defaults = pilot02 behaviour) ---
    queue_policy: str = "fifo"          # fifo | slack (serve the queued packet with least deadline slack)
    prune_stale_dest: bool = False      # drop packets whose destination's location record missed a refresh
    drop_infeasible: bool = False       # drop packets whose remaining TTL is below the estimated hops to go
    # --- policy / learning ---
    candidate_mode: str = "all"     # all | quadrant (quadrant only for historical controls)
    feasibility: str = "pre"        # pre | post (post reproduces the historical ineligible-relay channel)
    alpha: float = 0.05
    gamma: float = 1.0
    eps_start: float = 0.5
    eps_end: float = 0.05
    lambda_energy: float = 5.0      # steps charged per data-frame transmit energy unit
    lambda_lifetime: float = 0.0    # optional reward term on (1 - residual energy of relay)
    c_drop: float = 100.0           # terminal cost of a lost packet (steps)
    use_drop_notification: bool = True
    use_bootstrap: bool = True
    feature_mask: str = "full"      # full | no_queue | no_energy | no_link | geometry
    online_adaptation: bool = False
    weight_sharing: str = "per_node"  # per_node | shared (shared = parameter sharing during training, frozen copy per node)
    init: str = "zero"                # zero | heuristic (theta initialised to reproduce cost_heuristic ordering)
    init_scale: float = 20.0          # steps per unit of heuristic score for init=heuristic
    feature_interactions: bool = False  # lfa_q: append progress*linkq, queue*linkq, progress*queue
    # QMR adaptation (Liu et al. 2020, parameters from the authors' qmr.c)
    qmr_w: float = 0.8                  # delay weight W in the reward
    qmr_window: int = 5                 # delay window N
    qmr_ewma: float = 0.5               # B
    qmr_eps: float = 0.1                # exploration during learning
    qmr_not_ack: int = 3                # failed attempts before the no-ACK penalty
    qmr_r_update: int = 10              # steps between neighbour-churn (discount) updates (100 ms)
    qmr_online: bool = True             # QMR keeps updating Q in its test phase (as in qmr.c)
    # historical tabular learner
    hist_alpha: float = 0.01
    hist_gamma: float = 0.9
    hist_reward_delivery: float = 100.0
    hist_reward_hop: float = -1.0
    hist_reward_drop: float = -50.0
    hist_eps_start: float = 0.99
    hist_eps_decay: float = 0.00009
    hist_eps_min: float = 0.12
    # cost heuristic weights
    w_progress: float = 1.0
    w_queue: float = 0.5
    w_energy: float = 0.2
    w_link: float = 0.5
    w_stability: float = 0.2

    # ------------------------------------------------------------------ derived
    def airtime_steps(self, bits: int) -> float:
        return bits / self.bit_rate / STEP_SECONDS

    @property
    def nominal_range(self) -> float:
        """Distance at which SNR equals the threshold without shadowing (m)."""
        margin = self.tx_power_dbm - self.pl_d0_db - self.noise_dbm - self.sinr_threshold_db
        return 10 ** (margin / (10 * self.pl_exponent))

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def sha256(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Settings":
        unknown = set(d) - {f.name for f in dataclasses.fields(Settings)}
        if unknown:
            raise ValueError(f"Unknown settings: {sorted(unknown)}")
        return Settings(**d)


# Proposed redesigned reference conditions (EXPERIMENT_PROPOSAL.md §2). The
# historical loads (0.10 / 0.30) exceed the physical channel capacity.
SCENARIOS: dict[str, dict[str, Any]] = {
    "S1": {"load": 0.01},
    "S1_L02": {"load": 0.02},          # prespecified calibration condition (Codex receipt 0003)
    "S2": {"load": 0.01, "failure_ratio": 0.2},
    "S3": {"load": 0.03},
    "S1_hist_load": {"load": 0.10},    # historical loads kept as stress tests
    "S3_hist_load": {"load": 0.30},
}
SCENARIO_CODE = {"S1": 0, "S2": 1, "S3": 2, "S1_hist_load": 3, "S3_hist_load": 4, "S1_L02": 5}
INDEX_LIMIT = 100_000
REPLICATE_STRIDE = 1_000   # trace index = replicate * REPLICATE_STRIDE + episode index

SPLIT_BASE = {"train": 1_000_000, "tune": 2_000_000, "final": 3_000_000}


def trace_key(split: str, scenario: str, index: int) -> int:
    """Deterministic key of an exogenous trace. Splits and scenarios never overlap.

    pilot02: every scenario has an explicit code (pilot01 mapped all unnamed
    scenarios to 9, so S1_hist_load and S3_hist_load shared keys) and the index
    range is enforced. Sensitivity variants run with a base scenario's keys on
    purpose (paired exogenous inputs, different settings)."""
    if scenario not in SCENARIO_CODE:
        raise ValueError(f"no trace code for scenario {scenario!r}")
    if not 0 <= index < INDEX_LIMIT:
        raise ValueError(f"trace index {index} outside [0, {INDEX_LIMIT})")
    return SPLIT_BASE[split] + SCENARIO_CODE[scenario] * INDEX_LIMIT + index


def replicate_index(replicate: int, k: int) -> int:
    if not 0 <= k < REPLICATE_STRIDE:
        raise ValueError(f"episode index {k} outside [0, {REPLICATE_STRIDE})")
    return replicate * REPLICATE_STRIDE + k
