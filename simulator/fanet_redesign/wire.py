"""Explicit wire format: every frame is charged its serialized size.

pilot02 replaces the fixed header/control sizes of pilot01 with field budgets
that scale with the node count (node identifiers take ceil(log2 N) bits) and
with the per-packet visited list. All sizes are in bits and include a fixed
MAC header + FCS (MAC_BITS, 802.11-style 24-byte header + 4-byte FCS). PHY
preamble time is not modelled (declared).

The field tables below are the published encoding. `tests.py` checks that the
simulator charges exactly these sizes.
"""
from __future__ import annotations
import math

MAC_BITS = 224          # 24-byte MAC header + 4-byte FCS
COORD_BITS = 16         # one coordinate at 0.1 m resolution (area <= 6553.5 m)
VEL_BITS = 12           # one velocity component, 0.1 m/s resolution, signed (|v| <= 204.7 m/s)
PID_BITS = 32
STEP_BITS = 16          # absolute step stamp (wraps after 655 s; episodes are < 5 s)
TYPE_BITS = 4
VALUE_BITS = 32         # float32 cost-to-go carried on learner ACKs


def id_bits(n_nodes: int) -> int:
    return max(1, math.ceil(math.log2(max(2, n_nodes))))


def visited_count_bits(max_visited: int) -> int:
    return max(1, math.ceil(math.log2(max_visited + 1)))


def data_header_fields(n_nodes: int, max_visited: int, n_visited: int) -> dict:
    ib = id_bits(n_nodes)
    return {
        "type": TYPE_BITS,
        "pid": PID_BITS,
        "src": ib, "dst": ib, "prev_hop": ib,
        "generated_step": STEP_BITS,
        "expiry_step": STEP_BITS,
        "dst_pos": 2 * COORD_BITS,
        "dst_pos_step": STEP_BITS,           # age of the location-service entry
        "hops": 8,
        "visited_count": visited_count_bits(max_visited),
        "visited_ids": n_visited * ib,
    }


def data_frame_bits(payload_bits: int, n_nodes: int, max_visited: int, n_visited: int) -> int:
    return MAC_BITS + payload_bits + sum(data_header_fields(n_nodes, max_visited, n_visited).values())


def beacon_fields(n_nodes: int) -> dict:
    return {
        "type": TYPE_BITS, "sender": id_bits(n_nodes), "seq": STEP_BITS,
        "pos": 2 * COORD_BITS, "vel": 2 * VEL_BITS,
        "queue": 8, "energy": 8, "relay_ok": 1,
    }


def ack_fields(n_nodes: int, with_value: bool, extra_bits: int = 0) -> dict:
    ib = id_bits(n_nodes)
    f = {"type": TYPE_BITS, "pid": PID_BITS, "sender": ib, "receiver": ib, "flags": 2}  # delivered / refused
    if with_value:
        f["value"] = VALUE_BITS
    if extra_bits:
        f["policy_extra"] = extra_bits      # e.g. QMR: discount r (8) + queue wait (16)
    return f


def dn_fields(n_nodes: int) -> dict:
    ib = id_bits(n_nodes)
    return {"type": TYPE_BITS, "pid": PID_BITS, "sender": ib, "receiver": ib, "reason": 4}


def location_record_bits(n_nodes: int) -> int:
    return id_bits(n_nodes) + 2 * COORD_BITS + STEP_BITS


def beacon_bits(n_nodes: int) -> int:
    return MAC_BITS + sum(beacon_fields(n_nodes).values())


def ack_bits(n_nodes: int, with_value: bool, extra_bits: int = 0) -> int:
    return MAC_BITS + sum(ack_fields(n_nodes, with_value, extra_bits).values())


def gpsr_header_bits(n_nodes: int) -> int:
    """GPSR perimeter state carried in the data header (Karp & Kung 2000):
    mode flag, Lp and Lf positions, first perimeter edge e0 (two ids)."""
    return 1 + 4 * COORD_BITS + 2 * id_bits(n_nodes)


def dn_bits(n_nodes: int) -> int:
    return MAC_BITS + sum(dn_fields(n_nodes).values())


def location_frame_bits(n_nodes: int) -> int:
    """One aggregated location-table broadcast carrying one record per node."""
    return MAC_BITS + TYPE_BITS + n_nodes * location_record_bits(n_nodes)


def budget_table(n_nodes: int, max_visited: int, payload_bits: int) -> dict:
    """Human-readable budget used in RELEASE.md / MODEL_AND_METHOD.md."""
    return {
        "mac_bits": MAC_BITS,
        "id_bits": id_bits(n_nodes),
        "data_header_fields_empty_visited": data_header_fields(n_nodes, max_visited, 0),
        "data_frame_bits_min": data_frame_bits(payload_bits, n_nodes, max_visited, 0),
        "data_frame_bits_max": data_frame_bits(payload_bits, n_nodes, max_visited, max_visited),
        "beacon_fields": beacon_fields(n_nodes), "beacon_bits": beacon_bits(n_nodes),
        "ack_bits_plain": ack_bits(n_nodes, False), "ack_bits_with_value": ack_bits(n_nodes, True),
        "dn_fields": dn_fields(n_nodes), "dn_bits": dn_bits(n_nodes),
        "location_record_bits": location_record_bits(n_nodes),
        "location_frame_bits": location_frame_bits(n_nodes),
    }
