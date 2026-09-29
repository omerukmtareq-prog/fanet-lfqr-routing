# Metrics schema (METRICS_SCHEMA = 2, pilot02)

`records.json` in a run directory is a list of per-episode objects. Training
episodes (`phase = "train"`) are diagnostics only; evaluation episodes
(`phase = "eval"`) are the observations. Fields:

| Field | Meaning |
|---|---|
| method, scenario, replicate, episode, phase, split, index | job identity; `index` is the episode index inside the job; the trace index is replicate·1000 + index |
| trace_key, trace_sha256, channel_seed, policy_seed, settings_sha256 | provenance; identical `trace_sha256` across methods proves paired inputs |
| steps_run, seconds_run | steps simulated (early stop when all queues drain after the traffic horizon) and simulated seconds |
| generated, delivered, pdr | offered packets, delivered packets (unique, observer-counted), delivered/offered |
| delay_mean, delay_p95 | delivered-only delay in steps (× 10 ms) |
| delay_censored_mean | penalised delay: undelivered packets counted at TTL (not a survival estimator) |
| hops_mean | hops of delivered packets |
| attempts_total, attempts_per_delivered | data transmission attempts (deferred frames excluded) |
| losses | observer terminal loss per reason: expired, no_route, buffer_overflow (relay refused the last copy), source_overflow, failed_source, failed_queue, ineligible_relay, dedup_loss, horizon, dest_unreachable (destination's location record missed a refresh; only with prune_stale_dest), infeasible (TTL below estimated hops; only with drop_infeasible) |
| stale_discards | drops of a copy while another copy lived or after the packet was terminal (no terminal event) |
| duplicate_receptions, duplicate_deliveries, refusals | node-local duplicate-cache hits; observer duplicate deliveries (expected 0); relay refusals (full buffer) |
| deferrals | frames deferred by carrier sensing, per kind |
| energy_tx_j, energy_rx_j, energy_idle_j | radio-state energy (receiver-state model: RX while any frame is sensed ≥ CS threshold, decoded or not, overlaps counted once; TX precedence; IDLE otherwise; location-service share included) |
| energy_active_j, energy_total_j | tx+rx; tx+rx+idle |
| energy_total_per_offered_j, energy_total_per_delivered_j | total radio energy normalised (constraint metric) |
| energy_active_per_offered_j, energy_active_per_delivered_j | active-only (supplementary) |
| lifetime_projection_s | energy_budget_j / max over nodes of total radio power over the node's alive time: first-node-depletion EXTRAPOLATION from a short episode, not a measured lifetime (constraint metric) |
| lifetime_active_projection_s | the pilot01 active-only projection (supplementary) |
| load_balance_max_over_mean | max / mean active-energy rate over nodes |
| control_bits | beacon + ACK + DN + location-service bits (serialized sizes, wire.py) |
| location_bits, data_header_bits | location-service bits; network-layer data header bits (variable with visited list and N) |
| payload_bits_delivered, control_overhead_ratio, information_overhead_ratio | delivered payload; control/payload; (control + data headers + data MAC)/payload |
| frames, frames_decoded, frame_bits | frames transmitted, decoded and bits per kind |
| decisions, decision_us_mean | policy decisions and mean decision time (host measurement) |
| max_queue | maximum queue occupancy observed |
| epsilon, policy_updates, policy_memory_bytes, learner_mean_cost | learner diagnostics |
| location_service | periodic (default) or oracle (labelled control) |
| elapsed_seconds | wall time of the episode (host measurement) |
