# fanet-lfqr-routing

Code, trained models and results for the study *Learned Feature-based Q-Routing (LFQR) for flying ad hoc networks*.

Every result in the paper is produced by the simulator in `simulator/` at release `r2026.09.21-pilot03`; the campaign of 21 September 2026 and the trained models are included so the reported numbers can be reproduced.

* `simulator/` — release r2026.09.21-pilot03 of the `fanet_redesign` package (numpy only), including `runpod/` job generators and the pod agent.
* `campaign/r2026.09.21-pilot03/` — aggregated replicate-level results from the RunPod campaign of 2026-09-21 (pods A and B), tuning records and selections.

Code sha256 of the package used for every run: `4ab1ce51bc95f0a46be0cf1e4ba63afe6aeb1e266326af063e5f16292bef88e2`.
* `models/` — the 46 trained LFQR models (`lfqr_models.json`: 13 shared weights per model, full training and evaluation settings with hashes, stored evaluation PDR; `models_compact.tsv`: the same weights as a flat table; `lfqr_selected_config.json`: the locked configuration).
* `examples/load_and_evaluate.py` — loads a model, replays its frozen evaluation episodes and checks the PDR: `python examples/load_and_evaluate.py models/lfqr_models.json --all-fast`. All 46 models reproduce the campaign PDRs exactly (checked 2026-09-22).
