# Running a release on a RunPod CPU pod

The simulator is CPU-only (numpy). Use a RunPod **CPU pod**, not a GPU pod;
speed comes from running one job per vCPU in parallel. Every job is resumable
at episode granularity, so a pod restart only costs the episodes in progress.

## 1. Create the pod

REST API `POST https://rest.runpod.io/v1/pods` (header `Authorization: Bearer <key>`),
body used for phase 0:

```json
{"name": "omar-fanet-<release>-<phase>", "computeType": "CPU",
 "cpuFlavorIds": ["cpu3c"], "vcpuCount": 32, "containerDiskInGb": 20,
 "imageName": "python:3.11-slim", "ports": ["8080/http"],
 "env": {"AGENT_TOKEN": "<random 32-hex>", "AGENT_B64": "<base64 of runpod/pod_agent.py>"},
 "dockerStartCmd": ["bash", "-c", "pip install -q numpy==2.2.6 && echo $AGENT_B64 | base64 -d > /agent.py && python /agent.py"]}
```

The agent (`pod_agent.py`) listens on 8080 behind the RunPod HTTPS proxy
(`https://<pod_id>-8080.proxy.runpod.net`) and requires `X-Token` on every call
except `/health`. It only moves files and starts the declared runner.

## 2. Upload the release and start a phase

```
POST /upload?dest=rel          body: tar.gz of the release package/ directory
POST /exec {"id": "phase0", "cmd": "cd /workspace/rel && python -m fanet_redesign test > /workspace/logs/tests.txt 2>&1 && python -m fanet_redesign hash > /workspace/logs/hash.txt && python runpod/make_jobs.py phase0 /workspace/runs/phase0 > /workspace/jobs_phase0.txt && runpod/run_jobs.sh /workspace/jobs_phase0.txt 30"}
GET  /status?id=phase0         progress: log tail, number of DONE markers, load average
GET  /download?path=runs       tar.gz of all run directories (records, manifests, checkpoints)
```

The runner stops nothing on failure: each failed job is reported as `FAIL n`
with its log in `joblogs/`. Re-running the same `/exec` command resumes.

## 3. Collect, verify, stop

Download `runs`, `runs_conv` and `logs`, verify the package hash in
`logs/hash.txt` equals the release manifest, import into the project folder
and run `python -m fanet_redesign summarize --root <runs>`. Then **stop and
delete the pod** (`DELETE /v1/pods/<id>`); a stopped pod still bills its disk.

Container disk is ephemeral. For phases longer than about an hour attach a
network volume at /workspace, or download intermediate results periodically.
