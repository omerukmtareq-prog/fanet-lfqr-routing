#!/usr/bin/env bash
# Run a job list in parallel. Each line of JOBS is one complete, resumable
# `python -m fanet_redesign run ...` command. Re-running the same list resumes:
# finished jobs (DONE) return immediately, interrupted ones continue from their
# episode checkpoint. Usage: run_jobs.sh JOBS_FILE [PARALLEL]
set -u
JOBS=${1:?jobs file}
PAR=${2:-$(nproc)}
LOGDIR=$(dirname "$JOBS")/joblogs; mkdir -p "$LOGDIR"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
echo "start $(date -u +%FT%TZ) jobs=$(grep -c . "$JOBS") parallel=$PAR"
nl -ba "$JOBS" | xargs -P "$PAR" -L 1 bash -c 'n=$0; ( time "$@" ) > "'"$LOGDIR"'/job_$n.log" 2>&1 && echo "ok $n" || echo "FAIL $n (see job_$n.log)"'
echo "end $(date -u +%FT%TZ)"
