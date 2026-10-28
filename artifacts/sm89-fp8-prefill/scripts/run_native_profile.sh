#!/usr/bin/env bash
set -euo pipefail
R=/workspace/research/value_reboot_20261027
source /workspace/research/pr_engineering_20261026/engineering_env.sh
PY=/workspace/research/pr_engineering_20261026/.venv/bin/python
until test -f "$R/prefill_pr_v1/offline_gsm8k_9b/COMPLETE.json"; do sleep 5; done
exec 9>"$R/gpu_experiments.lock"
flock -x 9
if test -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"; then echo 'GPU occupied; profile not started'; exit 1; fi
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$R/native_baseline_runtime" "$PY" "$R/profile_native_prefill.py" --out "$R/prefill_pr_v1/native_profile/baseline"
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$R/vllm_fp8" "$PY" "$R/profile_native_prefill.py" --out "$R/prefill_pr_v1/native_profile/candidate"
