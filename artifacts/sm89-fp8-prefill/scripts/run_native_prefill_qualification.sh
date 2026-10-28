#!/usr/bin/env bash
set -euo pipefail
source /workspace/research/pr_engineering_20261026/engineering_env.sh
PR_ROOT=/workspace/research/value_reboot_20261027
PR_PY=/workspace/research/pr_engineering_20261026/.venv/bin/python
for ((attempt=0; attempt<360; attempt++)); do
  [[ -f "$PR_ROOT/prefill_pr_v1/ready.json" ]] && break
  sleep 5
done
test -f "$PR_ROOT/prefill_pr_v1/ready.json"
exec 9>"$PR_ROOT/gpu_experiments.lock"
flock -x 9
export CUDA_VISIBLE_DEVICES=0
PYTHONPATH="$PR_ROOT/native_baseline_runtime" "$PR_PY" "$PR_ROOT/qualify_native_prefill.py" --arm baseline
PYTHONPATH="$PR_ROOT/vllm_fp8" "$PR_PY" "$PR_ROOT/qualify_native_prefill.py" --arm candidate
"$PR_PY" - <<'PY'
from pathlib import Path
import json
r=Path('/workspace/research/value_reboot_20261027/prefill_pr_v1/qualification')
a=json.loads((r/'baseline/outputs.json').read_text());b=json.loads((r/'candidate/outputs.json').read_text())
assert a==b, 'Output hashes differ; inspect and do not proceed to serving claims'
(r/'COMPLETE.json').write_text(json.dumps({'complete':True,'exact_cases':len(a)}))
PY
