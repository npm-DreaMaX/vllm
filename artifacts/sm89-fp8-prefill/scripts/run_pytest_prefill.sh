#!/usr/bin/env bash
set -euo pipefail
source /workspace/research/pr_engineering_20261026/engineering_env.sh
PR_ROOT=/workspace/research/value_reboot_20261027
PR_PY=/workspace/research/pr_engineering_20261026/.venv/bin/python
while [[ ! -f "$PR_ROOT/prefill_pr_v1/qualification/COMPLETE.json" ]]; do sleep 5; done
exec 9>"$PR_ROOT/gpu_experiments.lock"
flock -x 9
cd "$PR_ROOT"
export CUDA_VISIBLE_DEVICES=0
for mode in baseline candidate; do
 if [[ "$mode" == baseline ]]; then
   export PYTHONPATH="$PR_ROOT/native_baseline_runtime:$PR_ROOT/vllm_fp8:$PR_ROOT"
 else
   export PYTHONPATH="$PR_ROOT/vllm_fp8:$PR_ROOT"
 fi
 export RESEARCH_CORE_SHA256=$("$PR_PY" -c "import json; print(json.load(open('$PR_ROOT/prefill_pr_v1/ready.json'))['${mode}_core_sha256'])")
 "$PR_PY" -m pytest -p pytest_runtime_audit_v3 --import-mode=importlib "$PR_ROOT/vllm_fp8/tests/kernels/quantization/test_cutlass_scaled_mm.py" -k 'fp8 or cutlass_subset' -v > "$PR_ROOT/prefill_pr_v1/pytest_${mode}.log" 2>&1
 done
"$PR_PY" -c "import json;json.dump({'complete':True},open('$PR_ROOT/prefill_pr_v1/pytest_complete.json','w'))"
