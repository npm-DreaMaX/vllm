#!/usr/bin/env bash
set -euo pipefail
source /workspace/research/pr_engineering_20261026/engineering_env.sh
PR_ROOT=/workspace/research/value_reboot_20261027
PR_PY=/workspace/research/pr_engineering_20261026/.venv/bin/python
PR_INCLUDES="$CUDA_HOME/targets/x86_64-linux/include"
for directory in /workspace/research/pr_engineering_20261026/.venv/lib/python3.12/site-packages/nvidia/*/include; do
    [[ "$directory" == */cu13/* ]] && continue
    PR_INCLUDES="$PR_INCLUDES:$directory"
done
export CPATH="$PR_INCLUDES" CPLUS_INCLUDE_PATH="$PR_INCLUDES"
mkdir -p "$PR_ROOT/prefill_pr_v1"
cmake --build "$PR_ROOT/main_cmake_build" --target _C_stable_libtorch -j 4
exec 9>"$PR_ROOT/gpu_experiments.lock"
flock -x 9
cmake --install "$PR_ROOT/main_cmake_build" --component _C_stable_libtorch
"$PR_PY" - <<'PY'
from pathlib import Path
import hashlib,json,subprocess
r=Path('/workspace/research/value_reboot_20261027')
d=subprocess.check_output(['git','-C',str(r/'vllm_fp8'),'diff'])
(r/'prefill_pr_v1/source.diff').write_bytes(d)
m=json.loads((r/'native_baseline_snapshot.json').read_text())
p=r/'vllm_fp8/vllm/_C_stable_libtorch.abi3.so'
m['candidate_core_sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
m['diff_sha256']=hashlib.sha256(d).hexdigest()
m['branch']='codex/sm89-fp8-prefill-20261028'
(r/'prefill_pr_v1/ready.json').write_text(json.dumps(m,indent=2))
PY
