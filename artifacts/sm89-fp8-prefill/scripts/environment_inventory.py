import importlib.metadata,json,platform,subprocess
from pathlib import Path
r=Path('/workspace/research/value_reboot_20261027')
data=dict(python=platform.python_version(),platform=platform.platform(),packages=sorted([dict(name=d.metadata['Name'],version=d.version) for d in importlib.metadata.distributions()],key=lambda x:x['name'].lower()),nvidia_smi=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,driver_version,memory.total,power.limit','--format=csv,noheader'],text=True),cuda_compiler=subprocess.check_output(['/workspace/tools/cuda-12.9/bin/nvcc','--version'],text=True))
(r/'prefill_pr_v1/environment.json').write_text(json.dumps(data,indent=2))
