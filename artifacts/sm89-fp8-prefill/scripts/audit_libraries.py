import hashlib,json
from pathlib import Path
r=Path('/workspace/research/value_reboot_20261027')
a=r/'native_baseline_runtime/vllm';b=r/'vllm_fp8/vllm'
def sha(p):return hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
rows=[]
for p in sorted(a.rglob('*.so')):
 q=b/p.relative_to(a)
 if q.exists():rows.append(dict(path=str(p.relative_to(a)),baseline_sha256=sha(p),candidate_sha256=sha(q),baseline_bytes=p.stat().st_size,candidate_bytes=q.stat().st_size))
assert rows and all(x['baseline_sha256']==x['candidate_sha256'] for x in rows if x['path']!='_C_stable_libtorch.abi3.so')
(r/'prefill_pr_v1/dependency_audit.json').write_text(json.dumps({'only_changed_library':'_C_stable_libtorch.abi3.so','libraries':rows},indent=2))
print(len(rows),'libraries audited')
