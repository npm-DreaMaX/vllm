"""Redact identifiers and archive numerical evidence without changing measurements."""
import gzip,hashlib,json,re,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT/'原始证据/prefill_pr_v1';OUT=ROOT/'public_evidence/artifacts/sm89-fp8-prefill'
assert (RAW/'serve_9b_final_v1/COMPLETE.json').exists()
assert (RAW/'offline_gsm8k_9b/COMPLETE.json').exists()
OUT.mkdir(parents=True,exist_ok=True);manifest=[]
def clean(x):
 if isinstance(x,str):
  x=x.replace('/workspace','/workspace').replace('<research-server>','<research-server>')
  x=re.sub(r'GPU-[0-9a-fA-F-]{16,}', '<GPU-UUID>',x)
  return x
 if isinstance(x,list):return [clean(v) for v in x]
 if isinstance(x,dict):return {k:clean(v) for k,v in x.items() if k not in ['uuid','UUID','gpu_uuids','generated_texts']}
 return x
def copy(src,rel):
 original=src.read_bytes()
 if src.name.endswith('.json.gz'):data=gzip.compress(json.dumps(clean(json.loads(gzip.decompress(original))),separators=(',',':')).encode(),mtime=0)
 elif src.suffix=='.json':data=(json.dumps(clean(json.loads(original)),indent=2)+'\n').encode()
 elif src.suffix=='.jsonl':data=(''.join(json.dumps(clean(json.loads(l)),separators=(',',':'),ensure_ascii=False)+'\n' for l in original.decode().splitlines() if l)).encode()
 elif src.suffix in ['.log','.txt','.py','.patch','.diff','.sh','.md','.cuh','.prom']:data=clean(original.decode()).encode()
 else:data=original
 dst=OUT/rel;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(data)
 manifest.append(dict(path=str(rel),original_sha256=hashlib.sha256(original).hexdigest(),published_sha256=hashlib.sha256(data).hexdigest(),bytes=len(data)))
for src in sorted(RAW.rglob('*')):
 if not src.is_file() or src.suffix not in ['.json','.jsonl','.log','.txt','.patch','.diff']:continue
 if 'warmup' in src.name or src.name.endswith('_command.json') and 'warm' in src.name:continue
 copy(src,src.relative_to(RAW))
for src in sorted((ROOT/'复现脚本').glob('*')):
 if src.is_file():copy(src,Path('scripts')/src.name)
for src in sorted((ROOT/'图表').glob('*')):
 if src.is_file():copy(src,Path('figures')/src.name)
for src in sorted((ROOT/'原始证据/diagnosis').rglob('*')):
 if src.is_file():copy(src,Path('diagnosis')/src.relative_to(ROOT/'原始证据/diagnosis'))
for name in ['PR_BODY.md','REVIEW.md']:
 copy(ROOT/name,name)
(OUT/'redaction_manifest.json').write_text(json.dumps(dict(policy='Private path prefix, server IP, GPU UUIDs redacted; benchmark generated_texts omitted. All timing vectors, counts, metrics and model-quality token sequences retained. Original hashes refer to the complete local delivery.',files=manifest),indent=2)+'\n')
print(len(manifest),'files',round(sum(x['bytes'] for x in manifest)/1024**2,2),'MiB')
