"""Native before/after launch resource trace, not a serving latency benchmark."""
import argparse,hashlib,json
from pathlib import Path
import torch,vllm
from vllm import _custom_ops
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args();args.out.mkdir(parents=True,exist_ok=False)
torch.manual_seed(42)
a=torch.randn(8192,4096,device='cuda',dtype=torch.bfloat16).to(torch.float8_e4m3fn)
b=torch.randn(24576,4096,device='cuda',dtype=torch.bfloat16).to(torch.float8_e4m3fn).T
y=torch.empty(8192,24576,device='cuda',dtype=torch.bfloat16);sa=torch.ones(8192,1,device='cuda');sb=torch.ones(1,device='cuda')
def run(): torch.ops._C.cutlass_scaled_mm(y,a,b,sa,sb,None)
for _ in range(10):run()
torch.cuda.synchronize()
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=True) as prof:
 for _ in range(3):
  with torch.profiler.record_function('native_prefill_M8192_N24576_K4096'):run()
 torch.cuda.synchronize()
prof.export_chrome_trace(str(args.out/'launch.trace.json'))
t=json.loads((args.out/'launch.trace.json').read_text());events=[e for e in t['traceEvents'] if e.get('cat')=='kernel']
core=next(Path(vllm.__file__).parent.glob('_C_stable_libtorch*.so'))
(args.out/'summary.json').write_text(json.dumps(dict(core_sha256=hashlib.sha256(core.read_bytes()).hexdigest(),scope='Instrumented native launches; use qualification timings for performance.',kernels=[{k:e[k] for k in ['name','dur','args']} for e in events]),indent=2))
