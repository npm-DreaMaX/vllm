import argparse
import hashlib
import json
import math
import statistics
import threading
import time
from pathlib import Path

import pynvml
import torch
import triton.testing
import vllm
from vllm import _custom_ops

ROOT=Path('/workspace/research/value_reboot_20261027')
p=argparse.ArgumentParser();p.add_argument('--arm',choices=['baseline','candidate'],required=True)
args=p.parse_args()
outdir=ROOT/'prefill_pr_v1/qualification'/args.arm
outdir.mkdir(parents=True,exist_ok=False)
manifest=json.loads((ROOT/'prefill_pr_v1/ready.json').read_text())
core=next(Path(vllm.__file__).parent.glob('_C_stable_libtorch*.so'))
sha=hashlib.sha256(core.read_bytes()).hexdigest()
assert sha==manifest[f'{args.arm}_core_sha256']
pynvml.nvmlInit();handle=pynvml.nvmlDeviceGetHandleByIndex(0)
def pids():return {p.pid for p in pynvml.nvmlDeviceGetComputeRunningProcesses(handle)}
assert not pids(), 'GPU0 occupied'
anchor=torch.ones(1,device='cuda');torch.cuda.synchronize()
deadline=time.monotonic()+5
while not pids() and time.monotonic()<deadline:time.sleep(.05)
own=pids();assert len(own)==1
invalid=threading.Event();stop=threading.Event()
def monitor():
    while not stop.wait(.2):
        if pids()!=own:
            invalid.set()
            (outdir/'INVALID.json').write_text('{"reason":"External GPU0 process"}')
            return
threading.Thread(target=monitor,daemon=True).start()
torch.set_num_threads(1)
(outdir/'protocol.json').write_text(json.dumps({'core_sha256':sha,'seed':42,
    'exactness':'Full output SHA256 plus CUDA Graph versus direct torch.equal',
    'performance':'>=1GiB rotating FP8 weights, three CUDA Graph timing samples',
    'gpu':torch.cuda.get_device_name(0)},indent=2))
shapes=[(1,4096,4096),(512,4096,4096),(1023,4096,4096),
        (1024,4080,4096),(1024,4096,2032),(1024,4096,2048),
        (1025,4096,4096),(2048,8192,4096),(2048,8208,4096),
        (2048,12288,4096),(8192,24576,4096),(2048,4096,12288),
        (1024,4096,16384),(2048,4096,32768)]
rows=[]
for idx,(m,n,k) in enumerate(shapes):
    for dtype in [torch.float16,torch.bfloat16]:
        for channel in [False,True]:
            torch.manual_seed(42+idx)
            x=torch.randn(m,k,device='cuda',dtype=torch.bfloat16).to(torch.float8_e4m3fn)
            w=torch.randn(n,k,device='cuda',dtype=torch.bfloat16).to(torch.float8_e4m3fn).T
            sa=torch.rand((m,1) if channel else (1,),device='cuda')+.5
            sb=(torch.rand((1,n) if channel else (1,),device='cuda')+.5)/math.sqrt(k)
            bias=torch.randn(n,device='cuda',dtype=dtype) if channel else None
            y=torch.empty((m,n),device='cuda',dtype=dtype)
            def call():torch.ops._C.cutlass_scaled_mm(y,x,w,sa,sb,bias)
            call();ref=y.clone()
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):call()
            graph.replay();torch.cuda.synchronize()
            assert torch.equal(y,ref)
            assert torch.isfinite(y).all()
            digest=hashlib.sha256(y.view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
            rows.append(dict(M=m,N=n,K=k,dtype=str(dtype),channel_and_bias=channel,
                             sha256=digest,graph_equal=True))
            (outdir/'outputs.json').write_text(json.dumps(rows,indent=2))
            del graph,x,w,sa,sb,bias,y,ref
            if invalid.is_set():raise RuntimeError('External GPU interference')
    print('correctness shape',m,n,k,flush=True)
timings=[]
for m,n,k in [(512,24576,4096),(1024,24576,4096),(8192,24576,4096),
              (8192,12288,4096),(8192,4096,12288),(2048,8192,4096),
              (1024,4096,32768)]:
    count=math.ceil(1024**3/(n*k))
    weights=[torch.randn(n,k,device='cuda',dtype=torch.bfloat16).to(torch.float8_e4m3fn).T for _ in range(count)]
    x=torch.randn(m,k,device='cuda',dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    sa=torch.ones(m,1,device='cuda');sb=torch.ones(1,device='cuda')
    y=torch.empty((m,n),device='cuda',dtype=torch.bfloat16)
    def run():
        for w in weights:torch.ops._C.cutlass_scaled_mm(y,x,w,sa,sb,None)
    samples=[triton.testing.do_bench_cudagraph(run,rep=250)*1000/count for _ in range(3)]
    timings.append(dict(M=m,N=n,K=k,samples_us=samples,median_us=statistics.median(samples)))
    (outdir/'timings.json').write_text(json.dumps(timings,indent=2))
    print(timings[-1],flush=True)
    del weights,x,sa,sb,y
    if invalid.is_set():raise RuntimeError('External GPU interference')
stop.set()
(outdir/'COMPLETE.json').write_text(json.dumps({'complete':True,'exact_cases':len(rows),'timings':len(timings)}))
