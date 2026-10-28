"""Swapped-card native A/B, repeated unprofiled 9B serving measurements."""
import fcntl
import json
import sys
import threading
import time
from pathlib import Path
import pynvml

ROOT=Path('/workspace/research/value_reboot_20261027')
sys.path.insert(0,str(ROOT))
import prefill_9b_server as base
from warm_serving import warm
h,pilot=base.h,base.pilot
h.OUT=ROOT/'prefill_pr_v1/serve_9b_final_v1'


def main():
    deadline=time.monotonic()+3600
    for marker in ['qualification/COMPLETE.json','pytest_complete.json']:
        while not (ROOT/'prefill_pr_v1'/marker).exists():
            if time.monotonic()>deadline:raise TimeoutError(marker)
            time.sleep(5)
    with (ROOT/'gpu_experiments.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        h.OUT.mkdir(exist_ok=False)
        workloads=[('prefill8k_o16_c1',16,8192,16,1),
                   ('long8k_c1',8,8192,128,1),
                   ('long8k_c4',16,8192,128,4),
                   ('long8k_c17',34,8192,128,17),
                   ('medium2k_c4',16,2048,128,4),
                   ('decode_control_c17',34,256,512,17)]
        (h.OUT/'protocol.json').write_text(json.dumps({
            'model':'Qwen3.5-9B','same_precision':'fp8_per_tensor/BF16 output',
            'rounds':[['native_gpu0','candidate_gpu1'],['candidate_gpu0','native_gpu1']],
            'concurrent_cards':True,'workloads':workloads,'repeat_per_start':3,
            'long_output_control':['long8k_o1024_c4',8,8192,1024,4],
            'warmup_seconds':45,'profiler':False,
            'sampling':'random exact lengths,seed42,temperature0,ignore_eos',
            'interpretation':'Six measurements per arm/workload from two starts/two cards; not six independent starts',
        },indent=2))
        pynvml.nvmlInit()
        handles=[pynvml.nvmlDeviceGetHandleByIndex(i) for i in [0,1]]
        def pids(g):return {p.pid for p in pynvml.nvmlDeviceGetComputeRunningProcesses(handles[g])}
        telemetry=threading.Thread(target=h.monitor,daemon=True);telemetry.start()
        servers=[]
        try:
            for rd,modes in enumerate([['native','candidate'],['candidate','native']]):
                assert not pids(0) and not pids(1),'GPUs already occupied'
                servers=[base.start(mode,rd*2+g,gpu=g) for g,mode in enumerate(modes)]
                h.parallel(servers,h.ready)
                own=[pids(g) for g in [0,1]]
                assert all(len(x)==1 for x in own),own
                invalid=threading.Event();stop=threading.Event()
                def watch():
                    while not stop.wait(.2):
                        if any(pids(g)!=own[g] for g in [0,1]):
                            invalid.set()
                            (h.OUT/f'INVALID_round{rd}.json').write_text('{"reason":"External GPU process"}')
                            return
                watcher=threading.Thread(target=watch,daemon=True);watcher.start()
                try:
                    h.parallel(servers,pilot.quality_preflight)
                    h.parallel(servers,pilot.bench,'warmup_8k',4,8192,16,4)
                    h.parallel(servers,lambda s:warm(s,h,17,seconds=45))
                    for repeat in [1,2,3]:
                        points=workloads if (rd+repeat)%2 else list(reversed(workloads))
                        for name,count,inp,out,c in points:
                            h.PHASE=f'round{rd}_{name}_r{repeat}'
                            if invalid.is_set():raise RuntimeError('External GPU interference')
                            h.parallel(servers,pilot.bench,f'{name}_r{repeat}',count,inp,out,c)
                            if invalid.is_set():raise RuntimeError('External GPU interference')
                    h.PHASE=f'round{rd}_long_output_control'
                    h.parallel(servers,pilot.bench,'long8k_o1024_c4',8,8192,1024,4)
                    if invalid.is_set():raise RuntimeError('External GPU interference')
                finally:
                    stop.set();watcher.join(timeout=5)
                for server in servers:h.stop(server)
                servers=[]
            (h.OUT/'COMPLETE.json').write_text('{"complete":true}')
        except BaseException as error:
            (h.OUT/'FAILED.json').write_text(json.dumps({'error':repr(error),'phase':h.PHASE}))
            raise
        finally:
            for server in servers:h.stop(server)
            h.STOP.set();telemetry.join(timeout=10)


if __name__=='__main__':main()
