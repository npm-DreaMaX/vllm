"""Quality only: fixed queue, baseline GPU0 and candidate GPU1."""
import fcntl
import json
import os
import subprocess
import time
from pathlib import Path
import pynvml

ROOT=Path('/workspace/research/value_reboot_20261027')
PY=ROOT.parent/'pr_engineering_20261026/.venv/bin/python'
OUT=ROOT/'prefill_pr_v1/offline_gsm8k_9b'
deadline=time.monotonic()+7200
while not (ROOT/'prefill_pr_v1/serve_9b_final_v1/COMPLETE.json').exists():
    if time.monotonic()>deadline:raise TimeoutError('Final serving run')
    time.sleep(5)
with (ROOT/'gpu_experiments.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX)
    pynvml.nvmlInit()
    for gpu in [0,1]:
        assert not pynvml.nvmlDeviceGetComputeRunningProcesses(pynvml.nvmlDeviceGetHandleByIndex(gpu)), 'GPU occupied'
    OUT.mkdir(exist_ok=False)
    processes=[]
    try:
        for gpu,mode in enumerate(['baseline','candidate']):
            env=os.environ.copy()
            runtime=ROOT/('native_baseline_runtime' if mode=='baseline' else 'vllm_fp8')
            env.update(CUDA_VISIBLE_DEVICES=str(gpu),PYTHONPATH=str(runtime),
                       HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',VLLM_GPU_SYNC_CHECK='error',
                       VLLM_USE_V2_MODEL_RUNNER='1',
                       TMPDIR='/workspace/tmp/fp8_eval_v3',
                       TMP='/workspace/tmp/fp8_eval_v3',
                       TEMP='/workspace/tmp/fp8_eval_v3',
                       VLLM_CACHE_ROOT=str(ROOT/'prefill_pr_v1/offline_cache'/mode/'vllm'),
                       TORCHINDUCTOR_CACHE_DIR=str(ROOT/'prefill_pr_v1/offline_cache'/mode/'inductor'))
            command=['taskset','-c','8-15' if gpu==0 else '16-23',str(PY),
                     str(ROOT/'offline_gsm8k_9b.py'),'--mode',mode,'--out',str(OUT/mode)]
            (OUT/f'{mode}_command.json').write_text(json.dumps({'argv':command,'gpu':gpu,'runtime':str(runtime)},indent=2))
            log=(OUT/f'{mode}.log').open('w')
            process=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT)
            processes.append((process,log))
        for process,log in processes:
            result=process.wait(timeout=3600)
            log.close()
            if result:raise RuntimeError(f'Quality worker exited {result}')
        a=[json.loads(x) for x in (OUT/'baseline/responses.jsonl').read_text().splitlines()]
        b=[json.loads(x) for x in (OUT/'candidate/responses.jsonl').read_text().splitlines()]
        assert len(a)==len(b)==1319
        rows=[dict(index=x['index'],text_equal=x['text']==y['text'],tokens_equal=x['token_ids']==y['token_ids'],
                   baseline_correct=x['answer']==x['label'],candidate_correct=y['answer']==y['label']) for x,y in zip(a,b)]
        (OUT/'comparison.json').write_text(json.dumps(rows,indent=2))
        result=dict(complete=True,questions=len(rows),exact_text=sum(x['text_equal'] for x in rows),
                    exact_tokens=sum(x['tokens_equal'] for x in rows),baseline_correct=sum(x['baseline_correct'] for x in rows),
                    candidate_correct=sum(x['candidate_correct'] for x in rows),
                    baseline_truncated=sum(x['finish_reason']=='length' for x in a),
                    candidate_truncated=sum(x['finish_reason']=='length' for x in b),
                    scope='Official full GSM8K, fixed offline queue, same architecture cards; no quality throughput claim')
        (OUT/'COMPLETE.json').write_text(json.dumps(result,indent=2))
        print(json.dumps(result),flush=True)
    except BaseException as error:
        (OUT/'FAILED.json').write_text(json.dumps({'error':repr(error)}))
        raise
    finally:
        for process,log in processes:
            if process.poll() is None:process.terminate()
            if not log.closed:log.close()
