"""Current-main serving A/B; all artifacts remain under the private workspace."""
from pathlib import Path
import concurrent.futures as cf
import datetime
import gzip
import hashlib
import json
import os
import re
import signal
import subprocess
import threading
import time
import urllib.request

ROOT = Path('/workspace')
ENG = ROOT / 'research/pr_engineering_20261026'
PY = str(ENG / '.venv/bin/python')
MODEL = ROOT / 'models/Qwen3-4B-Instruct-2507'
MODEL_ID = 'Qwen/Qwen3-4B-Instruct-2507'
STOP = threading.Event()
LOCK = threading.Lock()
NXT = ROOT / 'research/mlp_redesign_20261027'
OUT = NXT / 'serving_context_final_v1'
MODEL_KEY = '4b'
PHASE = 'init'


def event(kind, **fields):
    record = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'event': kind, **fields}
    with LOCK:
        with (OUT / 'events.jsonl').open('a') as file:
            file.write(json.dumps(record) + '\n')
    print(json.dumps(record), flush=True)


def request(port, path, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(f'http://127.0.0.1:{port}{path}', data=data,
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=600) as response:
        return response.read().decode()


def monitor():
    fields = 'index,memory.used,utilization.gpu,clocks.sm,clocks.mem,temperature.gpu,power.draw'
    with (OUT / 'gpu_telemetry.jsonl').open('w') as file:
        while not STOP.is_set():
            try:
                value = subprocess.check_output(['nvidia-smi', '--query-gpu=' + fields,
                    '--format=csv,noheader,nounits'], text=True, timeout=5)
                file.write(json.dumps({'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    'phase': PHASE, 'columns': fields, 'rows': value.splitlines()}) + '\n')
                file.flush()
            except Exception as error:
                event('telemetry_error', error=repr(error))
            STOP.wait(2)


def start(gpu, mode, round_number):
    folder = OUT / f'{MODEL_KEY}_round{round_number}_gpu{gpu}_{mode}'
    folder.mkdir()
    source = NXT / 'context_dispatch' if mode == 'packed' else ENG / 'vllm'
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(gpu), PYTHONPATH=str(source),
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONUNBUFFERED='1',
               VLLM_USE_RUST_FRONTEND='0', VLLM_USE_RUST_BENCH='0',
               VLLM_USE_V2_MODEL_RUNNER='1')
    env['VLLM_CACHE_ROOT'] = str(OUT.parent / 'final_cache' / MODEL_KEY / mode / 'vllm')
    env['TORCHINDUCTOR_CACHE_DIR'] = str(OUT.parent / 'final_cache' / MODEL_KEY / mode / 'inductor')
    trace = folder / 'trace'
    trace.mkdir()
    profiler = {'profiler': 'torch', 'torch_profiler_dir': str(trace),
                'torch_profiler_with_stack': False, 'torch_profiler_record_shapes': True,
                'ignore_frontend': True, 'detailed_trace_annotation': True,
                'delay_iterations': 16, 'max_iterations': 8}
    port = 18420 + gpu
    cores = '0-15' if gpu == 0 else '16-31'
    args = ['taskset', '-c', cores, PY, str(ENG / 'run_cli.py'), 'serve', str(MODEL),
            '--served-model-name', MODEL_ID, '--host', '127.0.0.1', '--port', str(port),
            '--tensor-parallel-size', '1', '--dtype', 'bfloat16', '--max-model-len', '10240',
            '--max-num-seqs', '32', '--max-num-batched-tokens', '8192',
            '--gpu-memory-utilization', '0.9', '--kv-cache-memory-bytes', str(KV_BYTES),
            '--seed', '42', '--no-enable-prefix-caching', '--enable-chunked-prefill',
            '--generation-config', 'vllm', '--attention-backend', 'FLASH_ATTN',
            '--profiler-config', json.dumps(profiler),
            '--compilation-config', json.dumps({'cudagraph_capture_sizes':[1,2,4,8,16,24,32]})]
    if mode == 'packed':
        args += ['--kernel-config', json.dumps({'mlp_backend': 'lossless_packed'})]
    metadata = {'model_revision': MODEL_REVISION, 'source_head': subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip(), 'source_patch_sha256':hashlib.sha256(subprocess.check_output(['git','diff','--binary'],cwd=source)).hexdigest(), 'argv': args, 'gpu': gpu, 'mode': mode, 'source': str(source),
                'environment': {key: env[key] for key in (
                    'PYTHONPATH', 'VLLM_CACHE_ROOT', 'TORCHINDUCTOR_CACHE_DIR',
                    'VLLM_USE_V2_MODEL_RUNNER', 'CUDA_VISIBLE_DEVICES')}}
    (folder / 'server_command.json').write_text(json.dumps(metadata, indent=2) + '\n')
    log = (folder / 'server.log').open('w')
    process = subprocess.Popen(args, cwd=source, env=env, stdout=log,
                               stderr=subprocess.STDOUT, start_new_session=True)
    return {'gpu': gpu, 'mode': mode, 'folder': folder, 'port': port,
            'process': process, 'log': log, 'source': source, 'environment': env}


def ready(server):
    deadline = time.monotonic() + 1800
    while time.monotonic() < deadline:
        if server['process'].poll() is not None:
            raise RuntimeError(f"Server exited: {server['folder']}")
        try:
            request(server['port'], '/health')
            event('server_ready', gpu=server['gpu'], mode=server['mode'])
            return
        except Exception:
            STOP.wait(2)
    raise TimeoutError('Server readiness')


def bench(server, name, count, input_len=8192, output_len=1024, concurrency=4, warmups=0, profile=False):
    folder, port = server['folder'], server['port']
    args = [PY, str(ENG / 'run_cli.py'), 'bench', 'serve', '--backend', 'vllm',
            '--base-url', f'http://127.0.0.1:{port}', '--endpoint', '/v1/completions',
            '--model', MODEL_ID, '--tokenizer', str(MODEL), '--dataset-name', 'random',
            '--random-input-len', str(input_len), '--random-output-len', str(output_len),
            '--random-range-ratio', '0', '--ignore-eos', '--temperature', '0',
            '--request-rate', 'inf', '--seed', '42', '--num-prompts', str(count),
            '--num-warmups', str(warmups), '--max-concurrency', str(concurrency),
            '--percentile-metrics', 'ttft,tpot,itl,e2el', '--metric-percentiles', '50,95,99',
            '--save-result', '--save-detailed', '--result-dir', str(folder),
            '--result-filename', name + '.json', '--disable-tqdm']
    if profile:
        args.append('--profile')
    (folder / (name + '_command.json')).write_text(json.dumps(args, indent=2) + '\n')
    event('bench_start', gpu=server['gpu'], mode=server['mode'], name=name, count=count,
          input_len=input_len, output_len=output_len, concurrency=concurrency)
    (folder / (name + '_before.prom')).write_text(request(port, '/metrics'))
    with (folder / (name + '.log')).open('w') as log:
        process = subprocess.run(args, cwd=server['source'], env=server['environment'],
                                 stdout=log, stderr=subprocess.STDOUT, timeout=2400)
    if process.returncode:
        raise RuntimeError(f'Benchmark failed: {folder}/{name}.log')
    data = json.loads((folder / (name + '.json')).read_text())
    assert data['completed'] == count and data['failed'] == 0
    assert set(data['input_lens']) == {input_len} and set(data['output_lens']) == {output_len}
    (folder / (name + '_after.prom')).write_text(request(port, '/metrics'))
    event('bench_complete', gpu=server['gpu'], mode=server['mode'], name=name,
          output_throughput=data['output_throughput'], mean_tpot_ms=data['mean_tpot_ms'])


def evaluate(server, questions):
    path = server['folder'] / 'gsm8k.jsonl'
    completed = 0
    def one(item):
        index, row = item
        body = {'model': MODEL_ID, 'messages': [{'role': 'user', 'content': row['question'] +
                '\nSolve step by step. End with #### followed by the final numeric answer.'}],
                'temperature': 0, 'seed': 42, 'max_tokens': 2048,
                'logprobs': True, 'top_logprobs': 5, 'stream': False,
                'chat_template_kwargs': {'enable_thinking': False}}
        response = json.loads(request(server['port'], '/v1/chat/completions', body))
        choice = response['choices'][0]
        text = choice['message']['content']
        target = row['answer'].split('####')[-1].strip().replace(',', '')
        tail = text.split('####')[-1] if '####' in text else text
        numbers = re.findall(r'-?\d[\d,]*(?:\.\d+)?', tail)
        predicted = numbers[-1].replace(',', '') if numbers else None
        return {'index': index, 'question': row['question'], 'expected': target,
                'predicted': predicted, 'correct': predicted == target, 'response': response}
    with path.open('w') as file, cf.ThreadPoolExecutor(max_workers=32) as pool:
        futures = [pool.submit(one, item) for item in enumerate(questions)]
        for future in cf.as_completed(futures):
            row = future.result()
            file.write(json.dumps(row) + '\n')
            file.flush()
            completed += 1
            if completed % 100 == 0:
                event('eval_progress', gpu=server['gpu'], mode=server['mode'], completed=completed)
    event('eval_complete', gpu=server['gpu'], mode=server['mode'], completed=completed)


def parallel(servers, function, *args, **kwargs):
    with cf.ThreadPoolExecutor(max_workers=len(servers)) as pool:
        futures = [pool.submit(function, server, *args, **kwargs) for server in servers]
        for future in futures:
            future.result()


def stop(server):
    if server['process'].poll() is None:
        os.killpg(server['process'].pid, signal.SIGINT)
        try:
            server['process'].wait(timeout=45)
        except subprocess.TimeoutExpired:
            os.killpg(server['process'].pid, signal.SIGTERM)
            server['process'].wait(timeout=30)
    server['log'].close()



def trace_counts(server):
    counts = {}
    paths = list((server['folder']/'trace').rglob('*.json.gz')) + list((server['folder']/'trace').rglob('*.pt.trace.json'))
    assert paths, 'Profiler produced no trace'
    for path in paths:
        opener = gzip.open if path.suffix == '.gz' else open
        with opener(path, 'rt') as file:
            trace = json.load(file)
        for item in trace['traceEvents']:
            if item.get('cat') == 'kernel':
                name = item.get('name','')
                counts[name] = counts.get(name,0)+1
    (server['folder']/'trace_kernel_counts.json').write_text(json.dumps(counts,indent=2)+'\n')
    packed = sum(count for name,count in counts.items() if '_packed_mlp_gemm' in name or '_packed_mlp_gemv' in name or '_lossless_packed_bf16_lm_head_kernel' in name)
    if server['mode'] == 'packed':
        assert packed > 0, 'Packed kernels absent from compiled serving decode'
        assert not any('unpack' in name for name in counts), 'Native reconstruction ran during decode profile'
    event('profile_intervention_verified',mode=server['mode'],packed_kernel_calls=packed)


def main():
    global PHASE, MODEL, MODEL_ID, MODEL_KEY, MODEL_REVISION, KV_BYTES
    # Keep formal timing independent of development probes on either GPU.
    deadline=time.monotonic()+1200
    while True:
        used=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
        if all(float(value)<100 for value in used.splitlines()):break
        if time.monotonic()>deadline:raise TimeoutError('Development probes still occupy GPU')
        STOP.wait(2)
    assert '1 passed' in (NXT/'context_model_generation_v1.log').read_text()
    assert 'COMPILED_COND_GATE_PASS' in (NXT/'context_model8b_v1.log').read_text()
    assert 'Failed' not in (NXT/'context_precommit_v3.log').read_text()
    OUT.mkdir(exist_ok=False)
    dataset = ROOT/'research/engine_opt_20261026/gsm8k_test.jsonl'
    questions=[json.loads(line) for line in dataset.read_text().splitlines()]
    assert len(questions)==1319
    configuration={
        'native_main':'ae3bc684e021f4b8c17c86a42e149b4482bc4ebe',
        'precision':'BF16; TP=1; MRV2; FLASH_ATTN; chunk8192; maxlen10240; maxseq32',
        'workloads':{'A':[8192,1024,1,16],'B':[8192,1024,4,32],
                     'C':[8192,128,1,16],'D':[8192,128,4,32]},
        'repeats':3, 'gpu_roles':[['native','packed'],['packed','native']],
        'warmups':'2*concurrency requests before each workload, excluded',
        'fixed_KV_bytes':{'4b':6*1024**3,'8b':5637144576},
        'extra_batch_controls':'128 input /128 output, concurrency7/16/32,64requests x3',
        'quality':'Full GSM8K 1319, C32, greedy, max2048, thinkingFalse, round1 eachmodel/mode',
        'dataset_sha256':hashlib.sha256(dataset.read_bytes()).hexdigest(),
        'formal_profiler':'disabled; separate8decodeiterations after16delay',
    }
    (OUT/'config.json').write_text(json.dumps(configuration,indent=2)+'\n')
    thread=threading.Thread(target=monitor,daemon=True);thread.start()
    servers=[]
    try:
        for MODEL_KEY,folder,MODEL_ID,MODEL_REVISION,KV_BYTES in [
            ('4b','Qwen3-4B-Instruct-2507','Qwen/Qwen3-4B-Instruct-2507','cdbee75f17c01a7cc42f958dc650907174af0554',6*1024**3),
            ('8b','Qwen3-8B','Qwen/Qwen3-8B','b968826d9c46dd6066d109eabc6255188de91218',5637144576),
        ]:
            MODEL=ROOT/'models'/folder
            for round_number,modes in enumerate([['native','packed'],['packed','native']],1):
                PHASE=f'{MODEL_KEY}_round{round_number}_startup'
                servers=[start(gpu,mode,round_number) for gpu,mode in enumerate(modes)]
                parallel(servers,ready)
                for name,concurrency,output_len,count in [('A',1,1024,16),('B',4,1024,32),('C',1,128,16),('D',4,128,32)]:
                    PHASE=f'{MODEL_KEY}_round{round_number}_{name}_warmup'
                    parallel(servers,bench,f'warmup_{name}',concurrency*2,concurrency=concurrency,output_len=output_len)
                    for repeat in range(1,4):
                        PHASE=f'{MODEL_KEY}_round{round_number}_{name}_{repeat}'
                        parallel(servers,bench,f'{name}_{repeat}',count,concurrency=concurrency,output_len=output_len)
                for concurrency in [7,16,32]:
                    PHASE=f'{MODEL_KEY}_round{round_number}_short_C{concurrency}'
                    parallel(servers,bench,f'warmup_short_C{concurrency}',concurrency*2,input_len=128,output_len=128,concurrency=concurrency)
                    for repeat in range(1,4):
                        parallel(servers,bench,f'short_C{concurrency}_{repeat}',64,input_len=128,output_len=128,concurrency=concurrency)
                PHASE=f'{MODEL_KEY}_round{round_number}_profile'
                parallel(servers,bench,'profile_decode4',4,output_len=64,profile=True)
                parallel(servers,bench,'profile_decode32',32,input_len=128,output_len=64,concurrency=32,profile=True)
                for server in servers:trace_counts(server)
                if round_number==1:
                    PHASE=f'{MODEL_KEY}_quality'
                    parallel(servers,evaluate,questions)
                for server in servers:stop(server)
                servers=[]
        (OUT/'COMPLETE.json').write_text(json.dumps({'complete':True})+'\n')
    finally:
        for server in servers:stop(server)
        STOP.set();thread.join(timeout=10)

if __name__=='__main__':main()
