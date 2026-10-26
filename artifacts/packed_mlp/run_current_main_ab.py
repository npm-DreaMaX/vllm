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

ROOT = Path('/opt/data/private/FangWang')
ENG = ROOT / 'research/pr_engineering_20261026'
PY = str(ENG / '.venv/bin/python')
MODEL = ROOT / 'models/Qwen3-4B-Instruct-2507'
MODEL_ID = 'Qwen/Qwen3-4B-Instruct-2507'
STOP = threading.Event()
LOCK = threading.Lock()
OUT = ENG / os.environ.get('ENGINEERING_AB_NAME', 'serving_ab_v1')
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
    folder = OUT / f'round{round_number}_gpu{gpu}_{mode}'
    folder.mkdir()
    source = ENG / ('mlp' if mode == 'packed' else 'vllm')
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(gpu), PYTHONPATH=str(source),
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONUNBUFFERED='1',
               VLLM_USE_RUST_FRONTEND='0', VLLM_USE_RUST_BENCH='0',
               VLLM_USE_V2_MODEL_RUNNER='1')
    env['VLLM_CACHE_ROOT'] = str(ENG / 'cache' / mode / 'vllm')
    env['TORCHINDUCTOR_CACHE_DIR'] = str(ENG / 'cache' / mode / 'inductor')
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
            '--max-num-seqs', '16', '--max-num-batched-tokens', '8192',
            '--gpu-memory-utilization', '0.9', '--kv-cache-memory-bytes', str(6 * 1024**3),
            '--seed', '42', '--no-enable-prefix-caching', '--enable-chunked-prefill',
            '--generation-config', 'vllm', '--attention-backend', 'FLASH_ATTN',
            '--profiler-config', json.dumps(profiler)]
    if mode == 'packed':
        args += ['--kernel-config', json.dumps({'mlp_backend': 'lossless_packed'})]
    metadata = {'argv': args, 'gpu': gpu, 'mode': mode, 'source': str(source),
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
                'logprobs': True, 'top_logprobs': 5, 'stream': False}
        response = json.loads(request(server['port'], '/v1/chat/completions', body))
        choice = response['choices'][0]
        text = choice['message']['content']
        target = row['answer'].split('####')[-1].strip().replace(',', '')
        tail = text.split('####')[-1] if '####' in text else text
        numbers = re.findall(r'-?\d[\d,]*(?:\.\d+)?', tail)
        predicted = numbers[-1].replace(',', '') if numbers else None
        return {'index': index, 'question': row['question'], 'expected': target,
                'predicted': predicted, 'correct': predicted == target, 'response': response}
    with path.open('w') as file, cf.ThreadPoolExecutor(max_workers=4) as pool:
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


def main():
    global PHASE
    used = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used',
                                    '--format=csv,noheader,nounits'], text=True)
    assert all(float(value) < 100 for value in used.splitlines()), 'GPU already occupied'
    OUT.mkdir(exist_ok=False)
    dataset = ROOT / 'research/engine_opt_20261026/gsm8k_test.jsonl'
    questions = [json.loads(line) for line in dataset.read_text().splitlines()]
    assert len(questions) == 1319
    (OUT / 'config.json').write_text(json.dumps({
        'baseline_main': 'ae3bc684e021f4b8c17c86a42e149b4482bc4ebe',
        'model_revision': 'cdbee75f17c01a7cc42f958dc650907174af0554',
        'rounds': [['native', 'packed'], ['packed', 'native']],
        'formal_primary': 'A 8K/1K C1:32 requests x3; B 8K/1K C4:64 requests x3, per GPU per round',
        'formal_controls': 'C 8K/128 C1:32 x3; D 8K/128 C4:64 x3; C7/C16 128/128:64 each',
        'cache': 'Fixed 6 GiB KV per server, no prefix cache; capacity tradeoff measured separately',
        'evaluation': 'Full GSM8K test 1319 zero-shot greedy, max2048, C4 round1; first64 repeated with GPUs exchanged round2',
        'dataset_sha256': hashlib.sha256(dataset.read_bytes()).hexdigest(),
        'profiling': 'Separate requests after all timed repeats; 8 iterations after16 delay; full traces checked for intervention',
    }, indent=2) + '\n')
    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    servers = []
    try:
        for round_number, modes in enumerate([['native', 'packed'], ['packed', 'native']], 1):
            PHASE = f'round{round_number}_startup'
            servers = [start(gpu, mode, round_number) for gpu, mode in enumerate(modes)]
            parallel(servers, ready)
            PHASE = f'round{round_number}_warmup'
            parallel(servers, bench, 'warmup', 8, warmups=4)
            for name, concurrency, output_len, count in [('A', 1, 1024, 32), ('B', 4, 1024, 64),
                                                        ('C', 1, 128, 32), ('D', 4, 128, 64)]:
                PHASE = f'round{round_number}_{name}_warmup'
                parallel(servers, bench, f'warmup_{name}', concurrency * 2,
                         concurrency=concurrency, output_len=output_len)
                for repeat in range(1, 4):
                    PHASE = f'round{round_number}_{name}_{repeat}'
                    parallel(servers, bench, f'{name}_{repeat}', count, concurrency=concurrency, output_len=output_len)
            PHASE = f'round{round_number}_fallback'
            for concurrency in [7, 16]:
                parallel(servers, bench, f'fallback_C{concurrency}', 64, input_len=128,
                         output_len=128, concurrency=concurrency)
            PHASE = f'round{round_number}_profile'
            parallel(servers, bench, 'profile_decode4', 4, output_len=64, profile=True)
            PHASE = f'round{round_number}_evaluation'
            parallel(servers, evaluate, questions if round_number == 1 else questions[:64])
            for server in servers:
                stop(server)
            servers = []
        PHASE = 'complete'
        (OUT / 'COMPLETE.json').write_text(json.dumps({'complete': True}) + '\n')
    finally:
        for server in servers:
            stop(server)
        STOP.set()
        thread.join(timeout=10)


if __name__ == '__main__':
    main()
