"""Pinned-main Qwen3.5-9B FP8 TP=1 baseline/candidate; native operators."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import threading
import time

import run_native_wave_pilot as pilot
h = pilot.h
ROOT = pilot.ROOT
h.OUT = ROOT / 'qwen35_9b_bf16_tp2_baseline_v1'
h.MODEL = h.ROOT / 'models/Qwen3.5-9B'
h.MODEL_ID = 'Qwen/Qwen3.5-9B'


def start(mode, run_id, gpu=0):
    metadata = json.loads((ROOT / 'qwen35_9b_download_complete.json').read_text())
    h.MODEL_REVISION = metadata['revision']
    manifest = json.loads((ROOT / 'prefill_pr_v1/ready.json').read_text())
    runtime = ROOT / ('native_baseline_runtime' if mode == 'native' else 'vllm_fp8')
    core_sha = manifest['baseline_core_sha256' if mode == 'native' else 'candidate_core_sha256']
    folder = h.OUT / f'{run_id}_{mode}_gpu{gpu}_tp1'
    folder.mkdir()
    trace = folder / 'trace'
    trace.mkdir()
    env = os.environ.copy()
    for key in ['ENGINE_OPT_MODE', 'ADA_SKINNY_MODE', 'FP8_WAVE_MODE']:
        env.pop(key, None)
    env.update(CUDA_VISIBLE_DEVICES=str(gpu), PYTHONPATH=str(runtime),
               RESEARCH_RUNTIME_ROOT=str(runtime),
               RESEARCH_CORE_SHA256=core_sha,
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONUNBUFFERED='1',
               VLLM_USE_RUST_FRONTEND='0', VLLM_USE_RUST_BENCH='0',
               VLLM_USE_V2_MODEL_RUNNER='1', VLLM_GPU_SYNC_CHECK='error',
               VLLM_CACHE_ROOT=str(ROOT/'qwen35_9b_cache/fp8_prefill_v1'/mode/'vllm'),
               TORCHINDUCTOR_CACHE_DIR=str(ROOT/'qwen35_9b_cache/fp8_prefill_v1'/mode/'inductor'),
               VLLM_NCCL_SO_PATH='/workspace/envs/vllm-dev/lib/python3.12/site-packages/nvidia/nccl/lib/libnccl.so.2')
    profiler = dict(profiler='torch', torch_profiler_dir=str(trace),
                    torch_profiler_with_stack=False, torch_profiler_record_shapes=True,
                    ignore_frontend=True, detailed_trace_annotation=True,
                    delay_iterations=24, max_iterations=12)
    port = 18660+gpu
    args = ['taskset', '-c', '8-15' if gpu==0 else '16-23', h.PY, str(ROOT/'native_main_cli.py'),
            'serve', str(h.MODEL), '--served-model-name', h.MODEL_ID,
            '--host', '127.0.0.1', '--port', str(port), '--language-model-only',
            '--dtype', 'bfloat16', '--quantization', 'fp8_per_tensor', '--tensor-parallel-size', '1',
            '--max-model-len', '10240', '--max-num-seqs', '64',
            '--max-num-batched-tokens', '8192', '--gpu-memory-utilization', '0.9',
            '--kv-cache-memory-bytes', str(6*1024**3), '--seed', '42',
            '--no-enable-prefix-caching', '--enable-chunked-prefill',
            '--generation-config', 'vllm', '--attention-backend', 'FLASH_ATTN',
            '--profiler-config', json.dumps(profiler), '--compilation-config',
            json.dumps({'max_cudagraph_capture_size': 64})]
    record = dict(argv=args, environment={key:env[key] for key in [
        'CUDA_VISIBLE_DEVICES','PYTHONPATH','RESEARCH_RUNTIME_ROOT','RESEARCH_CORE_SHA256',
        'VLLM_GPU_SYNC_CHECK','VLLM_CACHE_ROOT','TORCHINDUCTOR_CACHE_DIR']},
        main_pin=manifest['main_pin'], model_revision=h.MODEL_REVISION,
        precision='Online FP8 per-tensor weights, dynamic FP8 activations, BF16 output; text-only; no MTP',
        gpu_topology='One RTX 4090; no tensor-parallel communication',
        scope='Native source-built same-precision A/B; only guarded FP8 dispatch differs.')
    (folder/'server_command.json').write_text(json.dumps(record,indent=2)+'\n')
    log = (folder/'server.log').open('w')
    process = subprocess.Popen(args, cwd=runtime, env=env, stdout=log,
                               stderr=subprocess.STDOUT, start_new_session=True)
    return dict(gpu=gpu, mode=mode, folder=folder, port=port,
                process=process, log=log, source=runtime, environment=env)


def wait_for_inputs():
    deadline = time.monotonic()+8*3600
    for name in ['qwen35_9b_download_complete.json', 'native_baseline_snapshot.json']:
        while not (ROOT/name).exists():
            if time.monotonic()>deadline:
                raise TimeoutError(name)
            time.sleep(10)
    # Optional handoff from explicitly scheduled GPU qualification.
    handoff = os.environ.get('RESEARCH_GPU_HANDOFF')
    while handoff and not Path(handoff).exists():
        if time.monotonic()>deadline:
            raise TimeoutError(handoff)
        time.sleep(5)

