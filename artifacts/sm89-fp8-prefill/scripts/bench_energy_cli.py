"""Official bench CLI with GPU cumulative-energy counters at request boundaries.

Preheat separately, and disable this client's ready-check/warmup/probe requests.
BENCH_ENERGY_GPU_INDICES accepts physical GPU indices, including TP=2 runs.
"""
import json
import os
import sys
import time
from pathlib import Path

import pynvml
import vllm.benchmarks.serve as serve


def arg(name):
    return sys.argv[sys.argv.index(name)+1]


assert arg('--ready-check-timeout-sec') == '0'
assert arg('--num-warmups') == '0'
assert arg('--probe-request-rate') == '0'
expected = int(arg('--num-prompts'))
indices = os.environ.get('BENCH_ENERGY_GPU_INDICES', os.environ.get('BENCH_ENERGY_GPU_INDEX'))
assert indices, 'Explicit physical GPU indices are required'
gpus = [int(value) for value in indices.split(',')]
assert len(gpus) == len(set(gpus)), gpus
destination = Path(os.environ['BENCH_ENERGY_JSON'])
assert not destination.exists(), destination
pynvml.nvmlInit()
handles = [pynvml.nvmlDeviceGetHandleByIndex(gpu) for gpu in gpus]
state = {'gpus': gpus, 'gpu_uuids': [str(pynvml.nvmlDeviceGetUUID(h)) for h in handles],
         'expected_requests': expected, 'requests_started': 0, 'requests_finished': 0,
         'successful_requests': 0, 'actual_output_tokens': 0,
         'definition': 'Sum of selected GPUs device cumulative-energy differences, first formal dispatch through last formal completion; includes device idle energy inside that window, excludes client startup and separate warmup. Not wall-plug energy.'}


def read_energy():
    return [pynvml.nvmlDeviceGetTotalEnergyConsumption(handle) for handle in handles]


def wrap(original):
    async def request(*args, **kwargs):
        if state['requests_started'] == 0:
            state['start_monotonic_ns'] = time.monotonic_ns()
            state['start_wall_ns'] = time.time_ns()
            state['energy_start_mj_per_gpu'] = read_energy()
        state['requests_started'] += 1
        output = await original(*args, **kwargs)
        state['requests_finished'] += 1
        state['successful_requests'] += int(output.success)
        state['actual_output_tokens'] += int(output.output_tokens)
        if state['requests_finished'] == expected:
            state['energy_end_mj_per_gpu'] = read_energy()
            state['end_monotonic_ns'] = time.monotonic_ns()
            state['end_wall_ns'] = time.time_ns()
            state['energy_j_per_gpu'] = [(end-start)/1000 for start, end in
                zip(state['energy_start_mj_per_gpu'], state['energy_end_mj_per_gpu'])]
            state['energy_j'] = sum(state['energy_j_per_gpu'])
            state['window_seconds'] = (state['end_monotonic_ns']-state['start_monotonic_ns'])/1e9
            state['joules_per_output_token'] = (state['energy_j']/state['actual_output_tokens']
                                               if state['actual_output_tokens'] else None)
            destination.write_text(json.dumps(state, indent=2)+'\n')
        return output
    return request


for backend, original in list(serve.ASYNC_REQUEST_FUNCS.items()):
    serve.ASYNC_REQUEST_FUNCS[backend] = wrap(original)

from vllm.entrypoints.cli.main import main
main()
assert state['requests_started'] == state['requests_finished'] == expected, state
assert destination.exists(), state
