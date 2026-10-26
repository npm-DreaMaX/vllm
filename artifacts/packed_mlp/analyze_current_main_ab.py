"""Summarize immutable official benchmark JSON, raw profiles and full task eval."""
from collections import Counter, defaultdict
from pathlib import Path
import datetime as dt
import gzip
import json
import math
import re
import statistics as st

import numpy as np

ROOT = Path('/opt/data/private/FangWang/research/pr_engineering_20261026')
BASE = ROOT / 'serving_ab_v1'

def main():
    assert (BASE / 'COMPLETE.json').exists()
    result = {'benchmark': {}, 'quality': {}, 'profiling': [], 'requests': {},
              'limitations': ['GSM8K is zero-shot with our fixed prompt and numeric-answer extraction, not lm-eval 8-shot.',
                'Kernel sums are not a critical-path attribution.', 'NVML energy is an estimate of board energy over the client benchmark duration; 2s sampling and window alignment limit precision.']}
    keys = ['output_throughput', 'request_throughput', 'p50_ttft_ms', 'p95_ttft_ms',
            'p50_tpot_ms', 'p95_tpot_ms', 'p50_e2el_ms', 'p95_e2el_ms']
    folders = sorted(BASE.glob('round*'))
    events = [json.loads(s) for s in (BASE / 'events.jsonl').read_text().splitlines()]
    telemetry = [json.loads(s) for s in (BASE / 'gpu_telemetry.jsonl').read_text().splitlines()]
    def timestamp(value): return dt.datetime.fromisoformat(value).timestamp()
    samples = defaultdict(list)
    for record in telemetry:
        for line in record['rows']:
            parts = [float(s.strip()) for s in line.split(',')]
            samples[int(parts[0])].append((timestamp(record['utc']), *parts[1:]))
    starts, ends = {}, {}
    for event in events:
        if event['event'] in ('bench_start', 'bench_complete'):
            table = starts if event['event'] == 'bench_start' else ends
            table[(event['gpu'], event['mode'], event['name'], event['utc'][:13])] = event
    all_points = []
    for group in 'ABCD':
        rows = []
        for folder in folders:
            gpu = int(re.search(r'gpu(\d)', folder.name).group(1))
            mode = 'packed' if 'packed' in folder.name else 'native'
            for repeat in (1, 2, 3):
                name = f'{group}_{repeat}'
                file = folder / (name + '.json')
                data = json.loads(file.read_text())
                assert data['failed'] == 0 and data['completed'] == (32 if group in 'AC' else 64)
                row = {'file': str(file.relative_to(BASE)), 'gpu': gpu, 'mode': mode,
                       'round': int(folder.name[5]), 'repeat': repeat,
                       **{key: data[key] for key in keys}}
                end_events = [e for e in events if e['event'] == 'bench_complete' and e['gpu'] == gpu and e['mode'] == mode and e['name'] == name]
                assert len(end_events) == 1
                end = timestamp(end_events[0]['utc']); begin = end - data['duration']
                series = np.asarray(samples[gpu])
                ticks = np.r_[begin, series[(series[:, 0] > begin) & (series[:, 0] < end), 0], end]
                power = np.interp(ticks, series[:, 0], series[:, -1])
                row['nvml_board_joules_per_output_token_estimate'] = float(np.trapezoid(power, ticks) / sum(data['output_lens']))
                inside = series[(series[:, 0] >= begin) & (series[:, 0] <= end)]
                row['median_sm_clock_mhz'] = float(np.median(inside[:, 3]))
                row['median_temperature_c'] = float(np.median(inside[:, 5]))
                rows.append(row)
        summary = {}
        for mode in ['native', 'packed']:
            selected = [row for row in rows if row['mode'] == mode]
            summary[mode] = {key: {'median': st.median(r[key] for r in selected),
                'min': min(r[key] for r in selected), 'max': max(r[key] for r in selected)}
                for key in keys + ['nvml_board_joules_per_output_token_estimate']}
        summary['per_gpu'] = {}
        for gpu in [0, 1]:
            values = {mode: {key: st.median(r[key] for r in rows if r['gpu'] == gpu and r['mode'] == mode) for key in keys}
                      for mode in ['native', 'packed']}
            values['throughput_gain_pct'] = 100 * (values['packed']['output_throughput'] / values['native']['output_throughput'] - 1)
            values['ttft_p50_change_pct'] = 100 * (values['packed']['p50_ttft_ms'] / values['native']['p50_ttft_ms'] - 1)
            summary['per_gpu'][str(gpu)] = values
        result['benchmark'][group] = {'summary': summary, 'runs': rows}
        all_points.extend(rows)
    request_count = failed = 0
    preemptions = []
    for file in BASE.glob('round*/*.json'):
        if file.name.endswith('_command.json') or file.name == 'server_command.json': continue
        data = json.loads(file.read_text())
        if 'completed' not in data: continue
        request_count += data['completed']; failed += data['failed']
        def preempt(path):
            text = path.read_text()
            return sum(float(x) for x in re.findall(r'^vllm:num_preemptions_total(?:\{[^\n]*\})?\s+([\d.eE+-]+)', text, re.M))
        before, after = file.with_name(file.stem + '_before.prom'), file.with_name(file.stem + '_after.prom')
        if before.exists() and after.exists():
            preemptions.append({'file': str(file.relative_to(BASE)), 'delta': preempt(after) - preempt(before)})
    result['requests'] = {'completed_benchmark_json_requests_including_warmup_and_profile': request_count,
        'failed': failed, 'formal_ABCD_requests': 2304, 'fallback_control_requests': 512,
        'official_num_warmups_requests_not_in_json': 16,
        'preemption_total_delta': sum(row['delta'] for row in preemptions), 'preemptions_by_run': preemptions}
    quality = {}
    for folder in folders:
        rows = {}
        with (folder / 'gsm8k.jsonl').open() as file:
            for line in file:
                row = json.loads(line); choice = row['response']['choices'][0]
                rows[row['index']] = {'correct': row['correct'], 'predicted': row['predicted'],
                    'truncated': choice['finish_reason'] == 'length',
                    'completion_tokens': row['response']['usage']['completion_tokens'],
                    'tokens': [(x['token'], x['logprob']) for x in choice['logprobs']['content']]}
        quality[folder.name] = rows
        result['quality'][folder.name] = {'n': len(rows), 'correct': sum(x['correct'] for x in rows.values()),
            'accuracy': st.mean(x['correct'] for x in rows.values()), 'truncated': sum(x['truncated'] for x in rows.values()),
            'completion_tokens': sum(x['completion_tokens'] for x in rows.values())}
    native, packed = quality['round1_gpu0_native'], quality['round1_gpu1_packed']
    differences = np.array([int(packed[i]['correct']) - int(native[i]['correct']) for i in range(1319)])
    rng = np.random.default_rng(42)
    bootstrap = [float(differences[rng.integers(0, 1319, 1319)].mean()) for _ in range(10000)]
    discordant = Counter((native[i]['correct'], packed[i]['correct']) for i in range(1319))
    def compare(first, second):
        deltas = []; shared_decode_positions = 0; identical = 0
        for i in first.keys() & second.keys():
            a, b = first[i]['tokens'], second[i]['tokens']
            identical += [x[0] for x in a] == [x[0] for x in b]
            for position, (x, y) in enumerate(zip(a, b)):
                if x[0] != y[0]: break
                if position > 0:
                    deltas.append(abs(x[1] - y[1])); shared_decode_positions += 1
        return {'n': len(first.keys() & second.keys()), 'identical_generations': identical,
            'shared_decode_positions_excluding_prefill_first_token': shared_decode_positions,
            'shared_prefix_abs_logprob_delta_mean': st.mean(deltas),
            'shared_prefix_abs_logprob_delta_p99': float(np.quantile(deltas, .99)),
            'limitation': 'Only identical generated prefixes are compared; this is not teacher forcing and is selection-biased.'}
    result['quality']['paired_full_test'] = {'accuracy_delta': float(differences.mean()),
        'paired_bootstrap_95pct_interval': [float(x) for x in np.quantile(bootstrap, [.025, .975])],
        'native_only_correct': discordant[(True, False)], 'packed_only_correct': discordant[(False, True)],
        'generation_comparison': compare(native, packed)}
    result['quality']['native_repeat_control_64'] = compare(native, quality['round2_gpu1_native'])
    result['quality']['packed_repeat_control_64'] = compare(packed, quality['round2_gpu0_packed'])
    for folder in folders:
        for file in (folder / 'trace').glob('*.json.gz'):
            trace = json.load(gzip.open(file, 'rt'))
            kernels = [e for e in trace['traceEvents'] if e.get('cat') == 'kernel']
            counts = Counter(e['name'] for e in kernels)
            durations = defaultdict(float)
            for kernel in kernels: durations[kernel['name']] += kernel.get('dur', 0)
            packed_count = sum(v for k, v in counts.items() if '_packed_mlp_gemv' in k)
            assert packed_count == (576 if 'packed' in folder.name else 0), (folder, packed_count)
            result['profiling'].append({'file': str(file.relative_to(BASE)), 'gpu_kernel_count': len(kernels),
                'packed_mlp_kernel_count': packed_count, 'expected_packed_mlp_count': 72 * 8 if 'packed' in folder.name else 0,
                'packed_mlp_cumulative_gpu_us': sum(v for k, v in durations.items() if '_packed_mlp_gemv' in k),
                'total_cumulative_gpu_kernel_us': sum(durations.values()),
                'top20': [{'name': name, 'calls': counts[name], 'gpu_us': duration} for name, duration in sorted(durations.items(), key=lambda item: item[1], reverse=True)[:20]]})
    (ROOT / 'current_main_results.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'requests': {k:v for k,v in result['requests'].items() if k != 'preemptions_by_run'},
        'gains_pct_by_gpu': {g: {gpu: row['throughput_gain_pct'] for gpu, row in x['summary']['per_gpu'].items()} for g, x in result['benchmark'].items()},
        'quality': result['quality'], 'profiling_checks': [{k:v for k,v in x.items() if k != 'top20'} for x in result['profiling']]}, indent=2), flush=True)

if __name__ == '__main__': main()
