"""Use GPU execution scopes to exclude mixed-prefill steps from decode."""
import collections
import gzip
import json
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).parent
SOURCE = ROOT / 'qwen35_9b_fp8_diagnosis_v2'


def category(name):
    if 'cutlass::float_e4m3_t' in name:
        return 'FP8 linear'
    if 'gemm' in name or 'gemvx' in name:
        return 'BF16 linear / lm_head'
    if 'fused_recurrent_gated_delta_rule' in name:
        return 'GDN recurrent decode'
    if 'chunk_' in name or 'recompute_w_u' in name or 'inverse_kernel' in name:
        return 'GDN chunk prefill'
    if 'flash_fwd' in name:
        return 'Full attention'
    if 'conv1d' in name or 'post_conv' in name:
        return 'GDN convolution and preparation'
    if name.startswith('triton_'):
        return 'Compiled pointwise / reductions'
    return 'Other'


def main():
    result = {}
    for path in sorted(SOURCE.rglob('*.pt.trace.json.gz')):
        events = json.load(gzip.open(path, 'rt'))['traceEvents']
        scopes = [e for e in events if e.get('cat') == 'gpu_user_annotation'
                  and e.get('name','').startswith('execute_')]
        match = re.match(r'decode_c(\d+)_', path.name)
        if match:
            c = match.group(1)
            selected = [e for e in scopes if e['name'].startswith(f'execute_{c}_context_0(')
                        and f'_generation_{c}(' in e['name']]
            key = f'decode_c{c}'
        else:
            selected = [e for e in scopes if e['name'].startswith('execute_8192_context_1(')]
            key = 'prefill_8192_c1'
        assert selected
        steps = []
        kernels = [e for e in events if e.get('cat') == 'kernel' and e.get('ph') == 'X']
        for scope in selected:
            rows = [e for e in kernels if scope['ts'] <= e['ts']
                    and e['ts'] + e['dur'] <= scope['ts'] + scope['dur'] + 0.01]
            groups = collections.defaultdict(float)
            for e in rows:
                groups[category(e['name'])] += e['dur']
            steps.append(dict(name=scope['name'],scope_us=scope['dur'],
                              kernels=len(rows),summed_kernel_us=sum(groups.values()),
                              categories_us=dict(groups)))
        cats=sorted({k for step in steps for k in step['categories_us']})
        result[key]=dict(file=str(path.relative_to(ROOT)),selected_steps=len(steps),
                         excluded_steps=len(scopes)-len(steps),steps=steps,
                         median_scope_us=statistics.median(s['scope_us'] for s in steps),
                         mean_categories_us={k:statistics.mean(s['categories_us'].get(k,0) for s in steps) for k in cats},
                         median_categories_us={k:statistics.median(s['categories_us'].get(k,0) for s in steps) for k in cats})
        print(key,'steps',len(steps),'excluded',len(scopes)-len(steps),
              {k:round(v/1000,3) for k,v in result[key]['median_categories_us'].items()})
    (ROOT/'phase_analysis.json').write_text(json.dumps({
        'warning':'Separate instrumented runs. Kernel sums are not request latency; no speedup claims. Categories are name-based, raw names retained in trace_analysis.',
        'results':result},indent=2)+'\n')


if __name__=='__main__':
    main()
