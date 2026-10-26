"""Numerical probe: native 8K prefill, 1K forced decode, raw logprobs."""
from pathlib import Path
import hashlib
import json
import os
import sys

import numpy as np
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

ROOT = Path('/opt/data/private/FangWang/research/pr_engineering_20261026')
MODEL = '/opt/data/private/FangWang/models/Qwen3-4B-Instruct-2507'

def main():
    mode = sys.argv[1]
    out = ROOT / 'fixed_token_raw_logprobs_v1'
    out.mkdir(exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    dataset = Path('/opt/data/private/FangWang/research/engine_opt_20261026/gsm8k_test.jsonl')
    questions = [json.loads(line) for line in dataset.read_text().splitlines()][:32]
    text = '\n\n'.join('Question: ' + row['question'] + '\nAnswer: ' + row['answer'] for row in questions)
    pool = tokenizer.encode(text, add_special_tokens=False)
    ids = (pool * (8192 // len(pool) + 2))[:8192]
    cases = []
    for offset, text in [(0, ' the'), (127, '\n')]:
        token = tokenizer.encode(text, add_special_tokens=False)
        assert len(token) == 1
        prompt = ids[offset:] + ids[:offset]
        cases.append((prompt, token[0]))
    kwargs = {'mlp_backend': 'lossless_packed'} if mode == 'packed' else {}
    llm = LLM(model=MODEL, dtype='bfloat16', max_model_len=10240,
        tensor_parallel_size=1, max_num_seqs=4, max_num_batched_tokens=8192,
        gpu_memory_utilization=.9, kv_cache_memory_bytes=6 * 1024**3,
        enable_prefix_caching=False, enable_chunked_prefill=True, seed=42,
        logprobs_mode='raw_logprobs', kernel_config=kwargs,
        compilation_config={'cudagraph_capture_sizes': [1, 2, 4]},
        attention_config={'backend': 'FLASH_ATTN'})
    rows = []
    for repeat in range(2 if mode == 'native' else 1):
        for index, (prompt, token) in enumerate(cases):
            params = SamplingParams(temperature=0, seed=42, max_tokens=1024,
                ignore_eos=True, logprobs=5, allowed_token_ids=[token])
            result = llm.generate([{'prompt_token_ids': prompt}], params, use_tqdm=False)[0].outputs[0]
            assert result.token_ids == [token] * 1024
            raw = [entry[token].logprob for entry in result.logprobs]
            ranks = [entry[token].rank for entry in result.logprobs]
            assert min(raw) < -.1 and max(ranks) > 5, 'Reject processed or masked logprobs'
            rows.append({'case': index, 'repeat': repeat, 'prompt_tokens': prompt,
                'forced_token_id': token, 'generated_token_ids': result.token_ids,
                'raw_logprobs': raw, 'raw_ranks': ranks,
                'prompt_sha256': hashlib.sha256(np.asarray(prompt, dtype=np.int32).tobytes()).hexdigest()})
            (out / (mode + '.json')).write_text(json.dumps({'mode': mode,
                'protocol': '8192-token native prefill, 1024 single-request forced decode tokens; allowed_token_ids masks sampling but raw_logprobs are measured before that mask.',
                'scope': 'Numerical probe with artificial repeated target tokens; not task quality, not a serving performance result.',
                'rows': rows}, indent=2) + '\n')
    print('FIXED_TOKEN_RAW_LOGPROB_PROBE_PASS', mode, flush=True)

if __name__ == '__main__': main()
