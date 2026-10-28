#!/usr/bin/env bash
# Run from either a baseline or candidate vLLM checkout installed in its own environment.
# This is the equivalent CLI recipe; exact audited orchestration is archived alongside it.
set -euo pipefail
: "${MODEL_PATH:?Set MODEL_PATH to the pinned Qwen3.5-9B snapshot}"
: "${RESULT_DIR:?Set RESULT_DIR to a new result directory}"
mkdir -p "$RESULT_DIR"
export VLLM_USE_V2_MODEL_RUNNER=1 VLLM_GPU_SYNC_CHECK=error
export VLLM_USE_RUST_FRONTEND=0 VLLM_USE_RUST_BENCH=0
if [[ ${1:-} == serve ]]; then
  exec vllm serve "$MODEL_PATH" --served-model-name Qwen/Qwen3.5-9B \
    --host 127.0.0.1 --port 18660 --language-model-only --dtype bfloat16 \
    --quantization fp8_per_tensor --tensor-parallel-size 1 --max-model-len 10240 \
    --max-num-seqs 64 --max-num-batched-tokens 8192 --kv-cache-memory-bytes 6442450944 \
    --gpu-memory-utilization .9 --seed 42 --no-enable-prefix-caching \
    --enable-chunked-prefill --generation-config vllm --attention-backend FLASH_ATTN \
    --compilation-config '{"max_cudagraph_capture_size":64}'
fi
# Example 8K input / 128 output / concurrency 17. Run warmup before recording.
exec vllm bench serve --backend vllm --base-url http://127.0.0.1:18660 \
 --endpoint /v1/completions --model Qwen/Qwen3.5-9B --tokenizer "$MODEL_PATH" \
 --dataset-name random --random-input-len 8192 --random-output-len 128 \
 --random-range-ratio 0 --ignore-eos --temperature 0 --request-rate inf --seed 42 \
 --num-prompts 34 --num-warmups 0 --max-concurrency 17 --ready-check-timeout-sec 0 \
 --probe-request-rate 0 --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,95,99 \
 --save-result --save-detailed --result-dir "$RESULT_DIR" --result-filename long8k_c17.json
