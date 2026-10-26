# Packed BF16 MLP evidence

This branch stores evidence separately from the code PR. Source starts from pinned vLLM main ae3bc684 plus the original #55494 commits; the MLP increment is on `perf/packed-bf16-mlp-sm89`.

- `current_main_results.json`: all six-run summaries, per-run latencies, preemption deltas, GSM8K protocol/results, and decoded profile kernel tables.
- `kernel_bench_final.json`: eight real checkpoint layers per projection, three repeats, clean commit and source hashes. Earlier microbenchmarks are retained locally; this is the final frozen-source run.
- `throughput_and_dispatch.png`: median throughput with observed ranges and operator dispatch boundaries.
- `decode_native_gpu0.json.gz` / `decode_packed_gpu0.json.gz`: raw traces from the same physical GPU, eight decode steps, separate from timed benchmarks.
- `prefill_results.json`: same-GPU 8192-token prefill has identical kernel name/call-count maps; raw traces are kept in the Windows delivery.
- `run_current_main_ab.py` / `run_cli.py`: full official serving harness and NumPy seed wrapper, with machine paths explicitly recorded.

Formal serving/GSM8K used local d56c9ad / public 36706ab (identical tree 7ab21d4). Final lifecycle changes only normalize configured RL weight transfer to native and add regression coverage; the formal serving configuration has no RL weight transfer, and the GPU JIT computation and shared codec are unchanged; final guards also validate the input last dimension and avoid non-CUDA Triton imports. Later commits clarify docs and capture microbenchmark source hashes.

All original serving JSON, metric scrapes, logs, responses, environment/build records and traces are preserved under `D:\vllm_pr\工程交付` with checksum-verified archives. The 431 MiB raw serving folder is not committed to Git.
