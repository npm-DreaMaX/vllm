# Reproduction

Source base: `88d3995ea18de47ba5c306789aa615474ad74592`; candidate: `115a32c53a6a66c8d8e407c3f8858b2290a3b071`.
Model: `Qwen/Qwen3.5-9B`, revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`.

Build each revision's native extension, using the repository's `docs/contributing/incremental_build.md`. Keep the Python dependency environment and all other native extensions identical. The measured environment used PyTorch 2.14.0+cu126, Triton 3.8, CUDA compiler 12.9, driver 550.127.05, FA2 and one RTX4090 per server. A precompiled wheel alone will not incorporate the C++ change. Original build log, source hashes and runtime DSO audits are provided in the evidence.

Run the existing suite from both checkouts, using the same modified test file:

```bash
.venv/bin/python -m pytest tests/kernels/quantization/test_cutlass_scaled_mm.py -k 'fp8 or cutlass_subset' -v
pre-commit run --files csrc/quantization/w8a8/cutlass/scaled_mm_c2x_sm89_fp8_dispatch.cuh tests/kernels/quantization/test_cutlass_scaled_mm.py
```

`reproduce_serving.sh serve` starts the equivalent server; run `reproduce_serving.sh` in another terminal after warmup. Set `MODEL_PATH`, `RESULT_DIR` and `CUDA_VISIBLE_DEVICES` explicitly, and activate the intended checkout's environment first. This recipe omits the dormant profiler configuration and the original audit/energy wrappers; it is not represented as a separately executed benchmark.

The executed experiment is `run_prefill_9b_final.py`: two concurrent servers, then swap cards, three repeats per workload and startup, 45 seconds sustained warmup per startup. `protocol.json` and each `server_command.json` / `*_command.json` are authoritative for all arguments. The longer-output control runs once per startup. The final JSON reports all 76 measurements, not just favorable cases. Two starts per arm do not establish statistical independence of six repeats.

The original orchestration files retain their original workspace layout (public copies replace the private prefix with `/workspace`). They are an audit archive, not an installer: `run_native_wave_pilot.py` supplies shared benchmarking functions; `prefill_9b_server.py` overrides its old default model/configuration for 9B. Its unused 0.8B startup function was not called. `run_context_final_ab.py` supplies HTTP/telemetry helpers. Relocate those files and update the path constants together when reusing the exact harness. The sibling paths referenced in the scripts are preserved in recorded commands; model weights and compiled libraries are not bundled.

`qualify_native_prefill.py` performs 56 full-output hashes and graph/direct checks per arm, and seven rotating-weight microbenchmarks. `profile_native_prefill.py` records native launch resources; its profiled durations are not used as performance results. `offline_gsm8k_9b.py` uses the repository's official GSM8K helper with a fixed queue; `run_quality_9b_parallel.py` runs both arms on separate cards. Quality throughput is not compared. `analyze_final.py <serve_9b_final_v1>` reproduces the serving summary and checks counts/failures.

Every output folder is single-use. Preserve unsuccessful attempts instead of overwriting them. Ensure both GPUs are idle before repeating the crossover benchmark; the recorded run additionally monitors for foreign GPU processes.
