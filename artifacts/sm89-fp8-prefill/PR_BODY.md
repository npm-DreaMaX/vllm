<!-- markdownlint-disable -->

## Overview

Reduce the pipeline depth from five to three stages for large 128×128×64 SM89 FP8 GEMMs. This lowers prefill latency for Qwen3.5-9B on RTX4090 while retaining the existing tile, accumulation operator and epilogue.

## Claims

- For the measured 8K-input workloads, median mean TTFT decreases **4.51–7.14%**. At 8K input / 128 output / concurrency 17, output throughput increases **3.85%**; each card individually improves (4.09% and 3.53%).
- Shared memory for the affected tile decreases **80 → 48 KiB**. Native rotating-weight microbenchmarks for the three 8K prefill projection shapes improve **2.36–5.53%**.
- Short-input decode throughput is effectively unchanged (**+0.07%**); long-output throughput improves only **0.64%**. No universal speedup is claimed.

## Validation

**Same-precision native A/B:** Qwen3.5-9B revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`, online FP8 per-tensor weights / dynamic FP8 activations / BF16 output, TP=1, text-only, FA2, MRV2, `VLLM_GPU_SYNC_CHECK=error`, no MTP or prefix caching. Each server uses one RTX4090 (24GB, default clocks), 6GiB KV cache, max length 10240, max sequences 64, batched tokens 8192, balanced CUDA graphs capped at 64. Both arms use identical non-core native libraries. This is not an FP8-versus-BF16 comparison.

Two simultaneous servers are measured, then their cards are swapped. Each arm has **two starts, one per card**, with three repeats per workload/start after sustained warmup. The last control has one repeat/start. All **76 measurements / 1,520 requests** succeeded without foreign GPU processes. Inputs have exact random token lengths, seed 42, greedy decoding and ignored EOS. No profiling is active during timings.

Values below are medians of the per-run metrics (six runs per arm, last control two). TTFT includes serving/queue effects. Finite bursts do not model every production arrival distribution.

| Input / output / concurrency | Mean TTFT, ms: base → change | Output tokens/s: base → change |
|---|---:|---:|
| 8192 / 16 / 1 | 593.6 → 566.8 (-4.51%) | 20.61 → 21.33 (+3.48%) |
| 8192 / 128 / 1 | 591.5 → 564.5 (-4.57%) | 57.38 → 58.11 (+1.28%) |
| 8192 / 128 / 4 | 1631.5 → 1553.6 (-4.78%) | 129.83 → 133.24 (+2.62%) |
| 8192 / 128 / 17 | 3979.3 → 3695.0 (-7.14%) | 189.67 → 196.97 (+3.85%) |
| 2048 / 128 / 4 | 455.3 → 435.1 (-4.45%) | 229.20 → 231.79 (+1.13%) |
| 256 / 512 / 17 (decode control) | 336.0 → 318.8 (-5.11%) | 1074.92 → 1075.68 (+0.07%) |
| 8192 / 1024 / 4 (long-output control) | 1635.3 → 1552.9 (-5.04%) | 244.57 → 246.12 (+0.64%) |

<details><summary>Observed throughput ranges; not confidence intervals</summary>

| Input / output / concurrency | Baseline min–max | Candidate min–max |
|---|---:|---:|
| 8192 / 16 / 1 | 20.41–20.71 | 21.25–21.44 |
| 8192 / 128 / 1 | 56.85–57.60 | 58.02–58.31 |
| 8192 / 128 / 4 | 129.44–129.96 | 133.13–133.57 |
| 8192 / 128 / 17 | 189.38–189.80 | 196.18–197.62 |
| 2048 / 128 / 4 | 228.58–229.68 | 231.27–232.08 |
| 256 / 512 / 17 (decode control) | 1071.98–1076.29 | 1073.16–1077.13 |
| 8192 / 1024 / 4 (long-output control) | 244.15–244.99 | 246.10–246.15 |

Repeated measurements within one start are not independent starts. Full TTFT/TPOT/ITL/energy vectors, ranges and per-card comparisons are in [analysis.json](https://github.com/npm-DreaMaX/vllm/tree/evidence/sm89-fp8-prefill-20261028/artifacts/sm89-fp8-prefill/serve_9b_final_v1/analysis.json). The short-input TTFT difference varies by card and is not a claimed decode improvement.

</details>

| Correctness / quality | Baseline | Candidate |
|---|---:|---:|
| Official FP8 / CUTLASS subset tests | 265 passed, 31 skipped | 265 passed, 31 skipped |
| Additional native output hashes | 56 cases | 56/56 match baseline; graph/direct equal within each arm |
| Full GSM8K, 5-shot chat, thinking disabled, greedy, max 512 tokens | 1215/1319 (92.12%) | 1215/1319 (92.12%) |
| GSM8K exact generated token sequences | reference | 1319/1319 |
| GSM8K length-truncated responses | 26 | 26 |

GSM8K uses the official evaluator with a fixed offline queue and synchronous scheduling; it is a quality check, not a throughput comparison. Existing test tolerances are unchanged. The new test shapes cover M=1023/1024/1025 and a wide prefill projection with the existing scale/bias parameterization. Both versions pass because this is tuning rather than a correctness fix.

```bash
.venv/bin/python -m pytest tests/kernels/quantization/test_cutlass_scaled_mm.py -k 'fp8 or cutlass_subset' -v
pre-commit run --files csrc/quantization/w8a8/cutlass/scaled_mm_c2x_sm89_fp8_dispatch.cuh tests/kernels/quantization/test_cutlass_scaled_mm.py
```

[Exact serving commands and reproduction instructions](https://github.com/npm-DreaMaX/vllm/tree/evidence/sm89-fp8-prefill-20261028/artifacts/sm89-fp8-prefill/scripts/README.md), [kernel/profile figures](https://github.com/npm-DreaMaX/vllm/tree/evidence/sm89-fp8-prefill-20261028/artifacts/sm89-fp8-prefill/figures), [native qualification](https://github.com/npm-DreaMaX/vllm/tree/evidence/sm89-fp8-prefill-20261028/artifacts/sm89-fp8-prefill/qualification), [GSM8K results](https://github.com/npm-DreaMaX/vllm/tree/evidence/sm89-fp8-prefill-20261028/artifacts/sm89-fp8-prefill/offline_gsm8k_9b/COMPLETE.json), and [full review notes](https://github.com/npm-DreaMaX/vllm/tree/evidence/sm89-fp8-prefill-20261028/artifacts/sm89-fp8-prefill/REVIEW.md).

## Details

Profiling the 9B model identified large FP8 linear kernels as a substantial prefill cost. The existing five-stage 128×128×64 kernel reserves 80KiB shared memory per CTA; three stages use 48KiB, allowing higher theoretical residency within Ada's resource limits. Achieved occupancy was not measured. Native launch traces document the resource change; performance comes from separate unprofiled measurements. Prototype-extension timing was not used to claim native speedup.

The guard is **M>=1024, N>=4096, K>=2048**. The existing 256×128 branch for 4096<N<=8192 already uses three stages and is unaffected. Below-threshold shapes, INT8, batch-invariant dispatch and other architecture dispatchers are unchanged. The fallback, FP8 fast-accumulation operator and math tile remain the same. No new allocation, weight layout, host sync, model whitelist or user setting is added.

Limitations: only RTX4090 was measured; L4/L40/L40S remain unvalidated. A shallower pipeline can trade latency hiding for residency on unmeasured shapes. The K=32768 control is neutral. The native core grows by 1.28MB (~0.88%). Buildkite selects these tests on B200, which does not exercise this SM89 branch; local RTX4090 validation fills that gap, and the human submitter must confirm/rerun it before upstream publication.

Base: `88d3995ea18de47ba5c306789aa615474ad74592`; candidate: `115a32c53a6a66c8d8e407c3f8858b2290a3b071`. Runtime library hashes and dependency audit are included. Compilation preceded a whitespace-only clang-format change; [formatting_audit.json](https://github.com/npm-DreaMaX/vllm/tree/evidence/sm89-fp8-prefill-20261028/artifacts/sm89-fp8-prefill/formatting_audit.json) records both source hashes and confirms identical non-whitespace bytes. `pre-commit` and `git diff --check` pass.

Duplicate check on 2026-10-28 found no identical open change; the upstream dispatcher still matched the pinned base. [#61224](https://github.com/vllm-project/vllm/pull/61224) concerns block-scaled FP8, whereas this uses the existing tensor/channel-scale C2x path. [#62128](https://github.com/vllm-project/vllm/pull/62128) concerns INT8 small-M tiles, whereas this changes FP8 large-M pipeline depth. [Search record](https://github.com/npm-DreaMaX/vllm/tree/evidence/sm89-fp8-prefill-20261028/artifacts/sm89-fp8-prefill/diagnosis/upstream_overlap_check.json).

**AI assistance:** Codex (OpenAI) assisted investigation, implementation and validation. Human review and submission remain the responsibility of the submitter; this prepared personal-fork draft is not an assertion that upstream human-review requirements have already been fulfilled.

---

<details>
<summary> Pull Request Checklist </summary>

- [x] I used vLLM's `/pr-checklist` skill. (Mandatory for agents, optional for humans).
- [x] AI assistance was used during the creation of this PR.

- [x] **Design Fit:** Minimizes impact on core components, reuses existing functionality, and justifies added complexity.
- [x] **Testing and Validation:** Validates the change and ensures any added tests are meaningful and reliable, with CI coverage or documented CI resource constraints and validation performed outside CI.
- [x] **Code Quality and Style:** Keeps code and comments clear and concise, and updates relevant documentation and examples.
- [x] **Pull Request Contents:** Includes a brief summary and relevant links, supports claims with evidence, explains root causes and implementation trade-offs, and follows the contributing guide.
</details>

**BEFORE SUBMITTING, PLEASE READ <https://docs.vllm.ai/en/latest/contributing>** (anything written below this line will be removed by GitHub Actions)
