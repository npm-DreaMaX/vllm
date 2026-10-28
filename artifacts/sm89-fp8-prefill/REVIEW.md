# Review notes: SM89 FP8 prefill

## 1. Design fit

The patch changes the existing SM89 CUTLASS 2.x FP8 dispatcher, with 10 production insertions and 3 deletions. It reuses the same 128x128x64 tile, warp shape, instruction shape, fast-accumulation operator, epilogue and fallback. No model-specific branch, scheduler change, new dependency, new API or host synchronization is introduced. The extra template argument defaults to 5 and only selects 3 for M>=1024, N>=4096 and K>=2048. The 256x128 branch for 4096<N<=8192 already uses 3 stages and is unchanged. Small-M and below-threshold shapes stay on the original path. The top-level entry explicitly dispatches this function only when device capability is 8.9; batch-invariant dispatch is separate.

The motivation comes from Qwen3.5-9B profiling, not from the earlier 0.8B positive result. Existing 9B profiling identifies large FP8 GEMMs as a substantial prefill cost. Reducing pipeline depth lowers shared-memory usage from 80 to 48 KiB for the affected tile. This permits higher theoretical residency on Ada; it is not a measurement of achieved occupancy. Only native before/after builds support the claimed performance gains. Standalone prototype gains were larger and are intentionally not substituted for native results.

Tradeoffs: shallower pipelines may expose memory latency on unmeasured shapes or hardware. The tested large-K control is neutral, not positive. L4/L40/L40S were unavailable; do not generalize RTX4090 measurements to them. The native core grew from 145,832,896 to 147,117,176 bytes (+1,284,280 bytes, about 0.88%). The broader dispatch guard merits reviewer attention; it avoids a model whitelist but is not an exhaustive autotuner. This is an internal tuning change under an existing FP8 option; it adds no user-facing configuration or model/hardware support.

Duplicate-work check (2026-10-28): open PR keyword searches and the upstream header were checked. The header still matches the pinned base. PR #61224 targets block-scaled FP8 with per-token-group activations and block weights; this patch targets the existing tensor/channel-scale C2x path. PR #62128 targets INT8 small-M tile selection; this patch targets FP8 large-M pipeline depth. No identical open PR was identified. No linked bug/issue is claimed to be fixed. Recheck before upstream publication if time has passed.

## 2. Testing and validation

Both baseline and candidate: **265 passed, 31 skipped**, 459 deselected, using the same augmented official test file and `-k 'fp8 or cutlass_subset'`. The four new FP8 parameter shapes exercise the M threshold and wide projections through the actual operator; the existing scale/bias combinations and tolerances are retained. This is performance tuning: the baseline also passes correctness tests, as expected. No timing assertion or machine-specific speed threshold is added to CI.

Additional native qualification: 56 output SHA256 hashes agree across builds; graph replay equals direct invocation in each case. Cases include FP16/BF16 output, scale/bias variants, guard edges and K through 32768. This establishes equality on the tested cases, not a theorem for every input.

Final serving and full 1,319-question GSM8K results are in `PR_BODY.md` and the raw evidence. Serving uses MRV2 and `VLLM_GPU_SYNC_CHECK=error`. All baseline/candidate runs use the same FP8 precision and the same other native libraries. Two card-swapped starts per arm with three repeats per start measure variability but do not imply six independent starts. Finite bursts differ from a production arrival distribution; this is text-only, TP=1, FA2, no MTP, no prefix caching. Quality holds scheduling fixed with the offline synchronous scheduler; it is not a speed comparison or a comparison against BF16 model quality.

Buildkite's existing kernel group selects this file with `fp8 or cutlass_subset`; its B200 runner cannot exercise the changed SM89 branch. The exact SM89 path was validated locally on RTX4090. No claim of upstream CI completion is made. The human submitter must confirm the recorded validation and rerun relevant tests on the final revision before upstream publication.

## 3. Code quality and style

`pre-commit` and `git diff --check` passed. Initial formatting failure is preserved alongside the successful rerun. Native compilation preceded a clang-format-only split of two template-argument lines: the audit proves all non-whitespace bytes match; the actual loaded native core hash is recorded for every experiment. Do not claim that the final formatted file's byte hash equals the compilation input hash. Seven matching native library paths were audited; only the core library differs.

Two source files changed; no generated evidence or benchmark harness enters the source PR. Full raw artifacts, failed exploratory runs and frozen earlier candidates are retained locally. Public evidence removes identifiers and records original/published hashes without changing numerical observations. Test code reuses the existing parameterization rather than adding a separate benchmark test.

## 4. Pull request contents and human handoff

`PR_BODY.md` follows the repository template. It states configuration, absolute metrics, repeats and variability, correctness, model quality, controls and limitations; it discloses AI assistance. The source commit is `115a32c53a6a66c8d8e407c3f8858b2290a3b071`, with a `Generated-by: Codex (OpenAI)` trailer. No human sign-off is fabricated.

The supplied PowerShell script pushes only source/review-base/evidence branches to `npm-DreaMaX/vllm`, and creates a personal-fork Draft PR. It neither force-pushes nor opens an upstream PR. Publishing was explicitly left to the user. Its syntax and bundle transport are validated locally; GitHub writes were not executed.

Before opening an official upstream PR, the human submitter must review every changed line, understand and defend the threshold and pipeline choice, rerun relevant tests and comply with the contribution sign-off requirements. This comes directly from [vLLM AGENTS.md](https://github.com/vllm-project/vllm/blob/88d3995ea18de47ba5c306789aa615474ad74592/AGENTS.md): “The submitting human must review every changed line and run relevant tests.” The [pr-checklist skill](https://github.com/vllm-project/vllm/blob/88d3995ea18de47ba5c306789aa615474ad74592/.agents/skills/pr-checklist/SKILL.md) was applied; these four sections document its findings. These human responsibilities are not reported as already completed.
