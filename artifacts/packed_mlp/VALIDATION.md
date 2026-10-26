# Validation and provenance

- Native build: vLLM main `ae3bc684e021f4b8c17c86a42e149b4482bc4ebe`, built from source for SM89. Torch 2.14.0+cu126, Triton 3.8.0, build toolkit CUDA 12.9, driver 550.127.05, Python 3.12.15, Ubuntu 20.04/glibc 2.31. Newly built native binaries are shared only after an empty native-source diff check; binary SHA256 records are preserved locally.
- Serving/GSM8K source: local `d56c9ad170236bf015738ffedcd08ece8c39abde`, public `36706ab9f6bf742c3da949f32cbbb8998ae5b823`, identical tree `7ab21d43503781f3127d00e2d9646017ae81f488`.
- Final microbenchmark: clean local `c2c6612bd6501924d5d36ec50140d526b0e7955e`; its JSON records source hashes. Final docs-only commit follows that source. The two kernel files are unchanged between formal serving and the final candidate.
- Main validation: 16 tests passed with `VLLM_GPU_SYNC_CHECK=error`, including two real-model backend selections plus configured M2N fallback, all BF16 bit patterns, OpCheck and real checkpoint refit/CUDA Graph replay. A numerical token tie produces one expected `check_logprobs_close` warning; it is disclosed rather than suppressed.
- Dependency regression: 13 passed; the unrelated OPT E2E test was deselected. No new skip, xfail, tolerance relaxation or retry was added to hide a failure.
- Full pre-commit and commit hooks passed. The default lm-head and per-quant configuration compatibility tests run in the inherited suite.
- A warm-AOT regression initially failed with `KeyError: _packed_mlp_sign_mantissa` when weight transfer disabled preparation but retained the packed cache key. Normalizing the effective backend to `torch` fixes the actual entry-point test without clearing the AOT cache. Both failing and passing logs are preserved locally.
- Direct-checkpoint refits refresh fixed buffers; raw kernel-format updates are rejected for the packed backend. Configured RL weight-transfer engines use native layers because some transports update weights directly.
- H200 CI selects the tests, but cannot exercise SM89-specific preparation/performance. That coverage is supplied by local runs on both RTX 4090s; it is an explicit CI hardware gap.

The broad dense-weight integration question in RFC #63626 remains open. This is an incremental extension of #55494, not a competing claim to an original BF16 storage format. Performance qualification is limited to the stated model, shapes and hardware; native fallback does not remove the auxiliary-memory cost of an opted-in model.

This evidence is prepared for human review. Official submission must follow the human accountability requirements in the upstream AGENTS.md and contributing guide; this document does not attest that the human has reviewed all lines or personally run the tests.
