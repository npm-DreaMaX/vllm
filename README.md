# SM89 FP8 prefill: Qwen3.5-9B validation evidence

This branch contains evidence only; it is deliberately separate from the two-file source change.

- [PR description, results and limitations](artifacts/sm89-fp8-prefill/PR_BODY.md)
- [Review notes and human handoff](artifacts/sm89-fp8-prefill/REVIEW.md)
- [Reproduction instructions](artifacts/sm89-fp8-prefill/scripts/README.md)
- [Final serving measurements](artifacts/sm89-fp8-prefill/serve_9b_final_v1/analysis.json)
- [Full GSM8K comparison](artifacts/sm89-fp8-prefill/offline_gsm8k_9b/COMPLETE.json)
- [Redaction manifest](artifacts/sm89-fp8-prefill/redaction_manifest.json)

Source baseline: `88d3995ea18de47ba5c306789aa615474ad74592`.
Candidate: `115a32c53a6a66c8d8e407c3f8858b2290a3b071`.
AI assistance: Codex (OpenAI). Human upstream review remains required.

The initial serving screen and baseline diagnosis are labeled separately from the final crossover experiment. Profiled kernel durations are not used as unprofiled performance measurements. Model weights and compiled libraries are not included. Earlier 0.8B experiments do not support this PR's 9B claims and are excluded from this evidence branch.
