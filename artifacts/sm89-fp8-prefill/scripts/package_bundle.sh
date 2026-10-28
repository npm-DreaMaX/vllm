#!/usr/bin/env bash
set -euo pipefail
R=/workspace/research/value_reboot_20261027
S="$R/vllm_fp8"
E="$R/prefill_pr_v1/evidence_git"
B="$R/prefill_pr_v1/vllm-sm89-fp8-prefill.bundle"
SOURCE=codex/sm89-fp8-prefill-20261028
EVIDENCE=evidence/sm89-fp8-prefill-20261028
BASE=88d3995ea18de47ba5c306789aa615474ad74592
CANDIDATE=115a32c53a6a66c8d8e407c3f8858b2290a3b071
[[ $(git -C "$S" rev-parse "$SOURCE") == "$CANDIDATE" ]]
[[ -z $(git -C "$S" status --porcelain) ]]
test ! -e "$E"
git init -q "$E"
git -C "$E" symbolic-ref HEAD "refs/heads/$EVIDENCE"
git -C "$E" config user.name Codex
git -C "$E" config user.email noreply@openai.com
cp -a "$R/prefill_pr_v1/public_evidence/." "$E/"
git -C "$E" add .
git -C "$E" commit -q -m 'Archive native Qwen3.5-9B FP8 prefill validation' -m 'Generated-by: Codex (OpenAI)'
git -C "$S" fetch --quiet "$E" "HEAD:refs/heads/$EVIDENCE"
git -C "$S" bundle create "$B" "$SOURCE" "$EVIDENCE" "^$BASE"
git -C "$S" bundle verify "$B"
git -C "$S" bundle list-heads "$B"
git -C "$S" diff --check "$BASE" "$CANDIDATE"
git -C "$S" diff --stat "$BASE" "$CANDIDATE"
sha256sum "$B"
