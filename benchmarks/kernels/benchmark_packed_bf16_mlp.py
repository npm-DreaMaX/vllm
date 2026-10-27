# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark real BF16 projection weights without serving/profiler overhead."""

import argparse
import hashlib
import json
import statistics
import subprocess
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

from vllm.model_executor.kernels.linear.unquantized.packed_bf16_lm_head import (
    pack_bf16_weight,
)
from vllm.model_executor.kernels.linear.unquantized.packed_bf16_mlp import (
    _MEASURED_BATCHES,
    _project,
    _unpack,
)
from vllm.triton_utils import triton


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batches", default="1,2,3,4,7,8,16,17,24,32,33,128,8192")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--rep-ms", type=int, default=150)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    torch.set_num_threads(1)
    properties = torch.cuda.get_device_properties(
        torch.accelerator.current_device_index()
    )
    result = {
        "torch": torch.__version__,
        "triton": triton.__version__,
        "gpu": properties.name,
        "sm_count": properties.multi_processor_count,
        "reference": (
            "FP64 GEMM rounded to BF16 for fast kernels; "
            "exact weights for native fallback"
        ),
        "bf16_reduced_precision_reduction": (
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
        ),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "source_patch_sha256": hashlib.sha256(
            subprocess.check_output(["git", "diff", "--binary"])
        ).hexdigest(),
        "rows": [],
    }
    for folder in args.model_path:
        index = json.loads((folder / "model.safetensors.index.json").read_text())[
            "weight_map"
        ]
        config = json.loads((folder / "config.json").read_text())
        layers = [0, config["num_hidden_layers"] // 2, config["num_hidden_layers"] - 1]

        def load(name, folder=folder, index=index):
            with safe_open(folder / index[name], framework="pt", device="cpu") as file:
                return file.get_tensor(name)

        for role in ("gate_up", "down"):
            bank = []
            for layer in layers:
                prefix = f"model.layers.{layer}.mlp."
                weight = (
                    load(prefix + "down_proj.weight")
                    if role == "down"
                    else torch.cat(
                        [
                            load(prefix + "gate_proj.weight"),
                            load(prefix + "up_proj.weight"),
                        ]
                    )
                )
                weight = weight.cuda()
                packed = pack_bf16_weight(
                    weight, max_packed_fraction=0.90, min_dense_bytes=0, max_k=16384
                )
                if packed is None:
                    raise ValueError(
                        f"Weights cannot use packed storage: {folder}, {role}, {layer}"
                    )
                plan, planes = packed
                n, k = weight.shape
                dummy = weight.new_empty((1, k))
                decoded = _unpack(dummy, *planes, n, k)
                torch.testing.assert_close(
                    decoded.view(torch.int16), weight.view(torch.int16), atol=0, rtol=0
                )
                bank.append((weight, planes, plan))
            n, k = bank[0][0].shape
            split_k = min(
                8,
                triton.next_power_of_2(
                    triton.cdiv(
                        8 * properties.multi_processor_count, triton.cdiv(n, 16)
                    )
                ),
            )
            for m in map(int, args.batches.split(",")):
                inputs = [
                    torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
                    for _ in bank
                ]
                fast = m in _MEASURED_BATCHES.get((n, k), ())

                def native(inputs=inputs, bank=bank):
                    for x, (weight, _, _) in zip(inputs, bank):
                        F.linear(x, weight)

                def candidate_one(x, planes, fast=fast, n=n, k=k, split_k=split_k):
                    if fast:
                        return _project(x, *planes, n, k, split_k)
                    return F.linear(x, _unpack(x, *planes, n, k))

                def candidate(inputs=inputs, bank=bank, candidate_one=candidate_one):
                    for x, (_, planes, _) in zip(inputs, bank):
                        candidate_one(x, planes)

                actual = candidate_one(inputs[0], bank[0][1])
                if fast:
                    reference = F.linear(inputs[0].double(), bank[0][0].double()).to(
                        torch.bfloat16
                    )
                    torch.testing.assert_close(actual, reference, atol=1e-4, rtol=0.008)
                else:
                    torch.testing.assert_close(
                        actual, F.linear(inputs[0], bank[0][0]), atol=0, rtol=0
                    )
                samples = {"native_us": [], "candidate_us": []}
                for repeat in range(args.repeats):
                    functions = [("native_us", native), ("candidate_us", candidate)]
                    if repeat % 2:
                        functions.reverse()
                    for name, function in functions:
                        samples[name].append(
                            triton.testing.do_bench_cudagraph(function, rep=args.rep_ms)
                            * 1000
                            / len(bank)
                        )
                row = {
                    "model_path": str(folder),
                    "projection": role,
                    "N": n,
                    "K": k,
                    "M": m,
                    "layers": layers,
                    "path": "packed" if fast else "unpack_plus_native",
                    "dense_bytes": [plan.dense_bytes for _, _, plan in bank],
                    "encoded_bytes": [plan.packed_bytes for _, _, plan in bank],
                    "fallback_blocks": [plan.fallback_blocks for _, _, plan in bank],
                    **samples,
                }
                row["speedup"] = statistics.median(
                    samples["native_us"]
                ) / statistics.median(samples["candidate_us"])
                result["rows"].append(row)
                args.output.write_text(json.dumps(result, indent=2) + "\n")
                print(json.dumps(row), flush=True)
            del bank
            torch.accelerator.empty_cache()


if __name__ == "__main__":
    main()
