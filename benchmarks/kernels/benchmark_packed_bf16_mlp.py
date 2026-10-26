# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare packed MLP dispatch and native linear on checkpoint weight banks.

Example:
    python benchmarks/kernels/benchmark_packed_bf16_mlp.py \
        --model /path/to/Qwen3-4B-Instruct-2507 --output results.json

Multiple layers keep the weight working set larger than GPU L2. Inputs are
seeded synthetic BF16 activations; this is an operator benchmark, not serving.

"""

import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

from vllm.model_executor.kernels.linear.unquantized.packed_bf16_lm_head import (
    pack_bf16_weight,
)
from vllm.model_executor.kernels.linear.unquantized.packed_bf16_mlp import _project
from vllm.triton_utils import triton


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--layers", type=int, nargs="+", default=[0, 5, 10, 15, 20, 25, 30, 35]
    )
    parser.add_argument(
        "--batch-sizes", type=int, nargs="+", default=[1, 2, 3, 4, 7, 16, 128, 8192]
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rep-ms", type=int, default=100)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    weight_map = json.loads((args.model / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]

    def read(name):
        with safe_open(
            args.model / weight_map[name], framework="pt", device="cpu"
        ) as file:
            return file.get_tensor(name)

    result = {
        "command": vars(args) | {"model": str(args.model), "output": str(args.output)},
        "environment": {
            "torch": torch.__version__,
            "triton": triton.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(),
            "capability": torch.cuda.get_device_capability(),
            "python": platform.python_version(),
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "git_status": subprocess.check_output(
                ["git", "status", "--short"], text=True
            ),
            "kernel_source_sha256": {
                name: hashlib.sha256(
                    (
                        Path(__file__).resolve().parents[2]
                        / "vllm/model_executor/kernels/linear/unquantized"
                        / name
                    ).read_bytes()
                ).hexdigest()
                for name in ("packed_bf16_mlp.py", "packed_bf16_lm_head.py")
            },
        },
        "scope": (
            "Checkpoint weight bank, seeded synthetic inputs, CUDA Graph replay. "
            "No serving claim."
        ),
        "weights": [],
        "measurements": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for projection in ("gate_up", "down"):
        bank = []
        for layer in args.layers:
            prefix = f"model.layers.{layer}.mlp."
            weight = (
                read(prefix + "down_proj.weight")
                if projection == "down"
                else torch.cat(
                    [read(prefix + "gate_proj.weight"), read(prefix + "up_proj.weight")]
                )
            ).cuda()
            packed = pack_bf16_weight(
                weight,
                max_packed_fraction=0.9,
                min_dense_bytes=32 * 1024**2,
                max_k=9728,
                fallback_from_weight=True,
            )
            assert packed is not None
            plan, tensors = packed
            result["weights"].append(
                {
                    "projection": projection,
                    "layer": layer,
                    "N": plan.n,
                    "K": plan.k,
                    "native_bytes": plan.dense_bytes,
                    "auxiliary_bytes": plan.packed_bytes,
                    "fallback_blocks": plan.fallback_blocks,
                    "blocks": plan.block_count,
                }
            )
            bank.append((weight, tensors[:4]))
        for m in args.batch_sizes:
            inputs = [
                torch.randn(m, weight.shape[1], device="cuda", dtype=torch.bfloat16)
                for weight, _ in bank
            ]
            for x, (weight, tensors) in zip(inputs, bank):
                expected = F.linear(x, weight)
                actual = _project(x, weight, *tensors)
                if m in (1, 2, 4) or (projection == "gate_up" and m == 3):
                    torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.01)
                else:
                    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            for repeat in range(args.repeats):
                order = (
                    ("native", "packed") if repeat % 2 == 0 else ("packed", "native")
                )
                for implementation in order:

                    def call(inputs=inputs, bank=bank, implementation=implementation):
                        for x, (weight, tensors) in zip(inputs, bank):
                            if implementation == "native":
                                F.linear(x, weight)
                            else:
                                _project(x, weight, *tensors)

                    latency = triton.testing.do_bench_cudagraph(call, rep=args.rep_ms)
                    row = {
                        "projection": projection,
                        "M": m,
                        "repeat": repeat + 1,
                        "implementation": implementation,
                        "latency_us_per_layer": latency * 1000 / len(bank),
                    }
                    result["measurements"].append(row)
                    print(json.dumps(row), flush=True)
                    args.output.write_text(json.dumps(result, indent=2) + "\n")
        del bank, inputs, expected, actual, weight, tensors, packed
        torch.accelerator.empty_cache()


if __name__ == "__main__":
    main()
