# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in small-batch MLP projection using the shared lossless BF16 encoding."""

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from vllm import envs
from vllm.logger import init_logger
from vllm.model_executor.kernels.linear.unquantized.packed_bf16_lm_head import (
    _PACK_BLOCK,
    load_packed_bf16,
    pack_bf16_weight,
)
from vllm.model_executor.warmup.jit_warmup import VllmJitKernel
from vllm.model_executor.warmup.jit_warmup_triton_helper import TritonWarmupTensor
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)
_BATCHES = {(19456, 2560): (1, 2, 3, 4), (2560, 9728): (1, 2, 4)}
_PLANES = ("sign_mantissa", "exponent_nibbles", "base_exponent", "fallback_slot")
_PREFIX = "_packed_mlp_"


@triton.jit
def _packed_mlp_gemv(
    X,
    W,
    SM,
    E,
    B,
    S,
    Y,
    M: tl.constexpr,
    MP: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BK: tl.constexpr,
    PACK_BLOCK: tl.constexpr,
):
    n = tl.program_id(0) * 2 + tl.arange(0, 2)
    m = tl.arange(0, MP)
    kk = tl.arange(0, BK)
    acc = tl.full((MP, 2, BK), 0, tl.float32)
    for step in range(tl.cdiv(K, BK)):
        k = step * BK + kk
        offsets = n[:, None] * K + k[None, :]
        weight = load_packed_bf16(
            SM,
            E,
            B,
            S,
            W,
            offsets,
            (n[:, None] < N) & (k[None, :] < K),
            PACK_BLOCK,
            True,
        )
        x = tl.load(
            X + m[:, None] * K + k[None, :],
            (m[:, None] < M) & (k[None, :] < K),
            0,
        ).to(tl.float32)
        acc += x[:, None, :] * weight[None, :, :]
    tl.store(
        Y + m[:, None] * N + n[None, :],
        tl.sum(acc, 2),
        (m[:, None] < M) & (n[None, :] < N),
    )


def _launch_config(n: int, m: int) -> tuple[int, int]:
    return (512 if n == 19456 or m == 1 else 256), (8 if n == 19456 and m == 1 else 4)


class _MLPKernel(VllmJitKernel["_MLPKernel.CompileKey"]):
    @dataclass(frozen=True, slots=True)
    class CompileKey:
        n: int
        k: int
        m: int

    def dispatch(self, **kwargs: Any) -> CompileKey:
        return self.CompileKey(kwargs["n"], kwargs["k"], kwargs["m"])

    def get_warmup_keys(self, **kwargs) -> list[CompileKey]:
        return [self.dispatch(**kwargs)]

    def compile(self, compile_key: CompileKey) -> None:
        n, k, m = compile_key.n, compile_key.k, compile_key.m
        bk, warps = _launch_config(n, m)
        _packed_mlp_gemv.warmup(
            TritonWarmupTensor(torch.bfloat16, shape=(m, k)),
            TritonWarmupTensor(torch.bfloat16, shape=(n, k)),
            TritonWarmupTensor(torch.uint8),
            TritonWarmupTensor(torch.uint8),
            TritonWarmupTensor(torch.uint8),
            TritonWarmupTensor(torch.int32),
            TritonWarmupTensor(torch.bfloat16, shape=(m, n)),
            M=m,
            MP=triton.next_power_of_2(m),
            N=n,
            K=k,
            BK=bk,
            PACK_BLOCK=_PACK_BLOCK,
            num_warps=warps,
            grid=(1,),
        )


_KERNEL = _MLPKernel()


def _project(
    x: torch.Tensor,
    weight: torch.Tensor,
    sm: torch.Tensor,
    exponent: torch.Tensor,
    base: torch.Tensor,
    slots: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Use measured BF16 GEMV shapes; retain native GEMM for every other input."""
    n, k = weight.shape
    m = x.numel() // k
    if (
        m not in _BATCHES.get((n, k), ())
        or x.shape[-1] != k
        or x.dtype != torch.bfloat16
        or weight.dtype != torch.bfloat16
        or not x.is_cuda
        or x.device != weight.device
        or sm.device != x.device
        or not x.is_contiguous()
        or not weight.is_contiguous()
        or bias is not None
    ):
        return F.linear(x, weight, bias)
    result = x.new_empty((*x.shape[:-1], n))
    bk, warps = _launch_config(n, m)
    _packed_mlp_gemv[(cdiv(n, 2),)](
        x,
        weight,
        sm,
        exponent,
        base,
        slots,
        result,
        m,
        triton.next_power_of_2(m),
        n,
        k,
        bk,
        PACK_BLOCK=_PACK_BLOCK,
        num_warps=warps,
    )
    return result


def _fake(x, weight, sm, exponent, base, slots, bias=None):
    return x.new_empty((*x.shape[:-1], weight.shape[0]))


direct_register_custom_op(
    op_name="packed_bf16_mlp", op_func=_project, mutates_args=[], fake_impl=_fake
)


def prepare_packed_mlp(layer: torch.nn.Module) -> None:
    weight = layer.weight
    if (
        envs.VLLM_BATCH_INVARIANT
        or not current_platform.is_cuda()
        or not weight.is_cuda
        or weight.dtype != torch.bfloat16
        or not weight.is_contiguous()
        or tuple(weight.shape) not in _BATCHES
        or torch.cuda.get_device_capability(weight.device) != (8, 9)
    ):
        return
    packed = pack_bf16_weight(
        weight,
        max_packed_fraction=0.90,
        min_dense_bytes=32 * 1024**2,
        max_k=9728,
        fallback_from_weight=True,
    )
    if packed is None:
        if hasattr(layer, "_packed_mlp_shape"):
            raise RuntimeError("Cannot refresh packed MLP buffers during weight reload")
        return
    plan, tensors = packed
    # The loader restores original buffers after a refit. Refresh their storage
    # directly: non-persistent derived buffers are otherwise deliberately skipped.
    from vllm.model_executor.model_loader.reload.layerwise import get_layerwise_info

    info = get_layerwise_info(layer)
    original = info.kernel_tensors[1] if info.kernel_tensors else layer._buffers
    for name, tensor in zip(_PLANES, tensors):
        key = _PREFIX + name
        previous = original.get(key)
        if previous is not None:
            previous.copy_(tensor)
            tensor = previous
        if key in layer._buffers:
            setattr(layer, key, tensor)
        else:
            layer.register_buffer(key, tensor, persistent=False)
    layer._packed_mlp_shape = (plan.n, plan.k)
    for m in _BATCHES[(plan.n, plan.k)]:
        _KERNEL.register_warmup(n=plan.n, k=plan.k, m=m)
    logger.info_once(
        "Packed BF16 MLP: shape=(%d, %d), %.2f MiB auxiliary storage; "
        "native weights retained for fallback",
        plan.n,
        plan.k,
        plan.packed_bytes / 1024**2,
    )


def apply_packed_mlp(layer, x, bias=None):
    return torch.ops.vllm.packed_bf16_mlp(
        x, layer.weight, *(getattr(layer, _PREFIX + name) for name in _PLANES), bias
    )
