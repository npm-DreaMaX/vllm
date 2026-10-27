# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in compressed-only BF16 storage for measured Qwen MLP projections."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from vllm import envs
from vllm.config import get_current_vllm_config_or_none
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.kernels.linear.unquantized.packed_bf16_lm_head import (
    _PACK_BLOCK,
    _lossless_packed_bf16_lm_head_kernel,
    _packed_bf16_lm_head_impl,
    choose_launch_config,
    load_packed_bf16_pairs,
    load_packed_bf16_tile,
    pack_bf16_weight,
)
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.warmup.jit_warmup import VllmJitKernel
from vllm.model_executor.warmup.jit_warmup_triton_helper import TritonWarmupTensor
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

if TYPE_CHECKING:
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod

logger = init_logger(__name__)
# Every integer M in 1..32 was checked; this is not a device-wide batch threshold.
_MEASURED_BATCHES = {
    shape: tuple(range(1, 33))
    for shape in [(19456, 2560), (2560, 9728), (24576, 4096), (4096, 12288)]
}
_PLANES = (
    "sign_mantissa",
    "exponent_nibbles",
    "base_exponent",
    "fallback_slot",
    "fallback_bits",
)
_PREFIX = "_packed_mlp_"


@triton.jit
def _packed_mlp_gemv(
    X,
    SM,
    E,
    B,
    S,
    F,
    Y,
    M: tl.constexpr,
    MP: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BN: tl.constexpr,
    BP: tl.constexpr,
):
    n = tl.program_id(0) * BN + tl.arange(0, BN)
    m = tl.arange(0, MP)
    kk = tl.arange(0, BP)
    acc = tl.full((MP, BN, BP), 0, tl.float32)
    for step in range(tl.cdiv(K, 2 * BP)):
        p = step * BP + kk
        bits = load_packed_bf16_pairs(
            SM,
            E,
            B,
            S,
            F,
            n[:, None] * (K // 2) + p[None, :],
            (n[:, None] < N) & (p[None, :] < K // 2),
        )
        low = ((bits & 65535) << 16).to(tl.float32, bitcast=True)
        high = (bits & 0xFFFF0000).to(tl.float32, bitcast=True)
        x0 = tl.load(
            X + m[:, None] * K + 2 * p[None, :],
            (m[:, None] < M) & (2 * p[None, :] < K),
            0,
        ).to(tl.float32)
        x1 = tl.load(
            X + m[:, None] * K + 2 * p[None, :] + 1,
            (m[:, None] < M) & (2 * p[None, :] + 1 < K),
            0,
        ).to(tl.float32)
        acc = tl.fma(x0[:, None, :], low[None, :, :], acc)
        acc = tl.fma(x1[:, None, :], high[None, :, :], acc)
    tl.store(
        Y + m[:, None] * N + n[None, :],
        tl.sum(acc, 2),
        (m[:, None] < M) & (n[None, :] < N),
    )


@triton.jit
def _packed_mlp_gemm(
    X,
    SM,
    E,
    B,
    S,
    F,
    Y,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    SK: tl.constexpr,
):
    # Interleave M tiles within an N group to reuse decoded-weight source in L2.
    pid = tl.program_id(0)
    blocks_m = tl.cdiv(M, BM)
    pid_n = pid // blocks_m
    pid_m = pid % blocks_m
    split = tl.program_id(1)
    mi = pid_m * BM + tl.arange(0, BM)
    ni = pid_n * BN + tl.arange(0, BN)
    ki = tl.arange(0, BK)
    acc = tl.full((BM, BN), 0, tl.float32)
    count = tl.cdiv(K, BK * SK)
    for step in range(count):
        kk = (step * SK + split) * BK + ki
        w = load_packed_bf16_tile(SM, E, B, S, F, ni, kk, N, K, BK)
        x = tl.load(
            X + mi[:, None] * K + kk[None, :], (mi[:, None] < M) & (kk[None, :] < K), 0
        )
        acc = tl.dot(x, tl.trans(w), acc)
    offset = mi[:, None] * N + ni[None, :]
    if SK > 1:
        offset += split * M * N
    tl.store(Y + offset, acc, (mi[:, None] < M) & (ni[None, :] < N))


@triton.jit
def _reduce_split_k(P, Y, MN, SK: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    s = tl.arange(0, triton.next_power_of_2(SK))
    value = tl.load(
        P + s[:, None] * MN + i[None, :], (s[:, None] < SK) & (i[None, :] < MN), 0
    )
    tl.store(Y + i, tl.sum(value, 0), i < MN)


@triton.jit
def _unpack_kernel(X, SM, E, B, S, F, W, SIZE: tl.constexpr, BLOCK: tl.constexpr):
    pair = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    bits = load_packed_bf16_pairs(SM, E, B, S, F, pair, pair < SIZE // 2)
    tl.store(W.to(tl.pointer_type(tl.uint32)) + pair, bits, pair < SIZE // 2)


def _vector_config(n: int, k: int, m: int) -> tuple[int, int]:
    return (2 * _PACK_BLOCK if n >= k or m == 1 else _PACK_BLOCK) // 2, 4


class _MLPKernel(VllmJitKernel["_MLPKernel.CompileKey"]):
    @dataclass(frozen=True, slots=True)
    class CompileKey:
        n: int
        k: int
        m: int
        split_k: int

    def dispatch(self, **kwargs: Any) -> CompileKey:
        return self.CompileKey(kwargs["n"], kwargs["k"], kwargs["m"], kwargs["split_k"])

    def get_warmup_keys(self, **kwargs) -> list[CompileKey]:
        return [self.dispatch(**kwargs)]

    def compile(self, compile_key: CompileKey) -> None:
        n, k, m, sk = compile_key.n, compile_key.k, compile_key.m, compile_key.split_k
        x = TritonWarmupTensor(torch.bfloat16, shape=(m, k))
        planes = [
            TritonWarmupTensor(dtype)
            for dtype in (
                torch.uint8,
                torch.uint8,
                torch.uint8,
                torch.int32,
                torch.int16,
            )
        ]
        y = TritonWarmupTensor(torch.bfloat16, shape=(m, n))
        if m == 1 and n >= k:
            choice = choose_launch_config(k)
            _lossless_packed_bf16_lm_head_kernel.warmup(
                x,
                *planes,
                y,
                N=n,
                K=k,
                BLOCK_N=choice.block_n,
                BLOCK_K=triton.next_power_of_2(k),
                PACK_BLOCK=_PACK_BLOCK,
                num_warps=choice.num_warps,
                num_stages=1,
                grid=(1,),
            )
        elif m <= 4:
            bp, warps = _vector_config(n, k, m)
            _packed_mlp_gemv.warmup(
                x,
                *planes,
                y,
                m,
                triton.next_power_of_2(m),
                n,
                k,
                2,
                bp,
                num_warps=warps,
                grid=(1,),
            )
        else:
            partial = TritonWarmupTensor(torch.float32) if sk > 1 else y
            _packed_mlp_gemm.warmup(
                x,
                *planes,
                partial,
                m,
                n,
                k,
                max(16, triton.next_power_of_2(m)),
                16,
                _PACK_BLOCK,
                sk,
                num_warps=4,
                num_stages=1,
                grid=(1, 1),
            )
            if sk > 1:
                _reduce_split_k.warmup(
                    partial, y, m * n, sk, 512, num_warps=4, grid=(1,)
                )


_KERNEL = _MLPKernel()


def _project(
    x: torch.Tensor,
    sm: torch.Tensor,
    e: torch.Tensor,
    b: torch.Tensor,
    s: torch.Tensor,
    f: torch.Tensor,
    n: int,
    k: int,
    split_k: int = 1,
) -> torch.Tensor:
    m = x.numel() // k
    if (
        m not in _MEASURED_BATCHES.get((n, k), ())
        or x.shape[-1] != k
        or x.dtype != torch.bfloat16
        or not x.is_cuda
        or not x.is_contiguous()
    ):
        raise ValueError("Unsupported input reached the packed MLP fast branch")
    planes = (sm, e, b, s, f)
    if m == 1 and n >= k:
        config = choose_launch_config(k)
        return _packed_bf16_lm_head_impl(
            x, *planes, n, k, config.block_n, config.num_warps
        )
    y = x.new_empty((*x.shape[:-1], n))
    if m <= 4:
        bp, warps = _vector_config(n, k, m)
        _packed_mlp_gemv[(triton.cdiv(n, 2),)](
            x, *planes, y, m, triton.next_power_of_2(m), n, k, 2, bp, num_warps=warps
        )
    else:
        partial = (
            torch.empty((split_k, m, n), device=x.device, dtype=torch.float32)
            if split_k > 1
            else y
        )
        _packed_mlp_gemm[(triton.cdiv(n, 16), split_k)](
            x,
            *planes,
            partial,
            m,
            n,
            k,
            max(16, triton.next_power_of_2(m)),
            16,
            _PACK_BLOCK,
            split_k,
            num_warps=4,
            num_stages=1,
        )
        if split_k > 1:
            _reduce_split_k[(triton.cdiv(m * n, 512),)](
                partial, y, m * n, split_k, 512, num_warps=4
            )
    return y


def _dispatch(
    x: torch.Tensor,
    sm: torch.Tensor,
    e: torch.Tensor,
    b: torch.Tensor,
    s: torch.Tensor,
    f: torch.Tensor,
    n: int,
    k: int,
    split_k: int = 1,
    layer_name: str = "",
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    if (
        x.numel() // k in _MEASURED_BATCHES.get((n, k), ())
        and x.shape[-1] == k
        and x.dtype == torch.bfloat16
        and x.is_contiguous()
        and bias is None
    ):
        return _project(x, sm, e, b, s, f, n, k, split_k)
    layer = get_forward_context().no_compile_layers[layer_name]
    weight = _unpack(x, sm, e, b, s, f, n, k)
    return layer.quant_method.fallback.apply(layer, x, bias, weight=weight)


def _fake(x, sm, e, b, s, f, n, k, split_k=1, layer_name="", bias=None):
    return x.new_empty((*x.shape[:-1], n))


direct_register_custom_op(
    "packed_bf16_mlp", _dispatch, mutates_args=[], fake_impl=_fake
)


def _unpack(
    x: torch.Tensor,
    sm: torch.Tensor,
    e: torch.Tensor,
    b: torch.Tensor,
    s: torch.Tensor,
    f: torch.Tensor,
    n: int,
    k: int,
) -> torch.Tensor:
    weight = torch.empty((n, k), device=sm.device, dtype=torch.bfloat16)
    # The native branch needs one temporary BF16 projection, never a model cache.
    _unpack_kernel[(triton.cdiv(n * k // 2, 512),)](
        x, sm, e, b, s, f, weight, n * k, 512, num_warps=4
    )
    return weight


def _unpack_fake(x, sm, e, b, s, f, n, k):
    return torch.empty((n, k), device=sm.device, dtype=torch.bfloat16)


direct_register_custom_op(
    "unpack_bf16_mlp", _unpack, mutates_args=[], fake_impl=_unpack_fake
)


@torch.no_grad()
def _export_weight(layer, state_dict, prefix, local_metadata):
    if layer.weight is None:
        n, k = layer._packed_mlp_shape
        planes = [getattr(layer, _PREFIX + name) for name in _PLANES]
        sm, e, b, s, f = planes
        dummy = torch.empty((1,), device=planes[0].device, dtype=torch.bfloat16)
        state_dict[prefix + "weight"] = _unpack(dummy, sm, e, b, s, f, n, k).cpu()


class LosslessPackedMLPMethod(QuantizeMethodBase):
    """Store each weight once and decorate the original unquantized GEMM method."""

    supports_pre_processed_weights = False

    def __init__(self, fallback: "UnquantizedLinearMethod"):
        self.fallback = fallback
        config = get_current_vllm_config_or_none()
        assert config is not None
        self.kernel_config = config.kernel_config

    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "fallback"), name)

    def create_weights(self, layer, *args, **kwargs):
        return self.fallback.create_weights(layer, *args, **kwargs)

    def process_weights_after_loading(self, layer):
        from vllm.model_executor.model_loader.reload.layerwise import get_layerwise_info

        info = get_layerwise_info(layer)
        original = info.kernel_tensors[1] if info.kernel_tensors else layer._buffers
        was_packed = _PREFIX + "sign_mantissa" in original
        self.fallback.process_weights_after_loading(layer)
        weight = layer.weight
        self.kernel_config.record_mlp_weight_layout(
            layer.prefix,
            ("native", str(weight.dtype), tuple(weight.shape), tuple(weight.stride())),
        )
        if info.kernel_tensors is not None and not was_packed:
            return
        weight = layer.weight
        eligible = (
            weight.is_cuda
            and weight.dtype == torch.bfloat16
            and weight.is_contiguous()
            and tuple(weight.shape) in _MEASURED_BATCHES
            and weight.shape[1] % _PACK_BLOCK == 0
        )
        if not eligible:
            if was_packed:
                raise ValueError(
                    "Refit changes packed MLP eligibility; recreate the engine"
                )
            return
        packed = pack_bf16_weight(
            weight, max_packed_fraction=0.90, min_dense_bytes=32 * 1024**2, max_k=16384
        )
        if packed is None:
            if was_packed:
                raise ValueError(
                    "Refit cannot use the packed MLP layout; recreate the engine"
                )
            return
        plan, planes = packed
        previous = original.get(_PREFIX + "fallback_bits")
        capacity = (
            previous.shape[0]
            if previous is not None
            else 1
            if plan.fallback_blocks == 0
            else triton.cdiv(plan.fallback_blocks + 1, 1024) * 1024
        )
        if plan.fallback_blocks > capacity:
            raise ValueError(
                "Refit exceeds packed MLP exception capacity; recreate the engine"
            )
        stored_bytes = (
            plan.packed_bytes + (capacity - plan.fallback_blocks) * _PACK_BLOCK * 2
        )
        if previous is None and stored_bytes > 0.90 * plan.dense_bytes:
            return
        fallback = (
            previous
            if previous is not None
            else torch.empty(
                (capacity, _PACK_BLOCK), device=weight.device, dtype=torch.int16
            )
        )
        fallback[: plan.fallback_blocks].copy_(planes[-1])
        for name, tensor in zip(_PLANES, (*planes[:-1], fallback)):
            key = _PREFIX + name
            target = original.get(key)
            if target is not None:
                if target.shape != tensor.shape:
                    raise ValueError(
                        "Refit changes packed MLP storage; recreate the engine"
                    )
                if target.data_ptr() != tensor.data_ptr():
                    target.copy_(tensor)
                tensor = target
            if key in layer._buffers:
                setattr(layer, key, tensor)
            else:
                layer.register_buffer(key, tensor, persistent=False)
        layer._packed_mlp_shape = (plan.n, plan.k)
        units = current_platform.num_compute_units(weight.device.index or 0)
        split_k = min(
            8, triton.next_power_of_2(triton.cdiv(8 * units, triton.cdiv(plan.n, 16)))
        )
        layer._packed_mlp_split_k = split_k
        layer.register_parameter("weight", None)
        self.kernel_config.record_mlp_weight_layout(
            layer.prefix,
            (
                "packed_v1",
                plan.n,
                plan.k,
                split_k,
                tuple(
                    (
                        name,
                        str(getattr(layer, _PREFIX + name).dtype),
                        tuple(getattr(layer, _PREFIX + name).shape),
                    )
                    for name in _PLANES
                ),
            ),
        )
        if not hasattr(layer, "_packed_mlp_export_hook"):
            layer._packed_mlp_export_hook = layer.register_state_dict_post_hook(
                _export_weight
            )
        for m in _MEASURED_BATCHES[(plan.n, plan.k)]:
            _KERNEL.register_warmup(n=plan.n, k=plan.k, m=m, split_k=split_k)
        logger.debug(
            "Packed MLP %s stores %d bytes instead of %d",
            layer.prefix,
            stored_bytes,
            plan.dense_bytes,
        )

    def apply(self, layer, x, bias=None):
        shape = getattr(layer, "_packed_mlp_shape", None)
        if shape is None:
            return self.fallback.apply(layer, x, bias)
        n, k = shape
        planes = [getattr(layer, _PREFIX + name) for name in _PLANES]
        if x.device != planes[0].device:
            raise RuntimeError("Packed MLP input and weights must share a CUDA device")

        def fast(x, sm, e, b, s, f):
            return torch.ops.vllm.packed_bf16_mlp(
                x, sm, e, b, s, f, n, k, layer._packed_mlp_split_k
            )

        def native(x, sm, e, b, s, f):
            weight = torch.ops.vllm.unpack_bf16_mlp(x, sm, e, b, s, f, n, k)
            return self.fallback.apply(layer, x, bias, weight=weight)

        if (
            x.shape[-1] != k
            or not x.is_contiguous()
            or x.dtype != torch.bfloat16
            or not x.is_cuda
            or bias is not None
        ):
            return native(x, *planes)
        if not torch.compiler.is_compiling():
            return (
                fast(x, *planes)
                if x.numel() // k in _MEASURED_BATCHES[(n, k)]
                else native(x, *planes)
            )
        # Keep runtime dispatch inside the op: shape-dependent branches must
        # not become lifted predicates between piecewise compiled segments.
        return torch.ops.vllm.packed_bf16_mlp(
            x, *planes, n, k, layer._packed_mlp_split_k, layer.prefix, bias
        )


def maybe_enable_packed_mlp(gate_up, down):
    config = get_current_vllm_config_or_none()
    if (
        config is None
        or config.kernel_config.mlp_backend != "lossless_packed"
        or not current_platform.is_cuda()
        or envs.VLLM_BATCH_INVARIANT
        or not current_platform.is_device_capability(89)
        or current_platform.num_compute_units(torch.accelerator.current_device_index())
        != 128
        or config.lora_config is not None
        or config.parallel_config.tensor_parallel_size != 1
        or config.parallel_config.pipeline_parallel_size != 1
        or config.offload_config.uva.cpu_offload_gb > 0
        or config.offload_config.prefetch.offload_group_size > 0
    ):
        return
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod

    for layer in (gate_up, down):
        if (
            type(layer.quant_method) is UnquantizedLinearMethod
            and tuple(layer.weight.shape) in _MEASURED_BATCHES
        ):
            if layer.prefix in config.compilation_config.static_forward_context:
                raise ValueError(f"Duplicate packed MLP layer name: {layer.prefix}")
            layer.quant_method = LosslessPackedMLPMethod(layer.quant_method)
            config.compilation_config.static_forward_context[layer.prefix] = layer
