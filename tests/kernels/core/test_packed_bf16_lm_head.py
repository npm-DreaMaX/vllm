# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.config.kernel import KernelConfig
from vllm.model_executor.kernels.linear.unquantized import packed_bf16_lm_head
from vllm.model_executor.kernels.linear.unquantized.packed_bf16_lm_head import (
    LosslessPackedLMHeadMethod,
    _packed_bf16_lm_head_impl,
    choose_launch_config,
    pack_bf16_weight,
)
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

requires_cuda = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="the packed lm-head kernel is CUDA-only"
)


def test_lm_head_backend_config_normalizes_and_validates():
    """KernelConfig accepts CLI spelling and rejects unsafe storage limits."""
    config = KernelConfig(
        lm_head_backend=cast(Any, "lossless-packed"),
        mlp_backend=cast(Any, "lossless-packed"),
        linear_backend_per_quant={"nvfp4_w4a16": "humming"},
    )

    assert config.lm_head_backend == "lossless_packed"
    assert config.mlp_backend == "lossless_packed"
    assert config.linear_backend_per_quant == {"nvfp4_w4a16": "humming"}
    native = KernelConfig()
    assert (
        native.compute_hash()
        != KernelConfig(mlp_backend="lossless_packed").compute_hash()
    )
    with pytest.raises(ValueError):
        KernelConfig(lm_head_max_packed_fraction=0.0)


def _unpack_bits(tensors: tuple[torch.Tensor, ...], numel: int) -> torch.Tensor:
    sign_mantissa, exponent_nibbles, base_exponent, fallback_slot, fallback = tensors
    linear = torch.arange(sign_mantissa.numel(), device=sign_mantissa.device)
    block = linear // 256
    pair = exponent_nibbles[linear // 2].to(torch.int32)
    delta = (pair >> ((linear & 1) * 4)) & 0xF
    sm = sign_mantissa.to(torch.int32)
    bits = (
        ((sm & 0x80) << 8)
        | ((base_exponent[block].to(torch.int32) + delta) << 7)
        | (sm & 0x7F)
    )
    slots = fallback_slot[block].to(torch.int64)
    fallback_values = (
        fallback.flatten()[slots.clamp_min(0) * 256 + linear.remainder(256)].to(
            torch.int32
        )
        & 0xFFFF
    )
    return torch.where(slots >= 0, fallback_values, bits)[:numel].to(torch.int32)


@requires_cuda
@torch.inference_mode()
def test_packing_reconstructs_every_bf16_bit(monkeypatch):
    """Packed and fallback blocks together preserve every original BF16 bit."""
    monkeypatch.setattr(packed_bf16_lm_head, "_MIN_DENSE_BYTES", 0)
    weight = (torch.rand(521, 1023, device="cuda") + 0.5).to(torch.bfloat16)
    weight.view(-1)[:256] = torch.linspace(
        2**-20, 2**20, 256, dtype=torch.bfloat16, device="cuda"
    )

    packed = pack_bf16_weight(weight, max_packed_fraction=1.0)

    assert packed is not None
    plan, tensors = packed
    expected = weight.view(torch.int16).flatten().to(torch.int32) & 0xFFFF
    torch.testing.assert_close(
        _unpack_bits(tensors, plan.numel), expected, atol=0, rtol=0
    )
    assert plan.fallback_blocks > 0


@requires_cuda
@torch.inference_mode()
@pytest.mark.parametrize("n,k", [(33001, 1023), (8192, 2048)])
def test_single_token_projection_matches_torch(monkeypatch, n, k):
    """The fast path matches an unquantized projection at BF16 precision."""
    monkeypatch.setattr(packed_bf16_lm_head, "_MIN_DENSE_BYTES", 0)
    torch.manual_seed(7)
    magnitude = torch.rand(n, k, device="cuda") + 0.5
    signs = torch.where(
        torch.rand(n, k, device="cuda") > 0.5,
        torch.ones((), device="cuda"),
        -torch.ones((), device="cuda"),
    )
    weight = (magnitude * signs).to(torch.bfloat16)
    weight.view(-1)[:256] = torch.linspace(
        2**-20, 2**20, 256, dtype=torch.bfloat16, device="cuda"
    )
    x = torch.randn(1, k, dtype=torch.bfloat16, device="cuda")
    packed = pack_bf16_weight(weight, max_packed_fraction=1.0)

    assert packed is not None
    plan, tensors = packed
    sign_mantissa, exponent_nibbles, base_exponent, fallback_slot, fallback = tensors
    launch = choose_launch_config(k)
    actual = _packed_bf16_lm_head_impl(
        x,
        sign_mantissa,
        exponent_nibbles,
        base_exponent,
        fallback_slot,
        fallback,
        plan.n,
        plan.k,
        launch.block_n,
        launch.num_warps,
    )
    expected = torch.nn.functional.linear(x, weight)
    torch.testing.assert_close(actual, expected, atol=0.125, rtol=0.02)


class _FallbackMethod:
    interface_marker = "wrapped interface"

    def apply(self, layer, x, bias=None):
        del layer, bias
        return x + 1


def test_method_falls_back_for_non_cuda_input():
    """Unsupported runtime shapes retain the wrapped method's behavior."""
    method = LosslessPackedLMHeadMethod(_FallbackMethod())  # type: ignore[arg-type]
    layer = SimpleNamespace(_vllm_lossless_lm_head_meta=(8, 4, 2, 1))
    x = torch.zeros(2, 4)

    torch.testing.assert_close(method.apply(layer, x), x + 1)


def test_method_preserves_wrapped_interface():
    """The decorator exposes attributes implemented by its fallback."""
    method = LosslessPackedLMHeadMethod(_FallbackMethod())  # type: ignore[arg-type]

    assert method.interface_marker == "wrapped interface"


def test_batch_invariant_mode_skips_packed_state(monkeypatch):
    """Batch-invariant execution must always retain the stock linear path."""

    class _PackedStateMustNotBeRead:
        @property
        def _vllm_lossless_lm_head_meta(self):
            raise AssertionError("packed state was inspected")

    monkeypatch.setattr(packed_bf16_lm_head.envs, "VLLM_BATCH_INVARIANT", True)

    assert (
        packed_bf16_lm_head._try_apply_packed_weight(
            _PackedStateMustNotBeRead(), torch.zeros(1, 4), None
        )
        is None
    )


def test_parallel_lm_head_default_is_not_wrapped():
    """The default configuration does not alter the lm-head method."""
    config = VllmConfig(kernel_config=KernelConfig(lm_head_backend="torch"))
    with set_current_vllm_config(config):
        head = ParallelLMHead(64, 16, disable_tp=True)

    assert not isinstance(head.quant_method, LosslessPackedLMHeadMethod)


@requires_cuda
def test_parallel_lm_head_wraps_eligible_method_when_requested():
    """An eligible CUDA BF16 head receives the requested method decorator."""
    config = VllmConfig(kernel_config=KernelConfig(lm_head_backend="lossless_packed"))
    with set_current_vllm_config(config):
        head = ParallelLMHead(64, 16, params_dtype=torch.bfloat16, disable_tp=True)

    assert isinstance(head.quant_method, LosslessPackedLMHeadMethod)


@requires_cuda
@torch.inference_mode()
@pytest.mark.parametrize("n,k", [(19456, 2560), (2560, 9728)])
def test_mlp_projection_limits_fast_path_and_keeps_native_fallback(n, k):
    """Measured batches use GEMV; M7/M16 and prefill run native GEMM exactly."""
    from vllm.model_executor.kernels.linear.unquantized.packed_bf16_mlp import (
        _project,
    )

    torch.manual_seed(17)
    weight = (torch.randn(n, k, device="cuda") / k**0.5).to(torch.bfloat16)
    packed = pack_bf16_weight(
        weight,
        max_packed_fraction=0.9,
        min_dense_bytes=32 * 1024**2,
        max_k=9728,
        fallback_from_weight=True,
    )
    assert packed is not None
    _, tensors = packed
    sm, exponent, base, slots = tensors[:4]
    assert tensors[-1].numel() == 0
    inputs = {
        m: torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        for m in (1, 2, 3, 4, 7, 16, 128)
    }
    reference = {m: torch.nn.functional.linear(x, weight) for m, x in inputs.items()}
    for x in inputs.values():
        _project(x, weight, *tensors[:4])
    torch.accelerator.synchronize()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as profiler:
        actual = {m: _project(x, weight, *tensors[:4]) for m, x in inputs.items()}
        torch.accelerator.synchronize()
    fast_batches = (1, 2, 3, 4) if n == 19456 else (1, 2, 4)
    assert sum(event.name == "_packed_mlp_gemv" for event in profiler.events()) == len(
        fast_batches
    )
    for m in inputs:
        if m in fast_batches:
            torch.testing.assert_close(actual[m], reference[m], atol=0.02, rtol=0.01)
        else:
            torch.testing.assert_close(actual[m], reference[m], atol=0, rtol=0)
    x = torch.randn(4, k * 2, device="cuda", dtype=torch.bfloat16)[:, ::2]
    bias = torch.randn(n, device="cuda", dtype=torch.bfloat16)
    for value in (None, bias):
        torch.testing.assert_close(
            _project(x, weight, sm, exponent, base, slots, value),
            torch.nn.functional.linear(x, weight, value),
            atol=0,
            rtol=0,
        )
    torch.library.opcheck(
        torch.ops.vllm.packed_bf16_mlp.default,
        (inputs[4], weight, sm, exponent, base, slots),
    )


@triton.jit
def _decode_borrowed_weight(sm, exponent, base, slots, weight, output):
    offsets = tl.program_id(0) * 256 + tl.arange(0, 256)
    values = packed_bf16_lm_head.load_packed_bf16(
        sm, exponent, base, slots, weight, offsets, True, 256, True
    )
    bits = values.to(tl.int32, bitcast=True) >> 16
    tl.store(output + offsets, bits.to(tl.int16))


@requires_cuda
@torch.inference_mode()
@pytest.mark.parametrize("interleaved", [False, True])
def test_borrowed_weight_decode_preserves_all_bf16_bit_patterns(interleaved):
    """Both packed and borrowed blocks preserve zeros, subnormals and NaN bits."""
    bits = torch.arange(65536, device="cuda", dtype=torch.int32).to(torch.int16)
    bits = bits.reshape(256, 256)
    if interleaved:
        bits = bits.t().contiguous()
    weight = bits.view(torch.bfloat16)
    packed = pack_bf16_weight(
        weight, max_packed_fraction=0.9, min_dense_bytes=0, fallback_from_weight=True
    )
    assert packed is not None
    plan, tensors = packed
    assert plan.fallback_blocks == (256 if interleaved else 0)
    assert tensors[-1].numel() == 0
    decoded = torch.empty_like(bits)
    _decode_borrowed_weight[(256,)](*tensors[:4], weight, decoded)
    torch.testing.assert_close(decoded, bits, atol=0, rtol=0)


@requires_cuda
@pytest.mark.usefixtures("dist_init")
@torch.no_grad()
def test_packed_mlp_checkpoint_refit_refreshes_captured_buffers(tmp_path):
    """A real refit changes encoded values without invalidating a captured graph."""
    from transformers import Qwen3Config

    from vllm.config import ModelConfig
    from vllm.model_executor.layers.linear import ReplicatedLinear
    from vllm.model_executor.model_loader.reload.inplace import reload_weights
    from vllm.model_executor.model_loader.reload.layerwise import (
        record_metadata_for_reloading,
    )
    from vllm.model_executor.models.utils import AutoWeightsLoader

    class ProjectionModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.projection = ReplicatedLinear(
                2560,
                19456,
                bias=False,
                params_dtype=torch.bfloat16,
                return_bias=False,
            )

        def forward(self, x):
            return self.projection(x)

        def load_weights(self, weights):
            return AutoWeightsLoader(self).load_weights(weights)

    Qwen3Config(architectures=["Qwen3ForCausalLM"]).save_pretrained(tmp_path)
    config = VllmConfig(
        model_config=ModelConfig(
            model=str(tmp_path), skip_tokenizer_init=True, dtype="bfloat16"
        ),
        kernel_config=KernelConfig(mlp_backend="lossless_packed"),
    )
    with set_current_vllm_config(config), torch.device("cuda"):
        model = ProjectionModel()
        weight = torch.randn_like(model.projection.weight) / 2560**0.5
        model.load_weights([("projection.weight", weight)])
        record_metadata_for_reloading(model)
        model.projection.quant_method.process_weights_after_loading(model.projection)
        x = torch.randn(4, 2560, dtype=torch.bfloat16)
        buffers = dict(model.projection.named_buffers())
        if torch.cuda.get_device_capability() != (8, 9):
            assert not buffers
            torch.testing.assert_close(model(x), torch.nn.functional.linear(x, weight))
            return
        assert len(buffers) == 4
        pointers = {name: tensor.data_ptr() for name, tensor in buffers.items()}
        assert set(model.state_dict()) == {"projection.weight"}
        for _ in range(3):
            model(x)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = model(x)
        # Every pack block now needs the original-weight fallback. Its size must
        # not depend on the number of outliers or change captured addresses.
        new_weight = torch.zeros_like(weight)
        new_weight[:, ::2] = 0.125
        reload_weights(config, model, [("projection.weight", new_weight)], None, True)
        assert pointers == {
            name: tensor.data_ptr() for name, tensor in model.projection.named_buffers()
        }
        graph.replay()
        torch.testing.assert_close(
            output, torch.nn.functional.linear(x, new_weight), atol=0.02, rtol=0.01
        )
        with pytest.raises(ValueError, match="checkpoint-format"):
            reload_weights(config, model, [("projection.weight", weight)], None, False)
        torch.testing.assert_close(model.projection.weight, new_weight, atol=0, rtol=0)
