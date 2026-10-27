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


def test_packed_mlp_cache_key_distinguishes_exception_capacity():
    """A cached graph must never be reused with a different weight-buffer shape."""
    first = KernelConfig(mlp_backend="lossless_packed")
    second = KernelConfig(mlp_backend="lossless_packed")
    assert first.compute_hash() == second.compute_hash()
    first.record_mlp_weight_layout("mlp", ("packed_v1", (7168, 256)))
    second.record_mlp_weight_layout("mlp", ("packed_v1", (8192, 256)))
    assert first.compute_hash() != second.compute_hash()


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
@pytest.mark.parametrize(
    "n,k", [(19456, 2560), (2560, 9728), (24576, 4096), (4096, 12288)]
)
def test_compact_mlp_projection_matches_high_precision_gemm(n, k):
    """The shared/paired/TC paths round correctly without dense weight storage."""
    from vllm.model_executor.kernels.linear.unquantized.packed_bf16_mlp import (
        _project,
        _unpack,
    )

    torch.manual_seed(17)
    weight = (torch.randn(n, k, device="cuda") / k**0.5).to(torch.bfloat16)
    packed = pack_bf16_weight(
        weight, max_packed_fraction=0.9, min_dense_bytes=0, max_k=16384
    )
    assert packed is not None
    _, planes = packed
    sm, exponents, bases, slots, raw = planes
    x = torch.randn(1, k, device="cuda", dtype=torch.bfloat16)
    decoded = _unpack(x, sm, exponents, bases, slots, raw, n, k)
    torch.testing.assert_close(
        decoded.view(torch.int16), weight.view(torch.int16), atol=0, rtol=0
    )
    exact_weight = weight.double()
    split_k = 1 if n >= k else (8 if n == 2560 else 4)
    for m in range(1, 33):
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        reference = torch.nn.functional.linear(x.double(), exact_weight).to(
            torch.bfloat16
        )
        torch.testing.assert_close(
            _project(x, sm, exponents, bases, slots, raw, n, k, split_k),
            reference,
            atol=1e-4,
            rtol=0.008,
        )
    torch.library.opcheck(
        torch.ops.vllm.packed_bf16_mlp.default, (x, *planes, n, k, split_k)
    )
    torch.library.opcheck(torch.ops.vllm.unpack_bf16_mlp.default, (x, *planes, n, k))


@requires_cuda
@torch.inference_mode()
def test_compact_decode_preserves_every_bf16_pattern():
    """Packed and raw-exception blocks preserve zeros, subnormals and NaN bits."""
    from vllm.model_executor.kernels.linear.unquantized.packed_bf16_mlp import _unpack

    bits = torch.zeros(8 * 65536, device="cuda", dtype=torch.int16)
    bits[:65536] = (
        torch.arange(65536, device="cuda", dtype=torch.int32)
        .to(torch.int16)
        .reshape(256, 256)
        .t()
        .contiguous()
        .flatten()
    )
    weight = bits.reshape(-1, 256).view(torch.bfloat16)
    packed = pack_bf16_weight(weight, max_packed_fraction=0.9, min_dense_bytes=0)
    assert packed is not None
    plan, planes = packed
    assert plan.fallback_blocks == 256
    decoded = _unpack(weight[:1], *planes, *weight.shape)
    torch.testing.assert_close(
        decoded.view(torch.int16).flatten(), bits, atol=0, rtol=0
    )


@requires_cuda
@pytest.mark.usefixtures("dist_init")
@torch.no_grad()
def test_unrelated_linear_keeps_native_weights():
    """A matching matrix shape outside an MLP must not activate packed storage."""
    from vllm.model_executor.layers.linear import ReplicatedLinear

    config = VllmConfig(kernel_config=KernelConfig(mlp_backend="lossless_packed"))
    with set_current_vllm_config(config), torch.device("cuda"):
        layer = ReplicatedLinear(
            2560,
            19456,
            bias=False,
            params_dtype=torch.bfloat16,
            return_bias=False,
            disable_tp=True,
        )
        layer.weight.normal_(0, 0.02)
        layer.quant_method.process_weights_after_loading(layer)
        assert not list(layer.named_buffers())
        x = torch.randn(4, 2560, dtype=torch.bfloat16)
        torch.testing.assert_close(
            layer(x), torch.nn.functional.linear(x, layer.weight), atol=0, rtol=0
        )


@requires_cuda
@pytest.mark.usefixtures("dist_init")
@torch.no_grad()
def test_compact_mlp_refit_preserves_captured_graphs_and_native_prefill(tmp_path):
    """Real MLP loading/refit releases dense storage and refreshes captured planes."""
    from transformers import Qwen3Config

    from vllm.config import ModelConfig
    from vllm.model_executor.model_loader.reload.inplace import reload_weights
    from vllm.model_executor.model_loader.reload.layerwise import (
        record_metadata_for_reloading,
    )
    from vllm.model_executor.models.qwen2 import Qwen2MLP
    from vllm.model_executor.models.utils import AutoWeightsLoader

    class ProjectionModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = Qwen2MLP(2560, 9728, "silu", prefix="mlp")

        def load_weights(self, weights):
            return AutoWeightsLoader(self).load_weights(weights)

    Qwen3Config(architectures=["Qwen3ForCausalLM"]).save_pretrained(tmp_path)
    torch.manual_seed(33)
    config = VllmConfig(
        model_config=ModelConfig(
            model=str(tmp_path), skip_tokenizer_init=True, dtype="bfloat16"
        ),
        kernel_config=KernelConfig(mlp_backend="lossless_packed"),
    )
    from vllm.utils.torch_utils import set_default_torch_dtype

    with (
        set_current_vllm_config(config),
        torch.device("cuda"),
        set_default_torch_dtype(torch.bfloat16),
    ):
        model = ProjectionModel()
        weights = {
            "mlp.gate_up_proj.weight": torch.randn_like(model.mlp.gate_up_proj.weight)
            / 2560**0.5,
            "mlp.down_proj.weight": torch.randn_like(model.mlp.down_proj.weight)
            / 9728**0.5,
        }
        weights["mlp.gate_up_proj.weight"].view(-1)[:256] = 0.125
        model.load_weights(weights.items())
        record_metadata_for_reloading(model)
        before_packing_hash = config.compute_hash()
        for layer in (model.mlp.gate_up_proj, model.mlp.down_proj):
            layer.quant_method.process_weights_after_loading(layer)
        if (
            not current_platform.is_device_capability(89)
            or current_platform.num_compute_units(
                torch.accelerator.current_device_index()
            )
            != 128
        ):
            assert not list(model.named_buffers())
            x = torch.randn(7, 2560, dtype=torch.bfloat16)
            actual, _ = model.mlp.gate_up_proj(x)
            torch.testing.assert_close(
                actual,
                torch.nn.functional.linear(x, weights["mlp.gate_up_proj.weight"]),
                atol=0,
                rtol=0,
            )
            return
        assert model.mlp.gate_up_proj.weight is None
        assert model.mlp.down_proj.weight is None
        packed_hash = config.compute_hash()
        assert packed_hash != before_packing_hash
        buffers = dict(model.named_buffers())
        assert sum(t.numel() * t.element_size() for t in buffers.values()) < sum(
            t.numel() * t.element_size() for t in weights.values()
        )
        pointers = {name: t.data_ptr() for name, t in buffers.items()}
        x = torch.randn(7, 2560, dtype=torch.bfloat16)
        for _ in range(3):
            model.mlp.gate_up_proj(x)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output, _ = model.mlp.gate_up_proj(x)
        new_weights = {name: value * 0.5 for name, value in weights.items()}
        # Change one packed block into a raw exception while keeping captured storage.
        new_weights["mlp.gate_up_proj.weight"].view(-1)[:256:2] = 0
        reload_weights(config, model, new_weights.items(), None, True)
        assert pointers == {name: t.data_ptr() for name, t in model.named_buffers()}
        graph.replay()
        assert config.compute_hash() == packed_hash
        torch.testing.assert_close(
            output,
            torch.nn.functional.linear(x, new_weights["mlp.gate_up_proj.weight"]),
            atol=0.02,
            rtol=0.01,
        )
        x_prefill = torch.randn(128, 2560, dtype=torch.bfloat16)
        actual, _ = model.mlp.gate_up_proj(x_prefill)
        torch.testing.assert_close(
            actual,
            torch.nn.functional.linear(
                x_prefill, new_weights["mlp.gate_up_proj.weight"]
            ),
            atol=0,
            rtol=0,
        )
        assert model.mlp.gate_up_proj.weight is None
        exported = model.state_dict()
        assert set(exported) == set(new_weights)
        for name, value in new_weights.items():
            torch.testing.assert_close(exported[name], value.cpu(), atol=0, rtol=0)
        noncontiguous = torch.randn(4, 5120, dtype=torch.bfloat16)[:, ::2]
        layer = model.mlp.gate_up_proj
        bias = torch.randn(19456, dtype=torch.bfloat16)
        actual = layer.quant_method.apply(layer, noncontiguous, bias)
        torch.testing.assert_close(
            actual,
            torch.nn.functional.linear(
                noncontiguous, new_weights["mlp.gate_up_proj.weight"], bias
            ),
            atol=0,
            rtol=0,
        )
        with pytest.raises(ValueError, match="checkpoint-format"):
            reload_weights(config, model, new_weights.items(), None, False)
