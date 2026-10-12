#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Simulated (quantize-dequantize) FA quantization for ModelSlim checkpoints.

This is a *measurement* path, not a deployment path: the model keeps its
native weights and a native-dtype KV cache, and the attention backend replays
the quantization arithmetic on Q/K/V so the accuracy impact of a recipe can be
read off without needing the matching quantized attention kernel.

It exists because a recipe can make the exported checkpoint unusable by the
real path. ModelSlim's FA3 processor quantizes Q/K per token (dynamically) or
per head (statically) or per block, and a *dynamic* branch stores no tensors at
all -- ``QScope.PER_TOKEN`` is data-independent, so there is nothing to
calibrate and nothing to export (see ``fa3_per_token.py`` and
``processor.py`` in the ModelSlim FA3 processor). The recipe is then carried
only by a ``quant_type`` string, e.g. ``QK_INT8_DYNAMIC_V_INT8``, and every
runtime keyed on an exported ``fa_k.scale`` (including this project's own
``kvcache_quant_layers`` scan) silently sees no FA quantization at all.
"""

import torch
from vllm.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size
from vllm.logger import logger

from ..base import AscendAttentionScheme
from ..registry import register_scheme

#: Registry key, used internally to select this scheme.
FA_QUANT_SIM_SCHEME = "FAKQuantSim"

#: Per-attention-module ``quant_type`` emitted by ModelSlim when Q/K are
#: quantized per token (dynamic) and V per head (static). Assembled by
#: ``assemble_fa_quant_type`` in
#: ``msmodelslim/core/quant_service/modelslim_v1/save/ascendv1.py``: the
#: dynamic branches fold into ``QK_INT8_DYNAMIC`` and the static per-head V
#: becomes ``V_INT8`` (the ``STATIC`` suffix is only appended for
#: ``PER_CHANNEL``).
FA_QUANT_SIM_RECIPE = "QK_INT8_DYNAMIC_V_INT8"


def _fa_v_scale_weight_loader(param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
    """Loader for the per-KV-head ``fa_v.scale`` / ``fa_v.offset`` pair."""
    if param.numel() == 1 and loaded_weight.numel() == 1:
        param.data.fill_(loaded_weight.item())
        return

    loaded = loaded_weight.flatten() if loaded_weight.dim() != 1 else loaded_weight
    if loaded.numel() != param.numel():
        tp_rank = get_tensor_model_parallel_rank()
        tp_size = get_tensor_model_parallel_world_size()
        # Attention-module parameters bypass ColumnParallelLinear's sharding,
        # so every rank receives the full-width scale and has to slice it here,
        # following the same rule vLLM uses for KV heads under GQA: with
        # num_kv_heads < tp_size each head is replicated across
        # tp_size // num_kv_heads ranks and rank r owns head
        # r // (tp_size // num_kv_heads).
        num_kv_heads_total = loaded.numel()
        heads_per_rank_group = max(1, tp_size // num_kv_heads_total)
        src_head = tp_rank // heads_per_rank_group
        start = src_head * param.numel()
        if start + param.numel() > loaded.numel():
            raise AssertionError(
                "[vllm-ascend/FA-quant-sim] Cannot map the fa_v scale of "
                f"{loaded.numel()} heads onto this rank's {param.numel()} "
                f"heads (TP size {tp_size}, TP rank {tp_rank})."
            )
        loaded = loaded.narrow(0, start, param.numel())

    assert param.numel() == loaded.numel(), (
        "[vllm-ascend/FA-quant-sim] fa_v scale size mismatch: parameter has "
        f"{param.numel()} elements but the sliced weight has {loaded.numel()} "
        f"(TP size {get_tensor_model_parallel_world_size()}, TP rank "
        f"{get_tensor_model_parallel_rank()})."
    )
    param.data.copy_(loaded.to(param.dtype).view_as(param.data))


@register_scheme(FA_QUANT_SIM_SCHEME, "attention")
class AscendFAQuantSimAttentionMethod(AscendAttentionScheme):
    """QDQ stand-in for the ``QK_INT8_DYNAMIC_V_INT8`` FA recipe.

    Enabled by ``VLLM_ASCEND_FA_QUANT_SIMULATE=1``. It creates the V scale
    parameter the checkpoint does export (``fa_v.scale``, per KV head) and
    installs :class:`~vllm_ascend.attention.attention_v1.AscendFAQuantSimAttentionBackendImpl`
    on the layer, which performs the round trip. Q/K need no parameter: they
    are quantized dynamically at runtime.
    """

    def __init__(self, quant_description: dict, prefix: str):
        self.quant_description = quant_description
        self.prefix = prefix

    def create_weights(self, layer: torch.nn.Module) -> None:
        num_kv_heads = getattr(layer, "num_kv_heads", None)
        if num_kv_heads is None:
            raise NotImplementedError(
                "[vllm-ascend/FA-quant-sim] The simulated FA recipe needs a dense "
                f"full-attention layer exposing num_kv_heads; {self.prefix} has none. "
                "MLA layers follow a different (per-head, no per-token) recipe and "
                "should use the FAKQuant path instead."
            )

        dtype = torch.get_default_dtype()
        layer.fa_v = torch.nn.Module()
        for name, init in (
            ("scale", torch.ones(num_kv_heads, 1, dtype=dtype)),
            ("offset", torch.zeros(num_kv_heads, 1, dtype=torch.int8)),
        ):
            param = torch.nn.Parameter(init, requires_grad=False)
            param.weight_loader = _fa_v_scale_weight_loader
            layer.fa_v.register_parameter(name, param)

        # Upgrade impl to the QDQ subclass so the simulated forward is used for
        # every subsequent step. The base class's own forward stays reachable
        # through super(), which is how the QDQ output is fed on.
        if hasattr(layer, "impl"):
            from vllm_ascend.attention.attention_v1 import AscendFAQuantSimAttentionBackendImpl

            layer.impl.__class__ = AscendFAQuantSimAttentionBackendImpl

        logger.info_once(
            "[vllm-ascend/FA-quant-sim] Simulating FA quantization "
            f"({FA_QUANT_SIM_RECIPE}): Q/K int8 per token-head (dynamic), "
            "V int8 per head (checkpoint scale), KV cache in the model dtype."
        )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Reshape to the broadcast shape for the (T, H, D) projection output and
        # widen to fp32 to match the reference arithmetic. The scale stays on
        # the layer, where the backend reads it, exactly as the C8 and MXFP8
        # schemes leave their per-head cache scales.
        layer.fa_v.scale.data = layer.fa_v.scale.data.to(torch.float32).clamp_min(1e-10).view(1, -1, 1)
        # Symmetric per-head ignores the offset, so drop it instead of carrying
        # a parameter that is never read.
        layer.fa_v.offset = None

    def apply(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache,
        attn_metadata,
        attn_type,
        scale,
        output,
    ) -> torch.Tensor:
        raise RuntimeError(
            "[vllm-ascend/FA-quant-sim] AscendFAQuantSimAttentionMethod.apply should not "
            "be called. The simulated FA quantization is applied by "
            "AscendFAQuantSimAttentionBackendImpl."
        )
