# Copyright 2025 The Torch-Spyre Authors.
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

from contextlib import contextmanager

from typing import Callable, Optional, Sequence, Union, Tuple
import torch
import torch._decomp as decomp
from .errors import Unsupported
from . import customops  # noqa: F401
from torch_spyre._C import get_device_dtype, SpyreTensorLayout, as_strided_with_layout

import threading

# A module-level lock to make the CM thread-safe
_decompositions_lock = threading.RLock()

# Dictionary for Spyre-specific decompositions
spyre_decompositions: dict = {}

# Exclude specific Inductor default decompositions on Spyre.
# Some Inductor decompositions do not work reliably on the Spyre backend yet.
# We disable them here and rely on implicit fallbacks to eager ops instead. Once
# the blocking issues are resolved, these exclusions can be removed.
spyre_decompositions_to_exclude = [
    # The default decomposition for torch.new_ones (defined in pytorch/torch/refs/__init__.py)
    # uses torch.full, which is not yet supported in Spyre eager mode.
    # See: https://github.com/torch-spyre/torch-spyre/issues/128#issuecomment-3576168221
    torch.ops.aten.new_ones,
]


def register_spyre_decomposition(
    ops: Union[torch._ops.OperatorBase, list],
):
    """
    Register decompositions specifically for Spyre device.
    These will only be active when compiling for the Spyre device.
    """
    return decomp.register_decomposition(ops, spyre_decompositions)


# Context manager that enables spyre specific decompositions in addition to PyTorch in-tree decompositions
@contextmanager
def enable_spyre_decompositions(
    decomps: Optional[dict[torch._ops.OperatorBase, Callable]] = None,
):
    """
    CM that enables Spyre decompositions:
      - Temporarily adds relevant Spyre decompositions to provided decomposition table `decomps`
      - Restore original decompositions table on exit

    This CM is reentrant and safe under nested usage.

    Args:
        decomps: Decomposition table to modify. Maps operator overloads to their
            decomposition implementations. Defaults to PyTorch Inductor's global
            decomposition registry (torch._inductor.decomposition.decompositions).
    """
    if decomps is None:
        decomps = torch._inductor.decomposition.decompositions

    with _decompositions_lock:
        from torch_spyre.fallbacks import fallback_ops
        from torch._ops import OpOverload, OpOverloadPacket

        # Helper function to remove ops from decompositions
        def _fetch_and_remove_op(ops):
            _removed = {}
            for op in ops:
                if isinstance(op, OpOverloadPacket):
                    for overload_name in op.overloads():
                        opo = getattr(op, overload_name)
                        op_ret = decomps.pop(opo, None)
                        if op_ret is not None:
                            _removed[opo] = op_ret
                elif isinstance(op, OpOverload):
                    op_ret = decomps.pop(op, None)
                    if op_ret is not None:
                        _removed[op] = op_ret
            return _removed

        # 1. Add/override spyre-specific decompositions
        saved_intree_decompositions = {}
        for (
            spyre_decompositions_op,
            spyre_decompositions_impl,
        ) in spyre_decompositions.items():
            if spyre_decompositions_op in decomps:
                saved_intree_decompositions[spyre_decompositions_op] = decomps[
                    spyre_decompositions_op
                ]
            decomps[spyre_decompositions_op] = spyre_decompositions_impl

        # Attach to the function so we can restore on last exit
        enable_spyre_decompositions._saved_decompositions = saved_intree_decompositions

        # 2. Remove selected decompositions from Inductor's registry for spyre
        _removed_decompositions_to_exclude = _fetch_and_remove_op(
            spyre_decompositions_to_exclude
        )

        # Attach to the function so we can restore on last exit
        enable_spyre_decompositions._removed_decompositions_to_exclude = (
            _removed_decompositions_to_exclude
        )

        # 3. Remove selected decompositions for fallback ops defined in fallbacks.py
        _removed_decompositions_fallback_ops = _fetch_and_remove_op(fallback_ops)

        # Attach to the function so we can restore on last exit
        enable_spyre_decompositions._removed_decompositions_fallback_ops = (
            _removed_decompositions_fallback_ops
        )

        try:
            yield decomps
        finally:
            # Inverse order compared to when entering the context manager

            # 1. Revert selected decompositions that have been marked for fallback ops
            removed_decompositions_fallback_ops = getattr(
                enable_spyre_decompositions,
                "_removed_decompositions_fallback_ops",
                {},
            )
            [
                torch._decomp._add_op_to_registry(decomps, op, fn)
                for op, fn in removed_decompositions_fallback_ops.items()
            ]

            # 2. Revert selected decompositions that have been removed from Inductor's registry for spyre
            removed_decompositions_to_exclude = getattr(
                enable_spyre_decompositions,
                "_removed_decompositions_to_exclude",
                {},
            )
            [
                torch._decomp._add_op_to_registry(decomps, op, fn)
                for op, fn in removed_decompositions_to_exclude.items()
            ]

            # 3. Reset the saved in-tree lowerings if needed
            saved_intree_decompositions = getattr(
                enable_spyre_decompositions, "_saved_decompositions", {}
            )
            for (
                spyre_decompositions_op,
                spyre_decompositions_impl,
            ) in spyre_decompositions.items():
                if spyre_decompositions_op in saved_intree_decompositions:
                    decomps[spyre_decompositions_op] = saved_intree_decompositions[
                        spyre_decompositions_op
                    ]
                else:
                    decomps.pop(spyre_decompositions_op, None)

            # Clean up
            enable_spyre_decompositions._saved_decompositions = {}
            enable_spyre_decompositions._removed_decompositions_to_exclude = {}
            enable_spyre_decompositions._removed_decompositions_fallback_ops = {}


@decomp.register_decomposition([torch.ops.spyre.compact])
def compact_decomp(x: torch.Tensor) -> torch.Tensor:
    return torch.ops.spyre.slice(torch.ops.spyre.swap(x))


@register_spyre_decomposition([torch.ops.spyre.layer_norm])
def layernorm_decomp(
    input: torch.Tensor,
    normalized_shape: list[int],
    weight: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
) -> torch.Tensor:
    mean = torch.ops.spyre.exx2(input, 1.0 / normalized_shape[0], False)
    norm_mean = torch.ops.spyre.layernormscale(mean, eps)
    return torch.ops.spyre.layernormnorm(input, mean, norm_mean, weight, bias)


@register_spyre_decomposition([torch.ops.spyre.rms_norm])
def rmsnorm_decomp(
    input: torch.Tensor,
    normalized_shape: list[int],
    weight: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
) -> torch.Tensor:
    # TODO: limitation with mean on dim=-1, transpose for now to avoid
    # https://github.com/torch-spyre/torch-spyre/issues/632
    input = input.transpose(-1, -2).contiguous()
    eps = torch.ops.spyre.full(input.shape, eps, dtype=torch.float16, device="spyre")
    rsqrt_inp = torch.rsqrt(torch.mean(input * input, dim=-2, keepdim=True)) + eps
    output = (input * rsqrt_inp).transpose(-1, -2).contiguous()
    if weight is not None:
        output = output * weight
    return output


# TODO (imaihal): Inductor applies constant folding to torch.full, which allocates
# a one-element Spyre tensor. This currently fails because Spyre does not handle
# single-element tensors well.
# Ref: https://github.com/pytorch/pytorch/blob/v2.9.1/torch/_inductor/fx_passes/joint_graph.py#L324-L335
#
# To avoid constant folding, we introduce a custom op `spyre::full` that runs
# torch.full on CPU and copies the result to Spyre. Remove this workaround once
# Spyre supports one-element tensors.
@register_spyre_decomposition([torch.ops.aten.full])
def full_decomp(
    size: list[Union[int, torch.SymInt]],
    fill_value: torch.types.Number,
    dtype: Optional[torch.dtype] = None,
    layout: Optional[torch.layout] = None,
    device: Optional[torch.device] = None,
    pin_memory: Optional[bool] = None,
) -> torch.Tensor:
    assert layout in (torch.strided, None), f"doesn't support layout={layout}"
    assert not pin_memory, f"doesn't support pin_memory={pin_memory}"
    return torch.ops.spyre.full(size, fill_value, device, dtype=dtype)


"""
Hook torch.nn.functional.layer_norm to select spyre optimized version where applicable
"""
orig_layer_norm = torch.nn.functional.layer_norm


def spyre_layer_norm(
    input: torch.Tensor,
    normalized_shape: Sequence[int],
    weight: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
) -> torch.Tensor:
    if input.device.type == "spyre" and len(normalized_shape) == 1:
        return torch.ops.spyre.layer_norm(input, normalized_shape, weight, bias, eps)
    else:
        return orig_layer_norm(input, normalized_shape, weight, bias, eps)


torch.nn.functional.layer_norm = spyre_layer_norm

orig_rms_norm = torch.nn.functional.rms_norm


def spyre_rms_norm(
    input: torch.Tensor,
    normalized_shape: list[int],
    weight: Optional[torch.Tensor] = None,
    eps: Optional[float] = None,
) -> torch.Tensor:
    if input.device.type == "spyre" and len(normalized_shape) == 1:
        return torch.ops.spyre.rms_norm(input, normalized_shape, weight, eps)
    elif input.device.type == "spyre" and len(normalized_shape) != 1:
        raise Unsupported("RMSNorm reducing more than 1 dimension")
    else:
        return orig_rms_norm(input, normalized_shape, weight, eps)


torch.nn.functional.rms_norm = spyre_rms_norm

orig_gelu = torch.nn.functional.gelu


def spyre_gelu(
    input: torch.Tensor,
    approximate: str = "none",
) -> torch.Tensor:
    if input.device.type == "spyre":
        return torch.ops.spyre.gelu(input, approximate)
    else:
        return orig_gelu(input, approximate=approximate)


torch.nn.functional.gelu = spyre_gelu


orig_softplus = torch.nn.functional.softplus


def spyre_softplus(
    input: torch.Tensor, beta: float = 1.0, threshold: float = 20.0
) -> torch.Tensor:
    if input.device.type == "spyre":
        return torch.ops.spyre.softplus(input, beta, threshold)
    else:
        return orig_softplus(input, beta, threshold)


torch.nn.functional.softplus = spyre_softplus


@register_spyre_decomposition([torch.ops.aten.gt.Tensor, torch.ops.aten.gt.Tensor_out])
def gt_decomp(
    input: torch.Tensor, other: torch.Tensor, *, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    # TODO: Implement greaterthan in the backend compiler
    out_ge = torch.ge(input, other).to(dtype=torch.float16)
    out_ne = torch.ne(input, other).to(dtype=torch.float16)
    return torch.mul(out_ge, out_ne, out=out).to(dtype=torch.bool)


@register_spyre_decomposition([torch.ops.aten.lt.Tensor, torch.ops.aten.lt.Tensor_out])
def lt_decomp(
    input: torch.Tensor, other: torch.Tensor, *, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    # TODO: Implement lessthan in the backend compiler
    out_le = torch.le(input, other).to(dtype=torch.float16)
    out_ne = torch.ne(input, other).to(dtype=torch.float16)
    return torch.mul(out_le, out_ne, out=out).to(dtype=torch.bool)


@register_spyre_decomposition([torch.ops.aten.logical_not])
def logical_not_decomp(input: torch.Tensor) -> torch.Tensor:
    # Currently falling back to torch.zeros_like for dtypes other than bool
    # This is needed until scalar False/0.0 or constant tensor [False]/[0.0] is supported
    if input.dtype is torch.bool:
        zero = torch.ne(input, input)
    else:
        zero = torch.zeros_like(input)
    return torch.eq(input, zero)


# Monkey-patch FMS RotaryEmbedding.adjsuted_qk
try:
    from fms.modules.positions import RotaryEmbedding

    _original_adjusted_qk = RotaryEmbedding.adjusted_qk

    def spyre_rot_emb_adjusted_qk(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        past_kv_state: Optional[Tuple[torch.Tensor | None, torch.Tensor | None]] = None,
        use_cache=False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if q.device.type != "spyre":
            print("running on CPU")
            return _original_adjusted_qk(
                self, q, k, position_ids, past_kv_state, use_cache
            )
        assert len(q.size()) == 4
        assert len(k.size()) == 4

        # Spyre implementation
        seq_len = max(k.size(1), q.size(1))

        if position_ids is None:
            position_ids = torch.arange(seq_len, device="cpu").unsqueeze(0)
            if (
                use_cache
                and past_kv_state is not None
                and past_kv_state[0] is not None
                and past_kv_state[0].numel() > 0
            ):
                position_ids += past_kv_state[0].size(2)

        if self.partial_rope != 1.0:
            q_rope = q[..., : self.dim]
            k_rope = k[..., : self.dim]
        else:
            q_rope = q
            k_rope = k

        B, L, H, D = q_rope.shape  # example [B,L,H,D] = [2, 128, 8, 64]
        new_device_shape = [L, H, D // 128, 2, B, 64]
        new_dim_map = [1, 2, 3, 4, 0, 3]
        new_layout = SpyreTensorLayout(
            new_device_shape,  # [L, H, D/128, 2, B, 64]
            new_dim_map,  # [1, 2, 3, 4, 0, 3]
            get_device_dtype(torch.float16),
        )
        new_cpu_shape = (B, L, H, D // 2, 2)
        # Strides: assuming original [B,L,H,D] was contiguous with strides [L*H*D, H*D, D, 1]
        # For [B,L,H,D/2,2]: strides = [L*H*D, H*D, D, 2, 1]
        new_strides = (L * H * D, H * D, D, 2, 1)

        # Apply transformation
        q_rope_spyre = as_strided_with_layout(
            q_rope,
            new_cpu_shape,  # [B, L, H, D/2, 2]
            new_strides,  # [L*H*D, H*D, D, 2, 1]
            q_rope.storage_offset(),  # Usually 0
            new_layout,  # SpyreTensorLayout with new device shape & dim_map
        )
        k_rope_spyre = as_strided_with_layout(
            k_rope,
            new_cpu_shape,  # [B, L, H, D/2, 2]
            new_strides,  # [L*H*D, H*D, D, 2, 1]
            k_rope.storage_offset(),  # Usually 0
            new_layout,  # SpyreTensorLayout with new device shape & dim_map
        )

        # q_ = q_rope.view(*q.size()[:-1], -1, 2)  # B L H D/2 2
        # k_ = k_rope.view(*k.size()[:-1], -1, 2)

        # the max start position should be based on the max first position of each sequence
        # position_ids_cpu = position_ids.to('cpu')
        # max_start_pos = torch.max(position_ids_cpu[:, 0])
        max_start_pos = 4096
        alpha = self.compute_freqs_cis(q.device, max_seq_len=max_start_pos + seq_len)
        print(f"{alpha=}")
        print(f"{position_ids.device=}, {position_ids.to('cpu')=}")
        print(f"{q.device=}, {q.device.index=}")
        # TODO: cached freqs on CPU for now
        freqs = self.cached_freqs[q.device.index][alpha][position_ids]
        print(f"{freqs.shape=}")
        freqs_spyre = freqs.to(dtype=torch.float16).to("spyre")

        freqs_device_shape = [L, 1, D // 128, 2, 2, B, 64]
        freqs_dim_map = [1, 2, 3, 4, 5, 0, 3]

        freqs_layout = SpyreTensorLayout(
            freqs_device_shape, freqs_dim_map, get_device_dtype(torch.float16)
        )

        # Add dimension at position 2 for broadcasting with H
        # freqs_expanded: [B, L, 1, D/2, 2, 2]
        freqs_expanded = freqs_spyre[:, :, None, :, :, :].to(torch.float16)

        # Assuming contiguous: strides = [L*1*D/2*2*2, 1*D/2*2*2, D/2*2*2, 2*2, 2, 1]
        freqs_cpu_shape = (B, L, 1, D // 2, 2, 2)
        actual_strides = freqs_expanded.stride()

        freqs_transformed = as_strided_with_layout(
            freqs_expanded,
            freqs_cpu_shape,
            actual_strides,
            freqs_expanded.storage_offset(),
            freqs_layout,
        )

        q_out = (
            freqs_transformed[:, -q.size(1) :, :, :, :, :]
            .mul(q_rope_spyre.unsqueeze(4))
            .sum(5)
            .flatten(3)
        ).type_as(q)
        k_out = (
            freqs_transformed[:, -k.size(1) :, :, :, :, :]
            .mul(k_rope_spyre.unsqueeze(4))
            .sum(5)
            .flatten(3)
        ).type_as(k)

        # Handle partial rope
        if self.partial_rope != 1.0:
            q_out = torch.cat([q_out, q[..., self.dim :]], dim=-1)
            k_out = torch.cat([k_out, k[..., self.dim :]], dim=-1)

        return q_out, k_out

    # Monkey-patch the method
    RotaryEmbedding.adjusted_qk = spyre_rot_emb_adjusted_qk
except ImportError:
    # triggers when FMS not installed
    pass
