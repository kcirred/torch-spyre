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

from typing import Callable, Optional, Sequence, Union, Tuple, Any
import torch
import torch._decomp as decomp
from .errors import Unsupported
from . import customops  # noqa: F401

import threading

# A module-level lock to make the CM thread-safe
_decompositions_lock = threading.RLock()

# Dictionary for Spyre-specific decompositions
spyre_decompositions: dict = {}

# Exclude specific Inductor default decompositions on Spyre.
# Some Inductor decompositions do not work reliably on the Spyre backend yet.
# We disable them here and rely on implicit fallbacks to eager ops instead. Once
# the blocking issues are resolved, these exclusions can be removed.
#
spyre_decompositions_to_exclude: list = []


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
        from torch_spyre.ops.fallbacks import fallback_ops
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
# Implement ones via identity broadcast: create a size-1 tensor (ones_scalar), expand to
# target size, then clone (identity) to materialize. Clone op with identity is merged.
@register_spyre_decomposition([torch.ops.aten.ones.default])
def ones_decomp(
    size: Union[list, tuple],
    *,
    dtype: Optional[torch.dtype] = None,
    layout: Optional[torch.layout] = None,
    device: Optional[torch.device] = None,
    pin_memory: Optional[bool] = None,
) -> torch.Tensor:
    assert layout in (torch.strided, None), f"doesn't support layout={layout}"
    assert not pin_memory, f"doesn't support pin_memory={pin_memory}"
    scalar = torch.ops.spyre.ones_scalar(device, dtype=dtype)
    expanded = scalar.expand(size)
    return expanded.clone()


@register_spyre_decomposition([torch.ops.aten.new_ones.default])
def new_ones_decomp(
    self: torch.Tensor,
    size: Union[list, tuple],
    *,
    dtype: Optional[torch.dtype] = None,
    layout: Optional[torch.layout] = None,
    device: Optional[torch.device] = None,
    pin_memory: Optional[bool] = None,
) -> torch.Tensor:
    assert layout in (torch.strided, None), f"doesn't support layout={layout}"
    assert not pin_memory, f"doesn't support pin_memory={pin_memory}"
    dev = device if device is not None else self.device
    dt = dtype if dtype is not None else self.dtype
    scalar = torch.ops.spyre.ones_scalar(dev, dtype=dt)
    expanded = scalar.expand(size)
    return expanded.clone()


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
    _original_compute_freqs_cis = RotaryEmbedding.compute_freqs_cis

    def spyre_compute_freqs_cis(self, device, max_seq_len=2048):
        alpha = self.rope_scaling.get_alpha(max_seq_len)

        if device == torch.device("meta"):
            return alpha

        dev_idx = device.index

        if dev_idx not in self.cached_freqs:
            self.cached_freqs[dev_idx] = {}
        if dev_idx not in self.max_seq_len_cached:
            self.max_seq_len_cached[dev_idx] = 0

        if alpha not in self.cached_freqs[dev_idx]:
            # This avoids a graph break from computing scaled_max_seq_len if not needed
            scaled_max_seq_len = self.rope_scaling.scaled_max_seq_len(
                max_seq_len, alpha
            )
            if scaled_max_seq_len > self.max_seq_len_cached[dev_idx]:
                # This only runs if a particular combination of alpha
                # and max_seq_len hasn't been seen before
                freqs = self.rope_scaling.compute_scaled_freqs(device, alpha)
                t = torch.arange(scaled_max_seq_len, device="cpu", dtype=freqs.dtype)
                freqs = torch.outer(t, freqs.to("cpu")).float()
                self.max_seq_len_cached[dev_idx] = scaled_max_seq_len
                self.cached_freqs[dev_idx][alpha] = (
                    torch.stack(
                        [
                            torch.cos(freqs),
                            -torch.sin(freqs),
                            torch.sin(freqs),
                            torch.cos(freqs),
                        ],
                        dim=2,
                    )
                    .view(*freqs.size(), 2, 2)
                    .permute([0, 2, 3, 1])
                    .clone()
                    .to(dtype=torch.float16)
                )  # S, 64, 2, 2 -> S, 2, 2, 64
        return alpha

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
            k_rope: Any = k[..., : self.dim]
        else:
            q_rope = q
            k_rope = k

        # call interleave on the weight 64, 2

        # Added transpose to make B,L,H,D -> B,H,L,D
        # if only tranpose is called, the layout is still the same as original, but can be viewed as if it is transposed
        # but if I call contiguous() then it will rearrange the layout to match the new transpose.
        # q_rope = q_rope.transpose(1,2).contiguous()
        # k_rope = k_rope.transpose(1,2).contiguous()

        B, L, H, D = q_rope.shape
        print(f"{B=}{L=},{H=},{D=}")

        q_rope_spyre = q_rope.view(*q_rope.size()[:-1], 2, -1)  # B L H D/2 2
        k_rope_spyre = k_rope.view(*k_rope.size()[:-1], 2, -1)  # B L H D/2 2
        # 568-572 in granite.py, just remove hf_to_fms_adapter remove
        # rot emb has to be modified to 64, 2, 2under compute_freq_cis  instead of S, 64, 2,2 -> S, 2,2, 64
        # q_ = q_rope.view(*q.size()[:-1], -1, 2)  # B L H D/2 2
        # k_ = k_rope.view(*k.size()[:-1], -1, 2)

        # the max start position should be based on the max first position of each sequence
        # position_ids_cpu = position_ids.to('cpu')
        # max_start_pos = torch.max(position_ids_cpu[:, 0])
        max_start_pos = 4096
        if q.device.type != "spyre":
            alpha = _original_compute_freqs_cis(
                q.device, max_seq_len=max_start_pos + seq_len
            )
        else:
            alpha = self.compute_freqs_cis(
                q.device, max_seq_len=max_start_pos + seq_len
            )
        freqs = self.cached_freqs[q.device.index][alpha][position_ids]
        print(f"{freqs.shape=}")
        freqs_spyre = freqs.to(dtype=torch.float16).to("spyre")  # [B, L, D/2, 2, 2]
        print(f"{freqs_spyre.shape=}")
        freqs_expanded = freqs_spyre.unsqueeze(1)
        print(f"{freqs_expanded.shape=}")

        print(f"{q_rope_spyre.shape=}")

        freqs_q_modified = freqs_spyre[
            :, -q_rope_spyre.size(1) :, :, :, :
        ]  # [B, L, D/2, 2, 2]
        print(f"{freqs_q_modified.shape=}")
        freqs_q_modified = freqs_q_modified.unsqueeze(2)
        print(f"{freqs_q_modified.shape=}")
        q_rope_modified = q_rope_spyre.unsqueeze(-2)
        print(f"{q_rope_modified.shape=}")

        q_out = (
            freqs_q_modified.mul(q_rope_modified)
            # .sum(5)
            # .flatten(3)
        ).type_as(q)
        k_out = (
            freqs_spyre[:, -k_rope_spyre.size(1) :, :, :, :]
            .unsqueeze(2)  # -> [1, L, 1, 1,1,2,D/2]
            .mul(k_rope_spyre.unsqueeze(-2))  # [B, L, H, D/2, 1, 2] -> [B,L,H,1,2,D/2]
            # .sum(5)
            # .flatten(3)
        ).type_as(k)

        # Handle partial rope
        if self.partial_rope != 1.0:
            q_out = torch.cat([q_out, q[..., self.dim :]], dim=-1)
            k_out = torch.cat([k_out, k[..., self.dim :]], dim=-1)

        return q_out, k_out

    RotaryEmbedding.adjusted_qk = spyre_rot_emb_adjusted_qk
    RotaryEmbedding.compute_freqs_cis = spyre_compute_freqs_cis
except ImportError:
    # triggers when FMS not installed
    pass
