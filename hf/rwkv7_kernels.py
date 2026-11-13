"""Kernel backend for RWKV7.

This file is the only place that knows about FLA.  The rest of
``modeling_rwkv7.py`` calls these wrappers, so a CUDA/TileLang backend can be
dropped in by rewriting this module.

Backend selection
-----------------
``RWKV7_KERNEL_BACKEND``:
  * ``auto`` (default): use FLA if importable, otherwise pure PyTorch.
  * ``fla``: require FLA and use its Triton kernels.
  * ``torch``: pure PyTorch only (no FLA import at all).

``RWKV7_LOCAL_HELPERS=1`` additionally forces the local ``token_shift`` and
``sqrelu`` implementations even when the FLA backend is active.
"""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F


_BACKEND = os.environ.get("RWKV7_KERNEL_BACKEND", "auto").strip().lower()
if _BACKEND not in {"auto", "fla", "torch"}:
    raise ValueError(
        f"RWKV7_KERNEL_BACKEND must be 'auto', 'fla' or 'torch', got {_BACKEND!r}"
    )

_HAS_FLA = False
if _BACKEND in {"auto", "fla"}:
    try:
        from fla.ops.rwkv7 import chunk_rwkv7 as _fla_chunk_rwkv7
        from fla.ops.rwkv7 import (
            fused_mul_recurrent_rwkv7 as _fla_fused_mul_recurrent_rwkv7,
        )
        from fla.ops.rwkv7.fused_addcmul import fused_addcmul_rwkv7 as _fla_addcmul
        from fla.ops.rwkv7.fused_k_update import fused_k_rwkv7 as _fla_k_update
        from fla.ops.rwkv7.gate_output_correction import (
            gate_output_correction as _fla_output_correction,
        )
        from fla.modules.activations import ACT2FN as _FLA_ACT2FN
        from fla.modules.token_shift import token_shift as _fla_token_shift

        _HAS_FLA = True
    except ImportError:
        if _BACKEND == "fla":
            raise

BACKEND_NAME = "fla" if _HAS_FLA else "torch"


# ---------------------------------------------------------------------------
# FLA wrappers
# ---------------------------------------------------------------------------


def _fla_wkv_chunk(
    r: torch.Tensor,
    w: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kk: torch.Tensor,
    a: torch.Tensor,
    initial_state: torch.Tensor | None,
    output_final_state: bool,
    cu_seqlens: torch.LongTensor | None,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    return _fla_chunk_rwkv7(
        r=r,
        w=w,
        k=k,
        v=v,
        a=-kk,
        b=kk * a,
        scale=1.0,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        safe_gate=True,
        chunk_size=chunk_size,
    )


def _fla_wkv_recurrent(
    r: torch.Tensor,
    w: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kk: torch.Tensor,
    a: torch.Tensor,
    initial_state: torch.Tensor | None,
    output_final_state: bool,
    cu_seqlens: torch.LongTensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    return _fla_fused_mul_recurrent_rwkv7(
        r=r,
        w=w,
        k=k,
        v=v,
        kk=kk,
        a=a,
        scale=1.0,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
    )


# ---------------------------------------------------------------------------
# Pure-PyTorch kernels
# ---------------------------------------------------------------------------


def _torch_addcmul(
    x: torch.Tensor,
    delta: torch.Tensor,
    x_r: torch.Tensor,
    x_w: torch.Tensor,
    x_k: torch.Tensor,
    x_v: torch.Tensor,
    x_a: torch.Tensor,
    x_g: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    return (
        x + delta * x_r,
        x + delta * x_w,
        x + delta * x_k,
        x + delta * x_v,
        x + delta * x_a,
        x + delta * x_g,
    )


def _torch_k_update(
    k: torch.Tensor, a: torch.Tensor, k_a: torch.Tensor
) -> torch.Tensor:
    return k * (1 + (a - 1) * k_a)


def _torch_wkv(
    r: torch.Tensor,
    w: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kk: torch.Tensor,
    a: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Reference WKV recurrence in float32.

    State layout is ``[B, H, K, V]``; the update for each token is

        sa_i    = sum_j S[j, i] * kk_j
        S[j, i] = exp(w_j) * S[j, i] - (kk_j a_j) * sa_i + k_j v_i
        y_i     = sum_j r_j * S[j, i]
    """
    batch, time, heads, key_dim = r.shape
    value_dim = v.shape[-1]
    output_dtype = v.dtype

    r = r.float()
    w = w.float()
    k = k.float()
    v = v.float()
    kk = kk.float()
    a = a.float()

    if cu_seqlens is not None:
        if batch != 1:
            raise ValueError("cu_seqlens mode requires batch_size=1.")
        lengths = [int(x) for x in cu_seqlens]
        if lengths[-1] != time:
            raise ValueError(
                "cu_seqlens must cover the full packed sequence: "
                f"got last={lengths[-1]}, T={time}."
            )
        initial = initial_state.float() if initial_state is not None else None
        output = torch.empty_like(r, dtype=torch.float32)
        finals = []
        for seq_idx in range(len(lengths) - 1):
            start, end = lengths[seq_idx], lengths[seq_idx + 1]
            state = (
                initial[seq_idx].clone()
                if initial is not None
                else r.new_zeros((heads, key_dim, value_dim), dtype=torch.float32)
            )
            for t in range(start, end):
                kk_t = kk[0, t]
                a_t = a[0, t]
                w_t = w[0, t]
                k_t = k[0, t]
                v_t = v[0, t]
                r_t = r[0, t]
                sa = torch.einsum("hji,hj->hi", state, kk_t)
                state = (
                    state * torch.exp(w_t).unsqueeze(-1)
                    - (kk_t * a_t).unsqueeze(-1) * sa.unsqueeze(-2)
                    + k_t.unsqueeze(-1) * v_t.unsqueeze(-2)
                )
                output[0, t] = torch.einsum("hji,hj->hi", state, r_t)
            finals.append(state)
        final_state = torch.stack(finals) if output_final_state else None
        return output.to(dtype=output_dtype), final_state

    state = (
        initial_state.float()
        if initial_state is not None
        else r.new_zeros((batch, heads, key_dim, value_dim), dtype=torch.float32)
    )
    outputs = []
    for t in range(time):
        kk_t = kk[:, t]
        a_t = a[:, t]
        w_t = w[:, t]
        k_t = k[:, t]
        v_t = v[:, t]
        r_t = r[:, t]
        sa = torch.einsum("bhji,bhj->bhi", state, kk_t)
        state = (
            state * torch.exp(w_t).unsqueeze(-1)
            - (kk_t * a_t).unsqueeze(-1) * sa.unsqueeze(-2)
            + k_t.unsqueeze(-1) * v_t.unsqueeze(-2)
        )
        outputs.append(torch.einsum("bhji,bhj->bhi", state, r_t))
    output = torch.stack(outputs, dim=1)
    return output.to(dtype=output_dtype), (state if output_final_state else None)


def _torch_output_correction(
    o: torch.Tensor,
    r: torch.Tensor,
    k: torch.Tensor,
    r_k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
) -> torch.Tensor:
    correction = (
        (r * k * r_k.unsqueeze(0).unsqueeze(0)).sum(dim=-1, keepdim=True) * v
    ).view_as(o)
    return (o + correction) * g


# ---------------------------------------------------------------------------
# Public dispatch
# ---------------------------------------------------------------------------


def addcmul_rwkv7(
    x: torch.Tensor,
    delta: torch.Tensor,
    x_r: torch.Tensor,
    x_w: torch.Tensor,
    x_k: torch.Tensor,
    x_v: torch.Tensor,
    x_a: torch.Tensor,
    x_g: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    if _HAS_FLA:
        return _fla_addcmul(x, delta, x_r, x_w, x_k, x_v, x_a, x_g)
    return _torch_addcmul(x, delta, x_r, x_w, x_k, x_v, x_a, x_g)


def k_update_rwkv7(
    k: torch.Tensor, a: torch.Tensor, k_a: torch.Tensor
) -> torch.Tensor:
    if _HAS_FLA:
        return _fla_k_update(k, a, k_a)
    return _torch_k_update(k, a, k_a)


def wkv_chunk(
    r: torch.Tensor,
    w: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kk: torch.Tensor,
    a: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if _HAS_FLA:
        return _fla_wkv_chunk(
            r,
            w,
            k,
            v,
            kk,
            a,
            initial_state,
            output_final_state,
            cu_seqlens,
            chunk_size,
        )
    return _torch_wkv(
        r, w, k, v, kk, a, initial_state, output_final_state, cu_seqlens
    )


def wkv_recurrent(
    r: torch.Tensor,
    w: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kk: torch.Tensor,
    a: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if _HAS_FLA:
        return _fla_wkv_recurrent(
            r, w, k, v, kk, a, initial_state, output_final_state, cu_seqlens
        )
    return _torch_wkv(
        r, w, k, v, kk, a, initial_state, output_final_state, cu_seqlens
    )


def output_correction(
    o: torch.Tensor,
    r: torch.Tensor,
    k: torch.Tensor,
    r_k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
) -> torch.Tensor:
    if _HAS_FLA:
        return _fla_output_correction(o, r, k, r_k, v, g)
    return _torch_output_correction(o, r, k, r_k, v, g)


# ---------------------------------------------------------------------------
# token_shift / sqrelu
# ---------------------------------------------------------------------------


def _token_shift_torch(
    x: torch.Tensor,
    cu_seqlens: torch.LongTensor | None = None,
    cache: torch.Tensor | None = None,
    output_cache: bool = False,
):
    """Token shift used by both attention and FFN."""
    if cu_seqlens is not None:
        if x.dim() != 3 or x.shape[0] != 1:
            raise ValueError("cu_seqlens mode expects x with shape [1, T, D].")
        dim = x.shape[2]
        delta = torch.empty_like(x)
        if cache is not None and cache.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(
                "The number of cache rows must match the number of packed "
                f"sequences: {cache.shape[0]} vs {len(cu_seqlens) - 1}."
            )
        cache_out = []
        for seq_idx in range(len(cu_seqlens) - 1):
            start = int(cu_seqlens[seq_idx])
            end = int(cu_seqlens[seq_idx + 1])
            if end <= start:
                continue
            if cache is None:
                previous = torch.zeros(1, dim, device=x.device, dtype=x.dtype)
            else:
                previous = cache[seq_idx].reshape(1, dim)
            if end - start > 1:
                previous = torch.cat([previous, x[0, start : end - 1]], dim=0)
            delta[0, start:end] = previous - x[0, start:end]
            cache_out.append(x[0, end - 1])
        cache_tensor = (
            torch.stack(cache_out) if cache_out else x.new_zeros((0, dim))
        )
        if output_cache:
            return delta, cache_tensor
        return delta

    if cache is not None:
        if x.shape[1] == 1:
            previous = cache.reshape_as(x)
            next_cache = x[:, -1]
            if output_cache:
                return previous - x, next_cache
            return previous - x
        shifted = torch.cat((cache.reshape(x.shape[0], 1, -1), x[:, :-1]), dim=1)
    else:
        shifted = torch.cat((torch.zeros_like(x[:, :1]), x[:, :-1]), dim=1)

    delta = shifted - x
    if output_cache:
        return delta, x[:, -1]
    return delta


def _sqrelu_torch(x: torch.Tensor) -> torch.Tensor:
    r = F.relu(x.float())
    return (r * r).to(dtype=x.dtype)


_FORCE_LOCAL_HELPERS = (
    os.environ.get("RWKV7_LOCAL_HELPERS", "0") == "1" or not _HAS_FLA
)
if _HAS_FLA and not _FORCE_LOCAL_HELPERS:
    token_shift = _fla_token_shift
    sqrelu = _FLA_ACT2FN["sqrelu"]
else:
    token_shift = _token_shift_torch
    sqrelu = _sqrelu_torch
