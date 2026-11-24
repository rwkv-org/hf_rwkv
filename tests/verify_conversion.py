#!/usr/bin/env python3
"""Verify a converted RWKV7 checkpoint against the native .pth file.

What is checked
---------------
1. Every converted safetensors tensor is bit-identical to the expected native
   tensor (including the transposes/reshapes used by the LoRA parameters).
2. The HF tokenizer round-trips text like the reference RWKV byte tokenizer.
3. The HF model reproduces a pure-PyTorch implementation of the native RWKV7
   equations (logits, float32).
4. The incremental RWKV7 cache matches a full-sequence forward pass.

Usage
-----
    PYTHONPATH=/path/to/fla python tests/verify_conversion.py \
        --native models/rwkv7-g1d-0.4b-...-ctx8192.pth \
        --hf-dir /tmp/rwkv7-g1d-hf

Numerical checks need the same ``fla`` package that the converted
``modeling_rwkv7.py`` needs.  Use ``--no-numerical`` for a fast weight-only
check.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from convert import load_native_rwkv7_state_dict  # noqa: E402


# ---------------------------------------------------------------------------
# Native equations, written independently of modeling_rwkv7.py
# ---------------------------------------------------------------------------


def _layer_norm(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float
) -> torch.Tensor:
    return F.layer_norm(x, (x.shape[-1],), weight, bias, eps)


def _token_shift(x: torch.Tensor) -> torch.Tensor:
    return torch.cat((torch.zeros_like(x[:, :1]), x[:, :-1]), dim=1)


def _wkv_recurrence(
    r: torch.Tensor,
    w: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kk: torch.Tensor,
    a: torch.Tensor,
) -> torch.Tensor:
    """Native RWKV7 WKV recurrence, one token at a time.

    State layout: S[b, h, key_dim, value_dim].  The update is

        sa_i    = sum_j S[j, i] * kk_j
        S[j, i] = exp(w_j) * S[j, i] - (kk_j a_j) * sa_i + k_j v_i
        y_i     = sum_j r_j * S[j, i]
    """
    batch, time, heads, head_dim = r.shape
    value_dim = v.shape[-1]
    state = torch.zeros(
        batch, heads, head_dim, value_dim, device=r.device, dtype=torch.float32
    )
    outputs = []
    for t in range(time):
        kk_t = kk[:, t].float()
        a_t = a[:, t].float()
        w_t = w[:, t].float()
        k_t = k[:, t].float()
        v_t = v[:, t].float()
        r_t = r[:, t].float()
        sa = torch.einsum("bhji,bhj->bhi", state, kk_t)
        state = (
            state * torch.exp(w_t).unsqueeze(-1)
            - (kk_t * a_t).unsqueeze(-1) * sa.unsqueeze(-2)
            + k_t.unsqueeze(-1) * v_t.unsqueeze(-2)
        )
        outputs.append(torch.einsum("bhji,bhj->bhi", state, r_t))
    return torch.stack(outputs, dim=1)


def native_forward(
    state_dict: Mapping[str, torch.Tensor],
    input_ids: torch.Tensor,
    config: Mapping[str, Any],
) -> torch.Tensor:
    """Run the native RWKV7 equations in float32 and return logits."""
    device = input_ids.device
    hidden_size = int(config["hidden_size"])
    num_heads = int(config["num_heads"])
    head_dim = int(config["head_dim"])
    value_dim = int(config["value_dim"][0])
    head_v_dim = value_dim // num_heads
    num_layers = int(config["num_hidden_layers"])
    norm_eps = float(config["norm_eps"])
    group_eps = head_dim * norm_eps

    def get(name: str) -> torch.Tensor:
        tensor = state_dict[name]
        if tensor.device != device:
            tensor = tensor.to(device)
        return tensor.float()

    x = F.embedding(input_ids, get("emb.weight"))
    v_first: torch.Tensor | None = None

    for layer_idx in range(num_layers):
        prefix = f"blocks.{layer_idx}"
        if layer_idx == 0:
            x = _layer_norm(
                x, get(f"{prefix}.ln0.weight"), get(f"{prefix}.ln0.bias"), norm_eps
            )
        residual = x
        h = _layer_norm(
            x, get(f"{prefix}.ln1.weight"), get(f"{prefix}.ln1.bias"), norm_eps
        )
        delta = _token_shift(h) - h
        xr = h + delta * get(f"{prefix}.att.x_r").view(1, 1, -1)
        xw = h + delta * get(f"{prefix}.att.x_w").view(1, 1, -1)
        xk = h + delta * get(f"{prefix}.att.x_k").view(1, 1, -1)
        xv = h + delta * get(f"{prefix}.att.x_v").view(1, 1, -1)
        xa = h + delta * get(f"{prefix}.att.x_a").view(1, 1, -1)
        xg = h + delta * get(f"{prefix}.att.x_g").view(1, 1, -1)

        r = F.linear(xr, get(f"{prefix}.att.receptance.weight"))
        w_raw = get(f"{prefix}.att.w0").view(1, 1, -1) + torch.tanh(
            xw @ get(f"{prefix}.att.w1")
        ) @ get(f"{prefix}.att.w2")
        # The native CUDA kernel uses exp(-0.6065306597 * sigmoid(w)).
        w = -0.6065306597126334 * torch.sigmoid(w_raw)
        k = F.linear(xk, get(f"{prefix}.att.key.weight"))
        v = F.linear(xv, get(f"{prefix}.att.value.weight"))
        if layer_idx == 0:
            v_first = v
        else:
            assert v_first is not None
            v = v + (v_first - v) * torch.sigmoid(
                get(f"{prefix}.att.v0").view(1, 1, -1)
                + (xv @ get(f"{prefix}.att.v1")) @ get(f"{prefix}.att.v2")
            )
        a = torch.sigmoid(
            get(f"{prefix}.att.a0").view(1, 1, -1)
            + (xa @ get(f"{prefix}.att.a1")) @ get(f"{prefix}.att.a2")
        )
        g = torch.sigmoid(xg @ get(f"{prefix}.att.g1")) @ get(f"{prefix}.att.g2")

        batch, time, _ = x.shape
        kk = F.normalize(
            (k * get(f"{prefix}.att.k_k").view(1, 1, -1)).view(
                batch, time, num_heads, head_dim
            ),
            dim=-1,
            p=2.0,
        ).view(batch, time, -1)
        k = k * (1 + (a - 1) * get(f"{prefix}.att.k_a").view(1, 1, -1))

        r_h = r.view(batch, time, num_heads, head_dim)
        w_h = w.view(batch, time, num_heads, head_dim)
        k_h = k.view(batch, time, num_heads, head_dim)
        v_h = v.view(batch, time, num_heads, head_v_dim)
        kk_h = kk.view(batch, time, num_heads, head_dim)
        a_h = a.view(batch, time, num_heads, head_dim)

        o = _wkv_recurrence(r_h, w_h, k_h, v_h, kk_h, a_h).view(batch, time, -1)
        o = F.group_norm(
            o.view(batch * time, -1),
            num_heads,
            get(f"{prefix}.att.ln_x.weight"),
            get(f"{prefix}.att.ln_x.bias"),
            eps=group_eps,
        ).view(batch, time, -1)
        correction = (
            (
                r_h
                * k_h
                * get(f"{prefix}.att.r_k").unsqueeze(0)
            ).sum(dim=-1, keepdim=True)
            * v_h
        ).view(batch, time, -1)
        x = residual + F.linear(
            (o + correction) * g, get(f"{prefix}.att.output.weight")
        )

        residual = x
        h = _layer_norm(
            x, get(f"{prefix}.ln2.weight"), get(f"{prefix}.ln2.bias"), norm_eps
        )
        delta = _token_shift(h) - h
        ffn_in = h + delta * get(f"{prefix}.ffn.x_k").view(1, 1, -1)
        ffn_hidden = torch.relu(F.linear(ffn_in, get(f"{prefix}.ffn.key.weight"))) ** 2
        x = residual + F.linear(ffn_hidden, get(f"{prefix}.ffn.value.weight"))

    x = _layer_norm(
        x,
        get("ln_out.weight"),
        get("ln_out.bias") if "ln_out.bias" in state_dict else torch.zeros(hidden_size, device=device),
        norm_eps,
    )
    return F.linear(x, get("head.weight"))


# ---------------------------------------------------------------------------
# Conversion checks
# ---------------------------------------------------------------------------


def load_safetensors(hf_dir: Path) -> dict[str, torch.Tensor]:
    from safetensors.torch import load_file

    shards = sorted(hf_dir.glob("*.safetensors"))
    if not shards:
        raise SystemExit(f"No safetensors files found in {hf_dir}")
    merged: dict[str, torch.Tensor] = {}
    for shard in shards:
        for name, tensor in load_file(shard).items():
            if name in merged:
                raise SystemExit(f"Tensor {name} appears in more than one shard")
            merged[name] = tensor
    return merged


def check_weights(
    native: Mapping[str, torch.Tensor],
    converted: Mapping[str, torch.Tensor],
    config: Mapping[str, Any],
) -> None:
    from convert import map_native_state_dict

    expected = map_native_state_dict(native)
    missing = sorted(set(expected) - set(converted))
    extra = sorted(set(converted) - set(expected))
    if missing or extra:
        raise SystemExit(
            f"State dict key mismatch: missing={missing[:10]}, extra={extra[:10]}"
        )
    for name, expected_tensor in expected.items():
        actual = converted[name]
        if actual.dtype != expected_tensor.dtype:
            raise SystemExit(
                f"{name}: dtype {actual.dtype} != native {expected_tensor.dtype}"
            )
        if actual.shape != expected_tensor.shape:
            raise SystemExit(
                f"{name}: shape {tuple(actual.shape)} != native "
                f"{tuple(expected_tensor.shape)}"
            )
        if not torch.equal(actual, expected_tensor):
            raise SystemExit(f"{name}: values differ from the native checkpoint")
    print(f"[ok] weights: {len(converted)} tensors match the native checkpoint")


def check_tokenizer(hf_dir: Path, config: Mapping[str, Any]) -> None:
    from transformers import AutoTokenizer

    from hf.hf_rwkv_tokenizer import RWKVByteTokenizer

    vocab_file = hf_dir / "rwkv_vocab.txt"
    if not vocab_file.is_file():
        candidates = sorted(hf_dir.glob("rwkv_vocab*.txt"))
        if len(candidates) != 1:
            raise SystemExit(f"Cannot determine the vocabulary file in {hf_dir}")
        vocab_file = candidates[0]
    reference = RWKVByteTokenizer(vocab_file)
    tokenizer = AutoTokenizer.from_pretrained(str(hf_dir), trust_remote_code=True)

    samples = [
        "",
        "Hello world",
        "你好，世界",
        "Hello 你好 🚀",
        "Line1\n\nLine2",
        "The quick brown fox jumps over the lazy dog. " * 8,
    ]
    for text in samples:
        expected_ids = reference.encode(text)
        actual_ids = tokenizer.encode(text, add_special_tokens=False)
        if actual_ids != expected_ids:
            raise SystemExit(f"Tokenizer mismatch for {text!r}")
        if tokenizer.decode(actual_ids) != text:
            raise SystemExit(f"Tokenizer decode mismatch for {text!r}")

    special_ids = tokenizer.encode("<|endoftext|>", add_special_tokens=False)
    if special_ids != [config["eos_token_id"]]:
        raise SystemExit(
            f"<|endoftext|> maps to {special_ids}, expected "
            f"[{config['eos_token_id']}]"
        )
    expected_specials = (config["bos_token_id"], config["eos_token_id"], None, None)
    actual_specials = (
        tokenizer.bos_token_id,
        tokenizer.eos_token_id,
        tokenizer.pad_token_id,
        tokenizer.unk_token_id,
    )
    if actual_specials != expected_specials:
        raise SystemExit(
            f"Tokenizer special ids {actual_specials} != {expected_specials}"
        )
    if tokenizer.encode("hello")[0] != config["bos_token_id"]:
        raise SystemExit("Tokenizer does not prepend the RWKV BOS token by default.")
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": "A  \r\n\r\nB\t  \n\n"}],
        tokenize=False,
        add_generation_prompt=True,
    )
    if "Bot✿" in tokenizer.chat_template or "rwkv_prompt_template" in tokenizer.chat_template:
        if rendered != "User✿A\nB✿\nBot✿<think":
            raise SystemExit(f"Unexpected official RWKV ✿ template output: {rendered!r}")
    elif not rendered.startswith("User:"):
        raise SystemExit(f"Unexpected G1 text template output: {rendered!r}")

    generation_config = json.loads(
        (hf_dir / "generation_config.json").read_text(encoding="utf-8")
    )
    generation_eos = generation_config["eos_token_id"]
    if isinstance(generation_eos, int):
        generation_eos = [generation_eos]
    if config["eos_token_id"] not in generation_eos:
        raise SystemExit(
            "generation_config.json does not include the model's eos_token_id: "
            f"{generation_eos} vs {config['eos_token_id']}"
        )
    template_name = (
        "official ✿"
        if "Bot✿" in tokenizer.chat_template or "rwkv_prompt_template" in tokenizer.chat_template
        else "G1 text"
    )
    print(
        f"[ok] tokenizer: {tokenizer.vocab_size} tokens (incl. id-0 BOS/EOS), "
        f"{template_name} template, round-trip matches"
    )


def check_numerics(
    native: Mapping[str, torch.Tensor],
    hf_dir: Path,
    config: Mapping[str, Any],
    device: torch.device,
) -> None:
    from transformers import AutoModelForCausalLM
    from transformers.cache_utils import Cache, CacheLayerMixin

    if len(set(config["value_dim"])) != 1:
        raise SystemExit(
            "This reference implementation only covers checkpoints with a "
            "uniform value_dim across layers."
        )
    torch.manual_seed(0)
    input_ids = torch.randint(
        1,
        int(config["vocab_size"]) - 1,
        (1, 8),
        device=device,
    )
    reference = native_forward(native, input_ids, config)
    model = AutoModelForCausalLM.from_pretrained(
        str(hf_dir), trust_remote_code=True, dtype=torch.float32
    ).to(device)
    model.eval()
    with torch.no_grad():
        actual = model(input_ids, use_cache=False).logits.float()

    max_abs = (reference - actual).abs().max().item()
    cosine = F.cosine_similarity(
        reference.reshape(-1), actual.reshape(-1), dim=0
    ).item()
    if max_abs > 5e-3 or cosine < 0.9999:
        raise SystemExit(
            f"HF logits differ from the native reference: "
            f"max_abs={max_abs:.3g}, cosine={cosine:.6f}"
        )

    # Incremental cache must agree with a full forward pass.
    split = 4
    with torch.no_grad():
        full = model(input_ids, use_cache=False).logits.float()
        outputs = model(input_ids[:, :split], use_cache=True)
        cache = outputs.past_key_values
        if not isinstance(cache, Cache):
            raise SystemExit(f"past_key_values is not a transformers Cache: {type(cache)}")
        if not all(isinstance(layer, CacheLayerMixin) for layer in cache.layers):
            raise SystemExit("RWKV7 cache layers do not implement CacheLayerMixin")
        if cache.get_seq_length() != split:
            raise SystemExit(
                f"Cache seq length {cache.get_seq_length()} != prefill length {split}"
            )
        if list(cache.is_sliding) != [False] * len(cache.layers):
            raise SystemExit("RWKV7 cache layers must be non-sliding")
        incremental = []
        for index in range(split, input_ids.shape[1]):
            outputs = model(
                input_ids[:, index : index + 1],
                past_key_values=outputs.past_key_values,
                use_cache=True,
            )
            incremental.append(outputs.logits.float()[:, -1])
        incremental_logits = torch.stack(incremental, dim=1)
    cache_max_abs = (full[:, -len(incremental) :] - incremental_logits).abs().max().item()
    if cache_max_abs > 1e-3:
        raise SystemExit(
            f"Incremental cache differs from full forward: {cache_max_abs:.3g}"
        )

    print(
        f"[ok] numerics: max_abs={max_abs:.3g}, cosine={cosine:.6f}, "
        f"cache_max_abs={cache_max_abs:.3g}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", required=True, type=Path)
    parser.add_argument("--hf-dir", required=True, type=Path)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for numerical checks (default: cuda when available).",
    )
    parser.add_argument(
        "--no-numerical",
        action="store_true",
        help="Skip model loading and only verify weights/tokenizer.",
    )
    args = parser.parse_args()

    if not (args.hf_dir / "rwkv7_kernels.py").is_file():
        raise SystemExit(
            f"Missing kernel backend: {args.hf_dir / 'rwkv7_kernels.py'}"
        )
    native = load_native_rwkv7_state_dict(args.native)
    converted = load_safetensors(args.hf_dir)
    config = json.loads((args.hf_dir / "config.json").read_text(encoding="utf-8"))

    check_weights(native, converted, config)
    check_tokenizer(args.hf_dir, config)
    if not args.no_numerical:
        device = torch.device(args.device)
        if device.type != "cuda":
            raise SystemExit(
                "Numerical checks require a CUDA device because modeling_rwkv7 "
                "uses fla Triton kernels. Use --no-numerical for CPU-only checks."
            )
        check_numerics(native, args.hf_dir, config, device)

    print("All checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
