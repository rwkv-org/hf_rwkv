#!/usr/bin/env python3
"""Check packed-sequence (cu_seqlens) training for an RWKV7 HF model.

Tests:
1. packed forward logits match separate per-segment forwards;
2. training forward/backward works;
3. automatic loss masking does not cross segment boundaries;
4. gradients of the packed loss match the weighted sum of segment losses.

Run with the same FLA environment as the model:

    PYTHONPATH=/path/to/fla python tests/test_packing.py \
        --model /tmp/rwkv7-g1d-hf --dtype bfloat16
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Converted HF model directory.")
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Training dtype. float32 backward may be unsupported by the FLA chunk kernel.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--segments",
        default="7,13",
        help="Comma-separated packed segment lengths.",
    )
    parser.add_argument(
        "--no-grad-compare",
        action="store_true",
        help="Skip the packed-vs-separate gradient comparison.",
    )
    return parser.parse_args()


def cu_seqlens_from_lengths(lengths: list[int], device: torch.device) -> torch.Tensor:
    ends = torch.tensor(lengths, dtype=torch.int64, device=device).cumsum(0)
    starts = torch.zeros(1, dtype=torch.int64, device=device)
    return torch.cat((starts, ends.to(torch.int64)))


def main() -> int:
    args = parse_args()
    model_dir = Path(args.model)
    device = torch.device(args.device)
    if device.type != "cuda":
        raise SystemExit("Packing tests require CUDA because modeling_rwkv7 uses FLA Triton kernels.")
    dtype = getattr(torch, args.dtype)
    lengths = [int(x) for x in args.segments.split(",") if x.strip()]
    if not lengths or any(x <= 1 for x in lengths):
        raise SystemExit("Each packed segment must contain at least 2 tokens.")

    config = AutoConfig.from_pretrained(
        str(model_dir), trust_remote_code=True
    )
    vocab_size = int(config.vocab_size)
    total = sum(lengths)
    generation = torch.Generator(device="cpu").manual_seed(0)
    input_ids = torch.randint(
        1, vocab_size - 1, (1, total), generator=generation
    ).to(device)
    labels = torch.randint(
        1, vocab_size - 1, (1, total), generator=generation
    ).to(device)
    cu_seqlens = cu_seqlens_from_lengths(lengths, device)

    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir), trust_remote_code=True, dtype=dtype
    ).to(device)
    model.eval()

    # 1. packed forward vs separate forwards
    with torch.no_grad():
        packed = model(input_ids, cu_seqlens=cu_seqlens, use_cache=False).logits.float()
        pieces = []
        start = 0
        for length in lengths:
            pieces.append(
                model(input_ids[:, start : start + length], use_cache=False).logits.float()
            )
            start += length
        separate = torch.cat(pieces, dim=1)
    diff = (packed - separate).abs()
    cosine = F.cosine_similarity(
        packed.reshape(-1), separate.reshape(-1), dim=0
    ).item()
    print(
        f"[forward] packed vs separate: max={diff.max().item():.3g} "
        f"mean={diff.mean().item():.3g} cos={cosine:.8f}"
    )
    if cosine < 0.999:
        raise SystemExit("FAIL: packed logits do not match separate forwards")

    # Automodel THD passes ``seq_lens`` [batch, documents] instead of cu_seqlens.
    with torch.no_grad():
        seq_lens = torch.tensor([lengths], dtype=torch.long, device=device)
        via_seq_lens = model(input_ids, seq_lens=seq_lens, use_cache=False).logits.float()
    seq_diff = (via_seq_lens - packed).abs().max().item()
    if seq_diff > 1e-5:
        raise SystemExit(f"FAIL: seq_lens path differs from cu_seqlens ({seq_diff})")
    print(f"[seq_lens] matches cu_seqlens path, max={seq_diff:.3g}")

    # 2. training forward / backward
    model.train()
    model.zero_grad(set_to_none=True)
    outputs = model(
        input_ids,
        cu_seqlens=cu_seqlens,
        labels=labels,
        use_cache=False,
    )
    if outputs.loss is None or not torch.isfinite(outputs.loss):
        raise SystemExit("FAIL: packed training loss is not finite")
    try:
        outputs.loss.backward()
    except Exception as exc:  # noqa: BLE001 - report the FLA/kernel failure verbatim
        raise SystemExit(
            f"FAIL: backward failed with {type(exc).__name__}: {exc}"
        ) from exc
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    if not grads or not all(torch.isfinite(g).all() for g in grads):
        raise SystemExit("FAIL: packed backward produced missing/non-finite grads")
    print(f"[backward] ok, loss={outputs.loss.item():.6f}, {len(grads)} grad tensors")

    # 3. boundary mask: labels path must equal an explicitly masked shift_labels
    manual_shift = torch.cat(
        (labels[:, 1:], torch.full_like(labels[:, :1], -100)), dim=1
    )
    end = 0
    for length in lengths[:-1]:
        end += length
        manual_shift[:, end - 1] = -100
    with torch.no_grad():
        explicit = model(
            input_ids,
            cu_seqlens=cu_seqlens,
            shift_labels=manual_shift,
            use_cache=False,
        ).loss
    if not torch.allclose(outputs.loss, explicit, atol=1e-6, rtol=1e-6):
        raise SystemExit(
            "FAIL: automatic packed boundary masking does not match explicit "
            f"shift_labels ({outputs.loss.item()} vs {explicit.item()})"
        )
    print("[boundary] automatic mask matches explicit shift_labels")

    # 4. packed gradients vs weighted sum of separate segment losses
    if not args.no_grad_compare:
        packed_grads = {
            name: param.grad.detach().float().clone()
            for name, param in model.named_parameters()
            if param.grad is not None
        }
        model.zero_grad(set_to_none=True)
        total_valid = total - len(lengths)
        start = 0
        for length in lengths:
            segment_loss = model(
                input_ids[:, start : start + length],
                labels=labels[:, start : start + length],
                use_cache=False,
            ).loss
            (segment_loss * ((length - 1) / total_valid)).backward()
            start += length
        names = [
            "model.embeddings.weight",
            "lm_head.weight",
            "model.layers.0.attn.r_proj.weight",
            f"model.layers.{config.num_hidden_layers - 1}.attn.o_proj.weight",
        ]
        params = dict(model.named_parameters())
        min_cos = math.inf
        for name in names:
            if name not in packed_grads or params[name].grad is None:
                continue
            packed_grad = packed_grads[name]
            separate_grad = params[name].grad.detach().float()
            cos = F.cosine_similarity(
                packed_grad.reshape(-1), separate_grad.reshape(-1), dim=0
            ).item()
            min_cos = min(min_cos, cos)
            print(f"[grad] {name}: cos={cos:.8f}")
        if min_cos < 0.99:
            raise SystemExit("FAIL: packed and separate gradients disagree")

    print("PASS: packing forward/backward/loss masking work.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
