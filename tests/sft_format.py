#!/usr/bin/env python
"""One frozen RWKV-7 (G1) chat format for SFT, evaluation and serving.

Why this file exists
--------------------
The RWKV vocabulary is a *byte-level greedy* vocabulary: every byte has an id,
so any string tokenizes and ``apply_chat_template`` never fails.  The chat
format is therefore a free choice, not something the tokenizer enforces.  Pick
one, pin it in code, and use the same bytes for training and inference.

The format (upstream ``✿`` / "bot" style, the G1 post-training format)::

    System✿{system}✿
    User✿{user}✿
    Bot✿{answer}✿

Two details the stock Jinja template hides:

* ``✿`` (U+273F, token id 10060) is both the turn separator and the stop
  token.  It is a single token, so a turn boundary costs 2 tokens
  (``User`` + ``✿``) instead of ChatML's 7 per marker.
* ``apply_chat_template(..., add_generation_prompt=True, fake_cot_mode=True)``
  ends with ``...Bot✿<think></think`` -- **without** the closing ``>``.  The
  base model emits that ``>`` as its first token and then the answer, so the
  supervised span starts with ``>`` and the prompt stays byte-identical to the
  stock template.  Strip one leading ``>`` when serving.  Dropping the
  ``<think`` block altogether (see ``think="none"``) is the clean alternative,
  but then evaluation must not use the stock template.

Usage::

    from sft_format import encode_sft, chat_prompt

    ids, labels = encode_sft(tok, [{"role": "user", "content": "1+1=?"},
                                   {"role": "assistant", "content": "2"}])
    # ids    = [0, User, ✿, 1, +, 1, =, ?, ✿, \\n, Bot, ✿, <think>, </think>, >, 2, ✿]
    # labels = [-100 ... -100, >, 2, 10060]  (answer + its closing ✿)

    prompt = chat_prompt(tok, messages)     # inference, no trailing newline

Run ``python tests/sft_format.py`` for a self-check.
"""

from __future__ import annotations

import argparse
import sys
from typing import Iterable, Sequence

SEP = "\u273f"  # ✿  id 10060, separator + stop token
CLOSED_THINK = "<think></think"  # no trailing ">": the model emits it
OPEN_THINK = "<think>"
CLOSE_THINK = "</think>"
STOP_STRINGS = [SEP, "\nUser:", "\n### User"]
DEFAULT_SYSTEM = "You are a helpful assistant."


IM_START = "<|im_start|>"
IM_END = "<|im_end|>"


def _clean(text: str, sanitize: bool, markers: tuple[str, ...] = (SEP,), strip: bool = True) -> str:
    text = str(text).replace("\r\n", "\n").replace("\r", "\n")
    if strip:
        text = text.strip()
    for marker in markers:
        if marker in text:
            if not sanitize:
                raise ValueError(
                    f"content contains the control marker {marker!r}; it would "
                    "cut the turn in two.  Pass sanitize=True to strip it."
                )
            text = text.replace(marker, "")
    return text


def added_token_markers(tokenizer) -> tuple[str, ...]:
    """Every added-token string, longest first (for sanitizing content)."""
    decoder = getattr(tokenizer, "added_tokens_decoder", None) or {}
    strings = {
        str(token.content)
        for token in decoder.values()
        if str(token.content).strip()
    }
    return tuple(sorted(strings, key=len, reverse=True))


def build_pieces(
    messages: Sequence[dict],
    *,
    system: str | None = None,
    think: str = "closed",
    sanitize: bool = True,
    style: str = "bot",
    markers: tuple[str, ...] | None = None,
) -> tuple[str, list[tuple[int, int]]]:
    """Render a conversation and return ``(text, supervised_char_spans)``.

    ``style="bot"`` is the native G1 format (``System✿…✿`` / ``User✿…✿`` /
    ``Bot✿…✿``).  ``think`` then selects how the assistant turn is opened:

    * ``"closed"`` -- prompt ends with ``<think></think`` (no reasoning, the
      default; matches the base model's "fake cot" mode and the stock
      template);
    * ``"none"``   -- prompt ends with ``Bot✿`` (the base model also answers
      fine without a think block, probe it before relying on it);
    * ``"open"``   -- prompt ends with ``<think``; the message may carry a
      ``reasoning`` field, supervised as ``>reasoning</think>answer✿``.

    ``style="chatml"`` renders Qwen-compatible ChatML (see
    Qwen ChatML template); ``think`` selects none / closed / open as well
    (``closed`` writes the empty Qwen3.8 reasoning block as prompt bytes and
    ``open`` puts ``reasoning`` inside it).  The old sentence said ``think``
    was ignored because the whole
    answer lives in ``content``.
    """
    if style not in ("bot", "chatml"):
        raise ValueError(f"unknown style: {style!r}")
    if style == "bot" and think not in ("closed", "none", "open"):
        raise ValueError(f"unknown think mode: {think!r}")

    chunks: list[str] = []
    spans: list[tuple[int, int]] = []
    pos = 0

    def add(text: str, train: bool) -> None:
        nonlocal pos
        if not text:
            return
        chunks.append(text)
        if train:
            spans.append((pos, pos + len(text)))
        pos += len(text)

    if system is None:
        system = next(
            (m.get("content") for m in messages if m.get("role") == "system"), None
        )

    if style == "chatml":
        # ChatML is self-delimiting, so content is used verbatim (Qwen does not
        # strip either).  The header and the newline after <|im_end|> are
        # prompt bytes: generation stops at <|im_end|>, so the model never has
        # to produce the trailing newline.
        forbidden = markers or (IM_START, IM_END)
        clean = lambda t: _clean(t, sanitize, forbidden, strip=False)  # noqa: E731
        if system:
            add(f"{IM_START}system\n{clean(system)}{IM_END}\n", False)
        for message in messages:
            role = message.get("role")
            if role == "system":
                continue
            if role == "user":
                add(f"{IM_START}user\n{clean(message.get('content', ''))}{IM_END}\n", False)
            elif role == "assistant":
                add(f"{IM_START}assistant\n", False)
                content = clean(message.get("content", ""))
                if think == "closed":
                    # Qwen3.8 non-thinking prompt: an empty reasoning block the
                    # model has to close immediately.  It is prompt bytes, so
                    # only the answer is supervised.
                    add("<think>\n\n</think>\n\n", False)
                    add(f"{content}{IM_END}", True)
                elif think == "open":
                    reasoning = clean(message.get("reasoning") or "")
                    add("<think>\n", False)
                    add(f"{reasoning}\n</think>\n\n{content}{IM_END}", True)
                elif think == "auto":
                    # tags only when there is something to put in them: a row
                    # without reasoning must not carry an empty block
                    reasoning = clean(message.get("reasoning") or "").strip()
                    if reasoning:
                        add("<think>\n", False)
                        add(f"{reasoning}\n</think>\n\n{content}{IM_END}", True)
                    else:
                        add(f"{content}{IM_END}", True)
                else:
                    add(f"{content}{IM_END}", True)
                add("\n", False)
            elif role == "tool":
                # Qwen renders tool output as a user turn wrapped in
                # <tool_response>; it is prompt, never supervised.
                body = f"<tool_response>\n{clean(message.get('content', ''))}\n</tool_response>"
                add(f"{IM_START}user\n{body}{IM_END}\n", False)
            else:
                raise ValueError(f"unsupported role: {role!r}")
        return "".join(chunks), spans  # template keeps the final newline

    if system:
        add(f"System{SEP}{_clean(system, sanitize)}{SEP}\n", False)

    for message in messages:
        role = message.get("role")
        if role == "system":
            continue
        if role == "user":
            add(f"User{SEP}{_clean(message.get('content', ''), sanitize)}{SEP}\n", False)
        elif role == "assistant":
            content = _clean(message.get("content", ""), sanitize)
            if think == "closed":
                # the model's very first generated token is ">", closing
                # "</think"; it is supervised together with the answer
                add(f"Bot{SEP}{CLOSED_THINK}", False)
                add(f">{content}{SEP}", True)
            elif think == "open":
                reasoning = _clean(message.get("reasoning") or "", sanitize)
                add(f"Bot{SEP}{OPEN_THINK}", False)
                add(f">{reasoning}{CLOSE_THINK}{content}{SEP}", True)
            else:
                add(f"Bot{SEP}", False)
                add(f"{content}{SEP}", True)
            add("\n", False)
        else:
            raise ValueError(f"unsupported role: {role!r}")

    return "".join(chunks).rstrip("\n"), spans


def _token_byte_lengths(tokenizer, ids: Sequence[int]) -> list[int]:
    """Byte length of every token in the original string.

    Added special tokens (``<|im_start|>`` …) are not in the base byte
    vocabulary, so they are measured through their string form.
    """
    table = getattr(tokenizer, "_idx2token", None)
    if table is None:
        raise TypeError(
            "tokenizer does not expose _idx2token; pass a converted hf_rwkv "
            "tokenizer (hf_rwkv_tokenizer.RwkvTokenizer)"
        )
    lengths: list[int] = []
    for token_id in ids:
        piece = table.get(token_id)
        if piece is None:
            token = tokenizer.convert_ids_to_tokens(token_id)
            if isinstance(token, bytes):
                piece = token
            else:
                piece = str(token).encode("utf-8", errors="surrogateescape")
        lengths.append(len(piece))
    return lengths


def encode_sft(
    tokenizer,
    messages: Sequence[dict],
    *,
    system: str | None = None,
    think: str = "closed",
    sanitize: bool = True,
    add_bos: bool = True,
    strict: bool = False,
    style: str = "bot",
    markers: tuple[str, ...] | None = None,
) -> tuple[list[int], list[int]]:
    """Tokenize one conversation with assistant-only labels.

    Returns ``(input_ids, labels)``.  ``labels`` is ``-100`` outside the
    assistant turns and covers the answer text plus its closing ``✿``, so the
    model learns to end its turn.  The prompt ``✿`` (end of the user turn) and
    the ``<think></think`` prefix stay masked.

    The vocabulary is greedy over bytes, so the last prompt byte and the first
    answer byte can share one token (``">`"`` for a code answer).  Such a token
    is assigned to the answer, which keeps the prompt a token-prefix of the
    full string -- exactly the tokens a server feeds at inference time.
    ``strict=True`` rejects those rows instead.
    """
    if markers is None and style == "chatml":
        # Anything the tokenizer matches atomically must not appear inside
        # message content, or a user can inject a control token.
        markers = added_token_markers(tokenizer) or None
    text, char_spans = build_pieces(
        messages, system=system, think=think, sanitize=sanitize, style=style,
        markers=markers,
    )
    # Token offsets are byte offsets, so convert the character spans once.
    spans = [
        (len(text[:start].encode()), len(text[:end].encode()))
        for start, end in char_spans
    ]
    body = tokenizer(text, add_special_tokens=False).input_ids
    ids = ([tokenizer.bos_token_id] if add_bos else []) + list(body)
    lengths = _token_byte_lengths(tokenizer, body)
    starts = [0] * len(body)
    _offset = 0
    for _i, _length in enumerate(lengths):
        starts[_i] = _offset
        _offset += _length

    labels = [-100] * len(ids)
    offset = 1 if add_bos else 0
    span_index = 0
    merged = 0
    for index, token_id in enumerate(body):
        start = starts[index]
        end = start + lengths[index]
        while span_index < len(spans) and spans[span_index][1] <= start:
            span_index += 1
        if span_index >= len(spans):
            break
        span_start, span_end = spans[span_index]
        if end > span_start and start < span_end:
            labels[offset + index] = token_id
            if start < span_start:
                merged += 1
    if strict and merged:
        raise ValueError(
            f"{merged} token(s) merge the end of the prompt with the start of "
            "the answer; refusing to build the mask (strict=True)"
        )
    return ids, labels


def chat_prompt(
    tokenizer, messages: Sequence[dict], *, system: str | None = None,
    think: str = "closed", sanitize: bool = True, add_bos: bool = True,
    style: str = "bot",
) -> list[int]:
    """Prompt token ids for the next assistant turn (same bytes as training).

    Rendered by appending an empty assistant turn and cutting at its first
    supervised character, so the turn header (``Bot✿<think></think`` or
    ``<|im_start|>assistant\n``) is included exactly as during training.
    """
    text, spans = build_pieces(
        list(messages) + [{"role": "assistant", "content": ""}],
        system=system, think=think, sanitize=sanitize, style=style,
    )
    prompt_text = text[: spans[-1][0]]
    body = tokenizer(prompt_text, add_special_tokens=False).input_ids
    return ([tokenizer.bos_token_id] if add_bos else []) + list(body)


def _self_check(model_dir: str) -> int:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    conversation = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "我叫小明，记住了吗？"},
        {"role": "assistant", "content": "记住了，你叫小明。"},
        {"role": "user", "content": "print('hi') 会输出什么？"},
        {"role": "assistant", "content": "输出 hi，带一个换行。"},
    ]
    text, spans = build_pieces(conversation)
    print("rendered:")
    print(text.replace(SEP, f"[{SEP}]"))
    print(f"\nsupervised spans: {spans}")

    ids, labels = encode_sft(tokenizer, conversation)
    supervised = [t for t in labels if t != -100]
    decoded = tokenizer.decode(supervised, skip_special_tokens=False)
    expected = "".join(
        f">{m['content']}{SEP}" for m in conversation if m["role"] == "assistant"
    )
    print(f"\ntokens={len(ids)} supervised={len(supervised)}")
    print(f"decoded supervised: {decoded!r}")
    print(f"expected          : {expected!r}")
    assert decoded == expected, "mask does not cover exactly the assistant turns"

    # the first supervised token must directly follow "Bot✿<think></think>"
    prefix = text[: spans[0][0]]
    assert prefix.endswith(f"Bot{SEP}{CLOSED_THINK}"), prefix
    prompt = chat_prompt(tokenizer, conversation[:-1])
    assert ids[: len(prompt)] == prompt, "prompt/answer byte mismatch"
    stock = tokenizer.apply_chat_template(
        conversation[:2],
        tokenize=False,
        add_generation_prompt=True,
        fake_cot_mode=True,
    )
    assert text[: spans[0][0]] == stock, (text[: spans[0][0]], stock)
    print("\nprompt == supervised-prefix == stock template: OK")

    # greedy tokenizer must not merge across prompt/answer boundaries
    import random

    random.seed(0)
    pieces = ["北京", "hello", "a\nb", "  ", "😀", "```py\nx=1\n```", "一二三", "\t"]
    for _ in range(200):
        msgs = [
            {"role": "user", "content": "".join(random.choices(pieces, k=3))},
            {"role": "assistant", "content": "".join(random.choices(pieces, k=3))},
        ]
        encode_sft(tokenizer, msgs)
    print("200 random conversations: masks built without errors")
    print("\nself-check OK")
    return 0


def _chatml_check(model_dir: str) -> int:
    """Check the ChatML branch against an aligned model directory."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    conversation = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "1+1=?"},
        {"role": "assistant", "content": "2"},
        {"role": "user", "content": "再说一遍"},
        {"role": "assistant", "content": "等于 2。"},
    ]
    text, spans = build_pieces(conversation, style="chatml")
    stock = tokenizer.apply_chat_template(conversation, tokenize=False)
    print("rendered:", repr(text))
    assert text == stock, ("chatml render differs from apply_chat_template", text, stock)

    ids, labels = encode_sft(tokenizer, conversation, style="chatml")
    supervised = tokenizer.decode([t for t in labels if t != -100])
    expected = "".join(
        f"{m['content']}{IM_END}" for m in conversation if m["role"] == "assistant"
    )
    print("supervised:", repr(supervised))
    assert supervised == expected, (supervised, expected)

    prompt = chat_prompt(tokenizer, conversation[:2], style="chatml")
    assert ids[: len(prompt)] == prompt, "prompt/answer byte mismatch"
    assert tokenizer.decode(prompt).endswith(f"{IM_START}assistant\n")
    print("chatml self-check OK")
    return 0


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="/home/BOPENG/jl/hf_rwkv/models/rwkv7-g1l-1.5b-hf",
        help="converted hf_rwkv model directory (tokenizer source)",
    )
    parser.add_argument(
        "--chatml",
        default=None,
        help="check the ChatML branch against a Qwen-style model directory",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.chatml:
        return _chatml_check(args.chatml)
    return _self_check(args.model)


if __name__ == "__main__":
    sys.exit(main())
