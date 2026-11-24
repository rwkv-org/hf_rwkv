# RWKV7 native → HuggingFace converter

Convert a native RWKV-7 `.pth` checkpoint into a self-contained directory that
loads with `transformers` + `trust_remote_code=True`.

```bash
python convert.py \
  --source /mnt/data4/jl/models/rwkv-g1m-3b-temp-6056-ctx65536.pth \
  --output-dir /mnt/data4/jl/models/rwkv-g1m-3b-hf \
  --overwrite
```

The vocabulary is picked automatically from `hf/rwkv_vocab*.txt`; use
`--vocab-file` when the directory holds more than one candidate and
`--chat-template` to embed a different Jinja template.

## Chat template

`hf/chat_template.jinja` (Qwen3.8-Flash-Next ChatML) is copied into every
converted directory.

```
<|im_start|>system\n{system}<|im_end|>\n
<|im_start|>user\n{question}<|im_end|>\n
<|im_start|>assistant\n<think>\n\n</think>\n\n{answer}<|im_end|>\n
```

| argument | effect |
|---|---|
| `enable_thinking=True/False` | off ⇒ the assistant turn opens with an **empty** `<think>\n\n</think>\n\n` block |
| `reasoning_effort="xhigh"/"medium"/"low"` | prepends one instruction line to the system turn; default `xhigh` |

An assistant message may carry `reasoning_content`, rendered inside `<think>`.
Tool calls use XML (`<tool_call><function=name><parameter=k>v</parameter></function></tool_call>`),
tool results come back in a `<tool_response>` block.

Required control tokens — RWKV-7 G1 vocabularies have these slots free, so no
`resize_token_embeddings` is needed:

| token | id |
|---|---:|
| `<|endoftext|>` | 0 (pad) |
| `<|im_start|>` | 65530 |
| `<|im_end|>` | 65531 (eos) |
| `<think>` | 65532 |
| `</think>` | 65533 |

`<tool_call>`, `<function=`, `<parameter=` are ordinary text. The template also
mentions `<|vision_start|>` / `<|image_pad|>` / `<|vision_end|>` for multimodal
turns; a text-only vocabulary lacks them and `validate_chat_template()` skips
those tokens.

## Tokenizer behaviour

* 65,530 vocabulary entries plus id-0 `<|endoftext|>`
* byte-level and greedy: every string tokenizes, no OOV
* BOS/EOS = 0, `add_bos_token=True`, no PAD/UNK token

`apply_chat_template(tokenize=True)` does **not** insert BOS — prepend
`bos_token_id` (0) yourself for SFT or generation. The generated
`generation_config.json` stops on `<|endoftext|>`, `<|im_end|>` and `✿`, so pass
`tokenizer=tokenizer` to `generate()`.

## Bring up a new checkpoint (3B example)

```bash
# 0. huggingface.co is unreachable from this box; use the mirror
cd /mnt/data4/jl/models
curl -L --retry 5 -C - -o rwkv-g1m-3b-temp-6056-ctx65536.pth \
  https://hf-mirror.com/BlinkDL/temp-latest-training-models/resolve/main/rwkv-g1m-3b-temp-6056-ctx65536.pth

# 1. convert (bf16, 2 shards)
cd /home/BOPENG/jl/hf_rwkv
python convert.py \
  --source /mnt/data4/jl/models/rwkv-g1m-3b-temp-6056-ctx65536.pth \
  --output-dir /mnt/data4/jl/models/rwkv-g1m-3b-hf --overwrite

# 2. load check
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
p='/mnt/data4/jl/models/rwkv-g1m-3b-hf'
t=AutoTokenizer.from_pretrained(p, trust_remote_code=True)
m=AutoModelForCausalLM.from_pretrained(p, trust_remote_code=True, dtype='bfloat16')
print(len(t), m.config.vocab_size)
print(t.apply_chat_template([{'role':'user','content':'hi'}], tokenize=False, add_generation_prompt=True))"
```

The 3B checkpoint is `hidden 2560 / 32 layers / native ctx 65536`; a 32k pack
with gradient checkpointing takes ~26 GiB per H100.

## Kernel backend

All FLA imports live in `rwkv7_kernels.py`; replace `addcmul_rwkv7`,
`k_update_rwkv7`, `wkv_chunk`, `wkv_recurrent`, `output_correction` to plug in a
different backend. `RWKV7_KERNEL_BACKEND` selects `auto` (default), `fla` or
`torch`.

## Packed training

Pass cumulative lengths via `cu_seqlens` (batch size 1) with ordinary `labels`;
the last token of each segment is masked so it cannot predict the next segment.
Use bf16 — the FLA chunk backward kernel has a Blackwell shared-memory
limitation in fp32.

```python
out = model(input_ids, cu_seqlens=cu_seqlens, labels=labels)
out.loss.backward()
```

## Verify a conversion

```bash
PYTHONPATH=/path/to/fla python tests/verify_conversion.py \
  --native models/rwkv7-g1g-1.5b-20260526-ctx8192.pth \
  --hf-dir models/rwkv7-g1g-1.5b-hf
```

Checks every tensor against the native checkpoint, the tokenizer round-trip, HF
logits against a pure-PyTorch reference, and incremental decoding with the
RWKV7 cache against a full forward pass. `--no-numerical` skips the numerical
part.
