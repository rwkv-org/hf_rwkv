# RWKV7 native-to-HuggingFace converter

Convert the native RWKV7 `.pth` checkpoint into a self-contained directory that
works with `transformers` + `trust_remote_code=True`.

```bash
python convert.py \
  --source models/rwkv7-g1g-1.5b-20260526-ctx8192.pth \
  --output-dir models/rwkv7-g1g-1.5b-hf \
  --overwrite
```

The converter picks the vocabulary automatically from `hf/rwkv_vocab*.txt`.
Use `--vocab-file` if the directory contains more than one candidate, and
`--chat-template` to embed a custom Jinja template (the template is validated
against the vocabulary, so ChatML control tokens are not silently dropped).

Tokenizer behavior follows the upstream `transformers-rwkv` RWKV tokenizer:
65530 vocabulary entries including id-0 `<|endoftext|>`, BOS/EOS = 0,
`add_bos_token=True`, and no PAD/UNK token. The default chat template is the
upstream `✿` template (`hf/chat_template.jinja`); it supports the `bot`,
`assistant`, and `function_calling` styles plus `fake_cot_mode` and tools.
That single file is also copied into every converted directory. Use
`--chat-template /path/to/template.jinja` for an external G1 text or ChatML
template; it is validated against the vocabulary.

`apply_chat_template(tokenize=True)` follows the Transformers convention and
does not insert BOS itself. For SFT/generation, prepend
`tokenizer.bos_token_id` (0) or tokenize the rendered string with
`add_special_tokens=True`. The generated `generation_config.json` stops on
`<|endoftext|>`, `✿`, `\nUser:`, and `\n### User`. Because the generation
config contains `stop_strings`, pass `tokenizer=tokenizer` to `generate()`.

## Cache protocol

`modeling_rwkv7.py` uses the modern transformers cache protocol:
`RWKV7Cache(Cache)` is a container of `RWKV7CacheLayer(CacheLayerMixin)`
objects, and generation passes the cache through `past_key_values` (the same
convention used by FLA, Qwen3-Next and Kimi Linear/K3). `cache_params` is kept
only as a backwards-compatible alias.

## Kernel backend

All FLA imports are isolated in `rwkv7_kernels.py`. To switch to a CUDA or
TileLang implementation, replace these five functions:

```
addcmul_rwkv7
k_update_rwkv7
wkv_chunk
wkv_recurrent
output_correction
```

Backend selection is controlled by `RWKV7_KERNEL_BACKEND`:

- `auto` (default): use FLA kernels if importable, otherwise pure PyTorch;
- `fla`: require FLA;
- `torch`: pure PyTorch only, no FLA import.

`token_shift` and `sqrelu` use FLA when the FLA backend is active. Set
`RWKV7_LOCAL_HELPERS=1` to force their local implementations. The converter
copies `rwkv7_kernels.py` into every output directory, so a replacement
backend is self-contained.

## Packed training

Pass cumulative lengths via `cu_seqlens` (batch size must be 1) and normal
`labels`; the last token of each packed segment is automatically ignored by the
loss so it does not predict the next segment:

```python
out = model(input_ids, cu_seqlens=cu_seqlens, labels=labels)
out.loss.backward()
```

Use bf16 training. The FLA chunk backward kernel currently has a Blackwell
shared-memory limitation in fp32, while bf16 backward works.

```bash
PYTHONPATH=/path/to/fla python tests/test_packing.py \
  --model models/rwkv7-g1d-0.4b-hf --dtype bfloat16 --segments 30,40,58
```

This checks packed-vs-separate logits, forward/backward, boundary masking and
packed-vs-separate gradients.

## NeMo AutoModel smoke test

`tests/automodel_sft.yaml` is a one-step SFT recipe for NeMo AutoModel. Replace
the model/tokenizer paths and run from the Automodel checkout:

```bash
uv sync --no-default-groups --python 3.12
PYTHONPATH=/path/to/fla-MA \
  .venv/bin/automodel /path/to/hf_rwkv/tests/automodel_sft.yaml --nproc-per-node 1
```

New conversions use `model_type: rwkv7_hf`, so a full FLA install no longer
shadows `modeling_rwkv7.py`. To run without FLA at all, set
`RWKV7_KERNEL_BACKEND=torch`; FLA is then not imported and the pure-PyTorch
reference kernels are used.

```bash
RWKV7_KERNEL_BACKEND=torch \
  .venv/bin/automodel /path/to/hf_rwkv/tests/automodel_sft.yaml --nproc-per-node 1
```

Validated on an RTX PRO 6000 with both the FLA backend and the pure-PyTorch
backend:

- DDP and FSDP2 one-step SFT, loss/grad finite and validation loss finite;
- `use_cache=True` and generation under Transformers 5.18;
- checkpoint save + consolidated HF safetensors reload;
- THD packed training (`packed_sequence_size > 0`, `seq_lens`/`seq_lens_padded`).

## LoRA + FSDP2 + SwanLab

`tests/automodel_lora_sft.yaml` is a LoRA recipe for two GPUs. It patches
`r/k/v/o_proj` and the FFN `key/value` linears:

```bash
uv pip install "nemo-automodel[swanlab]"
PYTHONPATH=/path/to/fla-MA \
  .venv/bin/automodel tests/automodel_lora_sft.yaml --nproc-per-node 2
```

Measured on 2x RTX PRO 6000, sequence length 512, global batch 2, LoRA rank 16,
bf16, FLA kernels:

| model | trainable | mem/GPU | steady throughput |
|---|---:|---:|---:|
| 0.4b | 7.08M (1.55%) | 1.31 GiB | ~1.4-1.5k tok/s |
| 1.5b | 14.16M (0.92%) | 3.07 GiB | ~1.4k tok/s |

The first step takes 29 s (0.4b) / 141 s (1.5b) because of Triton autotuning;
later steps are sub-second. Throughput is total across both GPUs. The mock
dataset uses random tokens, so its loss is not meaningful.

## Verify a conversion

```bash
PYTHONPATH=/path/to/fla python tests/verify_conversion.py \
  --native models/rwkv7-g1g-1.5b-20260526-ctx8192.pth \
  --hf-dir models/rwkv7-g1g-1.5b-hf
```

The script checks every converted tensor against the native checkpoint, checks
the tokenizer round-trip, compares the HF logits with a pure-PyTorch RWKV7
reference, and verifies that incremental decoding with the RWKV7 cache matches
a full-sequence forward pass. Add `--no-numerical` for a fast weight/tokenizer
check.
