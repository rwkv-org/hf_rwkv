from __future__ import annotations

import argparse
from ast import literal_eval
import json
import re
import shutil
import sys
import tempfile
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from huggingface_hub import split_torch_state_dict_into_shards
from safetensors.torch import save_file


DEFAULT_ROOT = Path(__file__).resolve().parent
DEFAULT_HF_SOURCE = DEFAULT_ROOT / "hf"

# The tokenizer shim and the converter must agree on the file name.  We use a
# version-independent name so that a converted directory is self-contained
# regardless of which rwkv_vocab_*.txt was used as the source.
CANONICAL_VOCAB_NAME = "rwkv_vocab.txt"
VOCAB_GLOB = "rwkv_vocab*.txt"

MODEL_INDEX_NAME = "model.safetensors.index.json"
BLOCK_RE = re.compile(r"^blocks\.(\d+)\.(.+)$")
CTX_RE = re.compile(r"ctx(\d+)")
DEFAULT_MAX_POSITION_EMBEDDINGS = 8192
# FLA registers "rwkv7" globally with Transformers, which makes AutoModel load
# fla.models.rwkv7 instead of the auto_map in this directory.  A distinct
# model_type keeps this artifact on its own HF implementation.
DEFAULT_MODEL_TYPE = "rwkv7_hf"

EOS_TOKEN = "<|endoftext|>"
OPTIONAL_SPECIAL_TOKENS = (
    "<|im_start|>",
    "<|im_end|>",
    "<think>",
    "<tool_call>",
)
# Tokens whose presence is required by a chat template.  The simple RWKV
# template only needs the separator; the ChatML template needs the control
# tokens that the checkpoint must have been trained with.
# ``<think>`` / ``<tool_call>`` are ordinary text in the upstream RWKV
# templates, not vocabulary control tokens.  Only the turn separator is a
# required non-``<|...|>`` token; ``<|...|>`` tokens are discovered by regex.
CHAT_TEMPLATE_REQUIRED_TOKENS = ("✿",)
# Tokens the converter needs to look up while parsing the vocab.  Kept
# separate from template validation because <|endoftext|> is normally id 0
# and therefore absent from the base vocabulary.
VOCAB_SPECIAL_TOKENS = (EOS_TOKEN, *CHAT_TEMPLATE_REQUIRED_TOKENS)

# ChatML control tokens and the ids they take in the unused tail of a 65,536-row
# RWKV-7 vocabulary.  The converter adds any of these the vocabulary lacks, so a
# template that needs them no longer requires a separate alignment step.
CHATML_TOKEN_IDS = {
    "<|im_start|>": 65530,
    "<|im_end|>": 65531,
    "<think>": 65532,
    "</think>": 65533,
}
CHATML_ADDED_SPECIAL = ("<|im_start|>", "<|im_end|>")
CHATML_END_OF_TURN = "<|im_end|>"

DEFAULT_CHAT_TEMPLATE_PATH = DEFAULT_HF_SOURCE / "chat_template.jinja"
DEFAULT_CHAT_TEMPLATE = DEFAULT_CHAT_TEMPLATE_PATH.read_text(encoding="utf-8")

TOP_LEVEL_NAME_MAP = {
    "emb.weight": "model.embeddings.weight",
    "ln_out.weight": "model.norm.weight",
    "ln_out.bias": "model.norm.bias",
    "head.weight": "lm_head.weight",
}

BLOCK_NAME_MAP = {
    "ln1.weight": "attn_norm.weight",
    "ln1.bias": "attn_norm.bias",
    "ln2.weight": "ffn_norm.weight",
    "ln2.bias": "ffn_norm.bias",
    "att.x_r": "attn.x_r",
    "att.x_w": "attn.x_w",
    "att.x_k": "attn.x_k",
    "att.x_v": "attn.x_v",
    "att.x_a": "attn.x_a",
    "att.x_g": "attn.x_g",
    "att.k_k": "attn.k_k",
    "att.k_a": "attn.k_a",
    "att.r_k": "attn.r_k",
    "att.receptance.weight": "attn.r_proj.weight",
    "att.key.weight": "attn.k_proj.weight",
    "att.value.weight": "attn.v_proj.weight",
    "att.output.weight": "attn.o_proj.weight",
    "att.ln_x.weight": "attn.g_norm.weight",
    "att.ln_x.bias": "attn.g_norm.bias",
    "ffn.x_k": "ffn.x_k",
    "ffn.key.weight": "ffn.key.weight",
    "ffn.value.weight": "ffn.value.weight",
}

# name -> (HF name inside the block, transform)
LORA_SPECS = {
    "att.w1": ("attn.w_lora.lora.0.weight", "transpose"),
    "att.w2": ("attn.w_lora.lora.2.weight", "transpose"),
    "att.w0": ("attn.w_lora.lora.2.bias", "flatten"),
    "att.a1": ("attn.a_lora.lora.0.weight", "transpose"),
    "att.a2": ("attn.a_lora.lora.2.weight", "transpose"),
    "att.a0": ("attn.a_lora.lora.2.bias", "flatten"),
    "att.g1": ("attn.g_lora.lora.0.weight", "transpose"),
    "att.g2": ("attn.g_lora.lora.2.weight", "transpose"),
    "att.v1": ("attn.v_lora.lora.0.weight", "transpose"),
    "att.v2": ("attn.v_lora.lora.2.weight", "transpose"),
    "att.v0": ("attn.v_lora.lora.2.bias", "flatten"),
}

# These parameters exist in the native checkpoint but not in the HF model:
# layer 0 has no value-residual LoRA.
DROPPED_LAYER_ZERO_SUFFIXES = frozenset({"att.v0", "att.v1", "att.v2"})

REQUIRED_TOP_LEVEL_KEYS = ("emb.weight", "ln_out.weight", "head.weight")


class ConversionError(ValueError):
    """Raised when the native checkpoint cannot be converted safely."""


@dataclass(frozen=True)
class VocabInfo:
    path: Path
    token_ids: dict[str, int]
    min_token_id: int
    max_token_id: int
    num_tokens: int


@dataclass(frozen=True)
class ConversionSummary:
    source: Path
    output_dir: Path
    num_layers: int
    num_tensors: int
    num_shards: int
    vocab_file: str
    dtype: str


@dataclass
class ConversionArtifacts:
    state_dict: OrderedDict[str, torch.Tensor]
    model_config: dict[str, Any]
    generation_config: dict[str, Any]
    tokenizer_config: dict[str, Any]
    special_tokens_map: dict[str, Any]
    added_tokens: dict[str, int]
    chat_template: str


def load_torch_file(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - compatibility for older torch versions
        return torch.load(path, map_location="cpu")


def looks_like_native_rwkv7_state_dict(state_dict: Mapping[str, Any]) -> bool:
    if not all(key in state_dict for key in REQUIRED_TOP_LEVEL_KEYS):
        return False
    return (
        "blocks.0.att.receptance.weight" in state_dict
        and "blocks.0.ffn.key.weight" in state_dict
    )


def _unwrap_state_dict_prefix(
    state_dict: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor] | None:
    """Accept checkpoints saved through ``nn.DataParallel``/``torch.compile``."""
    for prefix in ("module.", "_orig_mod."):
        if state_dict and all(
            isinstance(key, str) and key.startswith(prefix) for key in state_dict
        ):
            return {key[len(prefix) :]: value for key, value in state_dict.items()}
    return None


def extract_native_rwkv7_state_dict(
    loaded: Any,
) -> dict[str, torch.Tensor] | None:
    if not isinstance(loaded, dict):
        return None
    if looks_like_native_rwkv7_state_dict(loaded):
        return loaded
    unwrapped = _unwrap_state_dict_prefix(loaded)
    if unwrapped is not None and looks_like_native_rwkv7_state_dict(unwrapped):
        return unwrapped
    for key in ("state_dict", "model", "weights"):
        nested = loaded.get(key)
        if not isinstance(nested, dict):
            continue
        if looks_like_native_rwkv7_state_dict(nested):
            return nested
        unwrapped = _unwrap_state_dict_prefix(nested)
        if unwrapped is not None and looks_like_native_rwkv7_state_dict(unwrapped):
            return unwrapped
    return None


def load_native_rwkv7_state_dict(source: Path) -> dict[str, torch.Tensor]:
    loaded = load_torch_file(source)
    state_dict = extract_native_rwkv7_state_dict(loaded)
    if state_dict is None:
        raise ConversionError(
            f"{source} does not look like a native RWKV7 checkpoint with the "
            "latest parameter names (emb.weight, blocks.0.att.receptance.weight, ...)."
        )
    return state_dict


def transform_tensor(tensor: torch.Tensor, transform: str | None) -> torch.Tensor:
    if transform is None:
        return tensor
    if transform == "transpose":
        if tensor.dim() != 2:
            raise ConversionError(
                f"Cannot transpose a {tensor.dim()}-D tensor with shape {tuple(tensor.shape)}."
            )
        return tensor.transpose(0, 1)
    if transform == "flatten":
        return tensor.reshape(-1)
    raise ConversionError(f"Unknown tensor transform: {transform!r}")


def iter_layer_indices(state_dict: Mapping[str, torch.Tensor]) -> list[int]:
    layer_indices = sorted(
        {
            int(match.group(1))
            for key in state_dict
            if (match := BLOCK_RE.fullmatch(key)) is not None
        }
    )
    if not layer_indices:
        raise ConversionError("Unable to infer the layer count from the checkpoint.")
    if layer_indices != list(range(len(layer_indices))):
        raise ConversionError(
            "Native checkpoint has non-contiguous block indices: "
            f"{layer_indices[:10]}{' ...' if len(layer_indices) > 10 else ''}."
        )
    return layer_indices


def map_native_weight(
    name: str, tensor: torch.Tensor
) -> list[tuple[str, torch.Tensor]]:
    if name in TOP_LEVEL_NAME_MAP:
        return [(TOP_LEVEL_NAME_MAP[name], tensor)]

    match = BLOCK_RE.fullmatch(name)
    if match is None:
        raise ConversionError(f"Unsupported native RWKV7 parameter: {name}")

    layer_idx = int(match.group(1))
    suffix = match.group(2)
    prefix = f"model.layers.{layer_idx}"

    if suffix == "ln0.weight":
        if layer_idx != 0:
            raise ConversionError(f"Unexpected ln0 parameter in layer {layer_idx}: {name}")
        return [(f"{prefix}.pre_norm.weight", tensor)]
    if suffix == "ln0.bias":
        if layer_idx != 0:
            raise ConversionError(f"Unexpected ln0 parameter in layer {layer_idx}: {name}")
        return [(f"{prefix}.pre_norm.bias", tensor)]

    mapped_name = BLOCK_NAME_MAP.get(suffix)
    if mapped_name is not None:
        transform = "flatten" if suffix in {"att.k_k", "att.k_a", "ffn.x_k"} else None
        return [(f"{prefix}.{mapped_name}", transform_tensor(tensor, transform))]

    lora_spec = LORA_SPECS.get(suffix)
    if lora_spec is None:
        raise ConversionError(f"Unsupported RWKV7 block parameter: {name}")

    # The HF model does not instantiate a value-residual LoRA on layer 0.
    if layer_idx == 0 and suffix in DROPPED_LAYER_ZERO_SUFFIXES:
        return []

    mapped_name, transform = lora_spec
    return [(f"{prefix}.{mapped_name}", transform_tensor(tensor, transform))]


def map_native_state_dict(
    state_dict: Mapping[str, torch.Tensor],
) -> OrderedDict[str, torch.Tensor]:
    converted: OrderedDict[str, torch.Tensor] = OrderedDict()
    for name, tensor in state_dict.items():
        if not isinstance(tensor, torch.Tensor):
            raise ConversionError(f"Native parameter {name!r} is not a tensor.")
        for mapped_name, mapped_tensor in map_native_weight(name, tensor):
            if mapped_name in converted:
                raise ConversionError(
                    f"Two native parameters map to the same HF parameter: {mapped_name}"
                )
            converted[mapped_name] = mapped_tensor
    return converted


def infer_max_position_embeddings(source: Path, override: int | None = None) -> int:
    if override is not None:
        if override <= 0:
            raise ConversionError("--max-position-embeddings must be positive.")
        return override
    match = CTX_RE.search(source.stem)
    if match is not None:
        return int(match.group(1))
    print(
        f"warning: {source.name} does not contain ctxNNNN; using "
        f"max_position_embeddings={DEFAULT_MAX_POSITION_EMBEDDINGS}. "
        "Pass --max-position-embeddings to override.",
        file=sys.stderr,
    )
    return DEFAULT_MAX_POSITION_EMBEDDINGS


def torch_dtype_to_hf_string(dtype: torch.dtype) -> str:
    names = {
        torch.float16: "float16",
        torch.float32: "float32",
        torch.bfloat16: "bfloat16",
    }
    if dtype not in names:
        raise ConversionError(f"Unsupported checkpoint dtype: {dtype}.")
    return names[dtype]


def resolve_vocab_path(hf_source: Path, explicit: Path | None = None) -> Path:
    if explicit is not None:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise ConversionError(f"Vocabulary file does not exist: {path}")
        return path

    candidates = sorted(Path(hf_source).glob(VOCAB_GLOB))
    if not candidates:
        raise ConversionError(
            f"No {VOCAB_GLOB!r} file found in {hf_source}. "
            "Pass --vocab-file explicitly."
        )
    if len(candidates) == 1:
        return candidates[0].resolve()

    preferred = Path(hf_source) / "rwkv_vocab_v20230424.txt"
    if preferred in candidates:
        return preferred.resolve()

    names = ", ".join(path.name for path in candidates)
    raise ConversionError(
        f"Multiple vocabulary files found in {hf_source}: {names}. "
        "Pass --vocab-file explicitly."
    )


def read_vocab_info(vocab_path: Path) -> VocabInfo:
    """Parse only the metadata and special-token rows of an RWKV vocab file.

    Tokenizing the whole 65k-line file is unnecessary and makes conversion
    needlessly slow.  Every row's id ordering is validated; the token literal
    and byte length are parsed only for the special-token rows.
    """
    token_ids: dict[str, int] = {}
    min_token_id = sys.maxsize
    max_token_id = -1
    num_tokens = 0
    last_id = -1

    with vocab_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.rstrip("\n")
            if not line:
                continue
            first_space = line.find(" ")
            last_space = line.rfind(" ")
            if first_space <= 0 or last_space <= first_space:
                raise ConversionError(
                    f"Malformed vocabulary line {line_number} in {vocab_path}: {line!r}"
                )
            token_id = int(line[:first_space])
            if token_id <= last_id:
                raise ConversionError(
                    f"Vocabulary ids are not strictly increasing at line {line_number} "
                    f"in {vocab_path}."
                )
            last_id = token_id
            min_token_id = min(min_token_id, token_id)
            max_token_id = max(max_token_id, token_id)
            num_tokens += 1

            # Only the handful of special tokens need to be decoded.  Checking
            # the raw line first avoids running literal_eval on every row.
            if not any(token in line for token in VOCAB_SPECIAL_TOKENS):
                continue
            try:
                token = literal_eval(line[first_space:last_space])
            except (SyntaxError, ValueError) as exc:
                raise ConversionError(
                    f"Cannot parse vocabulary line {line_number} in {vocab_path}: {exc}"
                ) from exc
            if isinstance(token, str):
                token_bytes = token.encode("utf-8")
            elif isinstance(token, bytes):
                token_bytes = token
            else:
                raise ConversionError(
                    f"Unsupported token value on line {line_number} in {vocab_path}."
                )
            expected_length = int(line[last_space + 1 :])
            if len(token_bytes) != expected_length:
                raise ConversionError(
                    f"Token length mismatch on line {line_number} in {vocab_path}: "
                    f"declared {expected_length}, actual {len(token_bytes)}."
                )
            try:
                token_text = token_bytes.decode("utf-8")
            except UnicodeDecodeError:
                continue
            if token_text in VOCAB_SPECIAL_TOKENS:
                token_ids[token_text] = token_id

    if num_tokens == 0:
        raise ConversionError(f"Vocabulary is empty: {vocab_path}")
    return VocabInfo(
        path=vocab_path,
        token_ids=token_ids,
        min_token_id=min_token_id,
        max_token_id=max_token_id,
        num_tokens=num_tokens,
    )


def resolve_special_token_ids(vocab: VocabInfo) -> tuple[int, list[str]]:
    """Return (eos_id, optional special tokens that already exist in the vocab).

    RWKV reserves token 0 for ``<|endoftext|>``.  Some vocabulary files also
    contain an explicit ``<|endoftext|>`` row (usually at the end); when that
    row exists we use its id instead of inventing one.
    """
    eos_id = vocab.token_ids.get(EOS_TOKEN)
    if eos_id is None:
        if vocab.min_token_id == 0:
            raise ConversionError(
                "Token id 0 is already used by the vocabulary, but the checkpoint "
                "uses id 0 as <|endoftext|>."
            )
        eos_id = 0
    optional = [
        token
        for token in OPTIONAL_SPECIAL_TOKENS
        if token in vocab.token_ids and vocab.token_ids[token] != eos_id
    ]
    return eos_id, optional


def validate_chat_template(
    chat_template: str, vocab: VocabInfo, eos_id: int, extra_tokens: set[str] | None = None
) -> None:
    available = set(vocab.token_ids)
    if extra_tokens:
        available |= set(extra_tokens)
    if eos_id == 0:
        available.add(EOS_TOKEN)
    # Catch every <|...|> control token, plus the RWKV separator/control
    # tokens that do not use that syntax.
    required = set(re.findall(r"<\|[^|>]+\|>", chat_template))
    for token in CHAT_TEMPLATE_REQUIRED_TOKENS:
        if token in chat_template:
            required.add(token)
    # vision tokens only appear in multimodal turns; a text-only vocabulary
    # legitimately lacks them, so they are not a hard requirement
    VISION_TOKENS = {"<|vision_start|>", "<|image_pad|>", "<|video_pad|>", "<|vision_end|>"}
    missing = sorted(
        token for token in required
        if token not in available and token not in VISION_TOKENS
    )
    if missing:
        raise ConversionError(
            "The chat template references tokens that are absent from "
            f"{vocab.path.name}: {missing}. Use the default RWKV template "
            "(hf/chat_template.jinja) or provide a vocabulary that contains "
            "these special tokens."
        )


def build_tokenizer_files(
    vocab: VocabInfo,
    chat_template: str,
    *,
    model_max_length: int | None = None,
    vocab_size: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, int], dict[str, int]]:
    eos_id, optional_special_tokens = resolve_special_token_ids(vocab)

    # ChatML templates need <|im_start|> / <|im_end|> (and the think pair); the
    # native vocabulary does not carry them, so park them on the unused tail.
    chatml_tokens: dict[str, int] = {}
    taken = set(vocab.token_ids.values())
    for token, token_id in CHATML_TOKEN_IDS.items():
        if token not in chat_template or token in vocab.token_ids:
            continue
        if vocab_size is not None and token_id >= vocab_size:
            raise ConversionError(
                f"The vocabulary has only {vocab_size} rows, so {token!r} "
                f"cannot take id {token_id}."
            )
        if token_id in taken:
            raise ConversionError(
                f"id {token_id} is already used, cannot park {token!r} there."
            )
        chatml_tokens[token] = token_id
        taken.add(token_id)
    if CHATML_END_OF_TURN in chatml_tokens:
        eos_id = chatml_tokens[CHATML_END_OF_TURN]

    added_tokens_decoder: dict[str, Any] = {}
    for token, token_id in sorted(chatml_tokens.items(), key=lambda kv: kv[1]):
        added_tokens_decoder[str(token_id)] = {
            "content": token,
            "lstrip": False,
            "normalized": False,
            "rstrip": False,
            "single_word": False,
            "special": token in CHATML_ADDED_SPECIAL,
        }
    if eos_id not in vocab.token_ids.values() and str(eos_id) not in added_tokens_decoder:
        added_tokens_decoder[str(eos_id)] = {
            "content": EOS_TOKEN,
            "lstrip": False,
            "normalized": False,
            "rstrip": False,
            "single_word": False,
            "special": True,
        }

    tokenizer_config = {
        "add_prefix_space": False,
        "tokenizer_class": "RwkvTokenizer",
        "use_fast": False,
        "clean_up_tokenization_spaces": False,
        "auto_map": {
            "AutoTokenizer": [
                "hf_rwkv_tokenizer.RwkvTokenizer",
                None,
            ]
        },
        "add_bos_token": True,
        "bos_token": EOS_TOKEN,
        "eos_token": CHATML_END_OF_TURN if chatml_tokens else EOS_TOKEN,
        "unk_token": None,
        "pad_token": EOS_TOKEN,
        "additional_special_tokens": list(optional_special_tokens)
        + [t for t in CHATML_ADDED_SPECIAL if t in chatml_tokens],
        "model_max_length": model_max_length or 1_000_000_000_000,
        "chat_template": chat_template,
        "added_tokens_decoder": added_tokens_decoder,
    }

    special_tokens_map = {
        "bos_token": EOS_TOKEN,
        "eos_token": CHATML_END_OF_TURN if chatml_tokens else EOS_TOKEN,
        "pad_token": EOS_TOKEN,
        "additional_special_tokens": list(optional_special_tokens)
        + [t for t in CHATML_ADDED_SPECIAL if t in chatml_tokens],
    }

    explicit_ids = {
        "bos_token_id": 0,
        "eos_token_id": eos_id if chatml_tokens else eos_id,
        "pad_token_id": 0 if chatml_tokens else None,
    }

    added_tokens = {EOS_TOKEN: 0}
    added_tokens.update(chatml_tokens)

    return tokenizer_config, special_tokens_map, explicit_ids, added_tokens


def _tensor_shape(
    state_dict: Mapping[str, torch.Tensor], name: str
) -> tuple[int, ...]:
    tensor = state_dict.get(name)
    if tensor is None:
        raise ConversionError(f"Native checkpoint is missing required parameter: {name}")
    return tuple(tensor.shape)


def _numel(
    state_dict: Mapping[str, torch.Tensor], name: str
) -> int:
    shape = _tensor_shape(state_dict, name)
    result = 1
    for size in shape:
        result *= size
    return result


def _check_shape(
    state_dict: Mapping[str, torch.Tensor],
    name: str,
    expected: tuple[int, ...],
    *,
    what: str,
) -> None:
    actual = _tensor_shape(state_dict, name)
    if actual != expected:
        raise ConversionError(
            f"Unexpected shape for {name}: expected {expected}, found {actual} ({what})."
        )


def _check_numel(
    state_dict: Mapping[str, torch.Tensor],
    name: str,
    expected: int,
    *,
    what: str,
) -> None:
    actual = _numel(state_dict, name)
    if actual != expected:
        raise ConversionError(
            f"Unexpected size for {name}: expected {expected} elements, found {actual} ({what})."
        )


def validate_native_state_dict(
    state_dict: Mapping[str, torch.Tensor],
) -> tuple[list[int], dict[str, Any]]:
    """Validate the native layout and return useful dimensions.

    The converter should fail here with a clear message rather than producing
    an HF directory that later fails inside ``load_state_dict``.
    """
    if not all(key in state_dict for key in REQUIRED_TOP_LEVEL_KEYS):
        missing = [key for key in REQUIRED_TOP_LEVEL_KEYS if key not in state_dict]
        raise ConversionError(f"Native checkpoint is missing required weights: {missing}")

    layer_indices = iter_layer_indices(state_dict)
    emb_shape = _tensor_shape(state_dict, "emb.weight")
    if len(emb_shape) != 2:
        raise ConversionError(f"emb.weight must be 2-D, found {emb_shape}.")
    vocab_size, hidden_size = emb_shape
    _check_shape(
        state_dict,
        "head.weight",
        (vocab_size, hidden_size),
        what="LM head must match the embedding matrix",
    )
    _check_shape(state_dict, "ln_out.weight", (hidden_size,), what="final norm")

    r_k_shape = _tensor_shape(state_dict, "blocks.0.att.r_k")
    if len(r_k_shape) != 2:
        raise ConversionError(f"blocks.0.att.r_k must be 2-D, found {r_k_shape}.")
    num_heads, head_dim = r_k_shape
    if head_dim <= 0 or hidden_size % head_dim != 0:
        raise ConversionError(
            f"head_dim={head_dim} does not divide hidden_size={hidden_size}."
        )
    if num_heads != hidden_size // head_dim:
        raise ConversionError(
            f"r_k implies num_heads={num_heads}, head_dim={head_dim}, but "
            f"hidden_size={hidden_size} implies {hidden_size // head_dim} heads."
        )

    dims = {
        "hidden_size": int(hidden_size),
        "num_heads": int(num_heads),
        "head_dim": int(head_dim),
        "intermediate_size": int(_tensor_shape(state_dict, "blocks.0.ffn.key.weight")[0]),
    }
    _check_shape(
        state_dict,
        "blocks.0.ffn.value.weight",
        (hidden_size, dims["intermediate_size"]),
        what="FFN inverse projection",
    )

    norm_bias = "ln_out.bias" in state_dict
    if norm_bias:
        _check_shape(state_dict, "ln_out.bias", (hidden_size,), what="final norm bias")

    value_dims: list[int] = []
    decay_ranks: set[int] = set()
    gate_ranks: set[int] = set()
    a_ranks: set[int] = set()
    v_ranks: set[int] = set()

    for layer_idx in layer_indices:
        prefix = f"blocks.{layer_idx}"
        for norm in ("ln1", "ln2"):
            _check_shape(
                state_dict, f"{prefix}.{norm}.weight", (hidden_size,), what=norm
            )
            if norm_bias:
                _check_shape(
                    state_dict, f"{prefix}.{norm}.bias", (hidden_size,), what=norm
                )
            elif f"{prefix}.{norm}.bias" in state_dict:
                raise ConversionError(
                    f"Layer {layer_idx} has {norm}.bias but ln_out.bias is absent; "
                    "norm_bias must be consistent across the model."
                )
        if layer_idx == 0:
            _check_shape(
                state_dict, f"{prefix}.ln0.weight", (hidden_size,), what="ln0"
            )
            if norm_bias:
                _check_shape(
                    state_dict, f"{prefix}.ln0.bias", (hidden_size,), what="ln0"
                )

        for name in ("x_r", "x_w", "x_k", "x_v", "x_a", "x_g"):
            _check_shape(
                state_dict,
                f"{prefix}.att.{name}",
                (1, 1, hidden_size),
                what=name,
            )
        _check_shape(
            state_dict, f"{prefix}.att.k_k", (1, 1, hidden_size), what="k_k"
        )
        _check_shape(
            state_dict, f"{prefix}.att.k_a", (1, 1, hidden_size), what="k_a"
        )
        _check_shape(
            state_dict,
            f"{prefix}.att.r_k",
            (num_heads, head_dim),
            what="r_k",
        )

        for proj in ("receptance", "key"):
            _check_shape(
                state_dict,
                f"{prefix}.att.{proj}.weight",
                (hidden_size, hidden_size),
                what=proj,
            )
        value_dim = int(_tensor_shape(state_dict, f"{prefix}.att.output.weight")[1])
        if value_dim % num_heads != 0:
            raise ConversionError(
                f"Layer {layer_idx}: value_dim={value_dim} is not divisible by "
                f"num_heads={num_heads}."
            )
        if value_dim < hidden_size:
            raise ConversionError(
                f"Layer {layer_idx}: value_dim={value_dim} must be >= hidden_size={hidden_size}."
            )
        _check_shape(
            state_dict,
            f"{prefix}.att.value.weight",
            (value_dim, hidden_size),
            what="value projection",
        )
        _check_shape(
            state_dict,
            f"{prefix}.att.output.weight",
            (hidden_size, value_dim),
            what="output projection",
        )
        _check_shape(
            state_dict, f"{prefix}.att.ln_x.weight", (value_dim,), what="ln_x"
        )
        _check_shape(
            state_dict, f"{prefix}.att.ln_x.bias", (value_dim,), what="ln_x bias"
        )
        value_dims.append(value_dim)

        w1_shape = _tensor_shape(state_dict, f"{prefix}.att.w1")
        if len(w1_shape) != 2:
            raise ConversionError(
                f"{prefix}.att.w1 must be 2-D, found {w1_shape}."
            )
        decay_rank = int(w1_shape[1])
        _check_shape(
            state_dict,
            f"{prefix}.att.w1",
            (hidden_size, decay_rank),
            what="decay low-rank input",
        )
        _check_shape(
            state_dict,
            f"{prefix}.att.w2",
            (decay_rank, hidden_size),
            what="decay low-rank output",
        )
        _check_numel(state_dict, f"{prefix}.att.w0", hidden_size, what="decay bias")
        decay_ranks.add(decay_rank)

        a1_shape = _tensor_shape(state_dict, f"{prefix}.att.a1")
        if len(a1_shape) != 2:
            raise ConversionError(
                f"{prefix}.att.a1 must be 2-D, found {a1_shape}."
            )
        a_rank = int(a1_shape[1])
        _check_shape(
            state_dict,
            f"{prefix}.att.a1",
            (hidden_size, a_rank),
            what="a low-rank input",
        )
        _check_shape(
            state_dict,
            f"{prefix}.att.a2",
            (a_rank, hidden_size),
            what="a low-rank output",
        )
        _check_numel(state_dict, f"{prefix}.att.a0", hidden_size, what="a bias")
        a_ranks.add(a_rank)

        g1_shape = _tensor_shape(state_dict, f"{prefix}.att.g1")
        if len(g1_shape) != 2:
            raise ConversionError(
                f"{prefix}.att.g1 must be 2-D, found {g1_shape}."
            )
        gate_rank = int(g1_shape[1])
        _check_shape(
            state_dict,
            f"{prefix}.att.g1",
            (hidden_size, gate_rank),
            what="gate low-rank input",
        )
        _check_shape(
            state_dict,
            f"{prefix}.att.g2",
            (gate_rank, value_dim),
            what="gate low-rank output",
        )
        gate_ranks.add(gate_rank)

        if layer_idx > 0:
            v1_shape = _tensor_shape(state_dict, f"{prefix}.att.v1")
            if len(v1_shape) != 2:
                raise ConversionError(
                    f"{prefix}.att.v1 must be 2-D, found {v1_shape}."
                )
            v_rank = int(v1_shape[1])
            _check_shape(
                state_dict,
                f"{prefix}.att.v1",
                (hidden_size, v_rank),
                what="value low-rank input",
            )
            _check_shape(
                state_dict,
                f"{prefix}.att.v2",
                (v_rank, value_dim),
                what="value low-rank output",
            )
            _check_numel(state_dict, f"{prefix}.att.v0", value_dim, what="value bias")
            v_ranks.add(v_rank)

        _check_numel(state_dict, f"{prefix}.ffn.x_k", hidden_size, what="ffn.x_k")
        _check_shape(
            state_dict,
            f"{prefix}.ffn.key.weight",
            (dims["intermediate_size"], hidden_size),
            what="FFN key projection",
        )
        _check_shape(
            state_dict,
            f"{prefix}.ffn.value.weight",
            (hidden_size, dims["intermediate_size"]),
            what="FFN value projection",
        )

    for kind, ranks in (
        ("decay_low_rank_dim", decay_ranks),
        ("gate_low_rank_dim", gate_ranks),
        ("a_low_rank_dim", a_ranks),
    ):
        if len(ranks) != 1:
            raise ConversionError(
                f"All layers must use the same {kind}; found {sorted(ranks)}."
            )
    if len(v_ranks) != 1:
        raise ConversionError(
            f"All non-zero layers must use the same v_low_rank_dim; found {sorted(v_ranks)}."
        )

    dims.update(
        {
            "vocab_size": int(vocab_size),
            "num_layers": len(layer_indices),
            "value_dim": value_dims,
            "decay_low_rank_dim": decay_ranks.pop(),
            "gate_low_rank_dim": gate_ranks.pop(),
            "a_low_rank_dim": a_ranks.pop(),
            "v_low_rank_dim": v_ranks.pop() if v_ranks else 0,
            "norm_bias": norm_bias,
        }
    )
    return layer_indices, dims


def build_model_config(
    state_dict: Mapping[str, torch.Tensor],
    source: Path,
    special_token_ids: Mapping[str, int],
    max_position_embeddings: int | None = None,
    model_type: str = DEFAULT_MODEL_TYPE,
) -> dict[str, Any]:
    _, dims = validate_native_state_dict(state_dict)
    hidden_size = dims["hidden_size"]
    intermediate_size = dims["intermediate_size"]
    hidden_ratio = (
        intermediate_size // hidden_size
        if intermediate_size % hidden_size == 0
        else intermediate_size / hidden_size
    )
    max_pos = infer_max_position_embeddings(source, max_position_embeddings)

    config = {
        "model_type": model_type,
        "architectures": ["RWKV7ForCausalLM"],
        "auto_map": {
            "AutoConfig": "configuration_rwkv7.RWKV7Config",
            "AutoModel": "modeling_rwkv7.RWKV7Model",
            "AutoModelForCausalLM": "modeling_rwkv7.RWKV7ForCausalLM",
        },
        "attn": None,
        "attn_mode": "chunk",
        "hidden_size": hidden_size,
        "hidden_ratio": hidden_ratio,
        "intermediate_size": intermediate_size,
        "num_hidden_layers": dims["num_layers"],
        "head_dim": dims["head_dim"],
        "num_heads": dims["num_heads"],
        "value_dim": dims["value_dim"],
        "decay_low_rank_dim": dims["decay_low_rank_dim"],
        "gate_low_rank_dim": dims["gate_low_rank_dim"],
        "a_low_rank_dim": dims["a_low_rank_dim"],
        "v_low_rank_dim": dims["v_low_rank_dim"],
        "hidden_act": "sqrelu",
        "max_position_embeddings": max_pos,
        "norm_first": True,
        "norm_bias": dims["norm_bias"],
        "norm_eps": 1e-5,
        "tie_word_embeddings": False,
        "use_cache": True,
        "initializer_range": 0.006,
        "fuse_norm": False,
        "fuse_cross_entropy": False,
        "fuse_linear_cross_entropy": False,
        "use_l2warp": False,
        "vocab_size": dims["vocab_size"],
        "torch_dtype": torch_dtype_to_hf_string(state_dict["emb.weight"].dtype),
        **special_token_ids,
    }
    return config


def build_generation_config(
    special_token_ids: Mapping[str, int],
    max_position_embeddings: int,
    extra_eos_token_ids: Sequence[int] = (),
    chatml: bool = False,
) -> dict[str, Any]:
    eos_token_ids = [special_token_ids["eos_token_id"]]
    for token_id in extra_eos_token_ids:
        if token_id not in eos_token_ids:
            eos_token_ids.append(token_id)
    eos_token_id: int | list[int] = (
        eos_token_ids[0] if len(eos_token_ids) == 1 else eos_token_ids
    )
    return {
        "bos_token_id": special_token_ids["bos_token_id"],
        "eos_token_id": eos_token_id,
        "pad_token_id": special_token_ids["pad_token_id"],
        # Keep this small and explicit: the converter should not bake an
        # arbitrary sampling policy into the model directory.
        "do_sample": False,
        "max_new_tokens": min(max_position_embeddings, 2048),
        # Runtime stop markers for the embedded chat template.
        "stop_strings": (
            ["<|im_end|>"] if chatml else ["✿", "\nUser:", "\n### User"]
        ),
    }


def _expected_hf_keys(config: Mapping[str, Any]) -> set[str]:
    keys = {
        "model.embeddings.weight",
        "model.norm.weight",
        "lm_head.weight",
    }
    if config["norm_bias"]:
        keys.add("model.norm.bias")
    for layer_idx in range(config["num_hidden_layers"]):
        prefix = f"model.layers.{layer_idx}"
        if layer_idx == 0:
            keys.add(f"{prefix}.pre_norm.weight")
            if config["norm_bias"]:
                keys.add(f"{prefix}.pre_norm.bias")
        for norm in ("attn_norm", "ffn_norm"):
            keys.add(f"{prefix}.{norm}.weight")
            if config["norm_bias"]:
                keys.add(f"{prefix}.{norm}.bias")
        for name in ("x_r", "x_w", "x_k", "x_v", "x_a", "x_g", "k_k", "k_a", "r_k"):
            keys.add(f"{prefix}.attn.{name}")
        for name in ("r_proj", "k_proj", "v_proj", "o_proj"):
            keys.add(f"{prefix}.attn.{name}.weight")
        for lora in ("w_lora", "a_lora"):
            keys.add(f"{prefix}.attn.{lora}.lora.0.weight")
            keys.add(f"{prefix}.attn.{lora}.lora.2.weight")
            keys.add(f"{prefix}.attn.{lora}.lora.2.bias")
        keys.add(f"{prefix}.attn.g_lora.lora.0.weight")
        keys.add(f"{prefix}.attn.g_lora.lora.2.weight")
        if layer_idx > 0:
            keys.add(f"{prefix}.attn.v_lora.lora.0.weight")
            keys.add(f"{prefix}.attn.v_lora.lora.2.weight")
            keys.add(f"{prefix}.attn.v_lora.lora.2.bias")
        keys.add(f"{prefix}.attn.g_norm.weight")
        keys.add(f"{prefix}.attn.g_norm.bias")
        keys.add(f"{prefix}.ffn.x_k")
        keys.add(f"{prefix}.ffn.key.weight")
        keys.add(f"{prefix}.ffn.value.weight")
    return keys


def validate_converted_state_dict(
    converted: Mapping[str, torch.Tensor], config: Mapping[str, Any]
) -> None:
    expected = _expected_hf_keys(config)
    actual = set(converted)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing or unexpected:
        details = []
        if missing:
            details.append(f"missing={missing[:10]}")
        if unexpected:
            details.append(f"unexpected={unexpected[:10]}")
        raise ConversionError(
            "Converted state dict does not match the HF model layout: "
            + ", ".join(details)
        )


def check_output_dir(output_dir: Path, source: Path, overwrite: bool) -> None:
    output_dir = output_dir.resolve()
    source = source.resolve()
    if output_dir.parent == output_dir:
        raise ConversionError(f"Refusing to use a filesystem root as output: {output_dir}")
    if output_dir == DEFAULT_ROOT or DEFAULT_ROOT.is_relative_to(output_dir):
        raise ConversionError(
            f"Refusing to overwrite the repository directory: {output_dir}"
        )
    if source.is_relative_to(output_dir):
        raise ConversionError(
            "Output directory would contain the source checkpoint and could "
            f"delete it on --overwrite: {output_dir}"
        )
    if output_dir.exists():
        if not output_dir.is_dir():
            raise ConversionError(f"Output path is not a directory: {output_dir}")
        if any(output_dir.iterdir()) and not overwrite:
            raise ConversionError(
                f"Output directory already exists and is not empty: {output_dir}. "
                "Pass --overwrite to replace it."
            )


def _make_temp_output_dir(output_dir: Path) -> Path:
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    return Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.tmp-",
            dir=str(output_dir.parent),
        )
    )


def commit_output_dir(temp_dir: Path, output_dir: Path, overwrite: bool) -> None:
    backup: Path | None = None
    if output_dir.exists():
        if any(output_dir.iterdir()):
            if not overwrite:
                raise ConversionError(
                    f"Output directory already exists and is not empty: {output_dir}"
                )
            backup = output_dir.with_name(
                f".{output_dir.name}.bak-{uuid.uuid4().hex}"
            )
            output_dir.rename(backup)
        else:
            output_dir.rmdir()
    try:
        temp_dir.rename(output_dir)
    except BaseException:
        if backup is not None and not output_dir.exists():
            backup.rename(output_dir)
        raise
    # tempfile.mkdtemp creates 0700 directories; converted models are normal
    # user-readable artifacts.
    output_dir.chmod(0o755)
    if backup is not None:
        shutil.rmtree(backup)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_weight_files(
    state_dict: Mapping[str, torch.Tensor],
    output_dir: Path,
    max_shard_size: str,
) -> int:
    state_dict_split = split_torch_state_dict_into_shards(
        state_dict,
        filename_pattern="model{suffix}.safetensors",
        max_shard_size=max_shard_size,
    )

    for filename, tensor_names in state_dict_split.filename_to_tensors.items():
        shard = {
            name: state_dict[name].contiguous()
            for name in tensor_names
        }
        save_file(shard, str(output_dir / filename), metadata={"format": "pt"})

    if state_dict_split.is_sharded:
        write_json(
            output_dir / MODEL_INDEX_NAME,
            {
                "metadata": state_dict_split.metadata,
                "weight_map": state_dict_split.tensor_to_filename,
            },
        )

    return len(state_dict_split.filename_to_tensors)


def warn_untrained_special_tokens(
    state_dict: Mapping[str, torch.Tensor],
    vocab: VocabInfo,
    eos_id: int,
    added: Mapping[str, int] | None = None,
) -> None:
    """Report special tokens the checkpoint never trained.

    Tokens this converter just parked on free ids are expected to be zero — the
    checkpoint predates them — so they are reported as a note, not a warning.
    """
    embedding = state_dict.get("emb.weight")
    if embedding is None:
        return
    added = dict(added or {})
    # name the ids we introduced first, then fall back to the vocabulary names
    candidates = {**vocab.token_ids, **added}
    candidates[EOS_TOKEN] = 0
    for token, token_id in sorted(candidates.items(), key=lambda item: item[1]):
        if not (0 <= token_id < embedding.shape[0]):
            continue
        if embedding[token_id].float().abs().max().item() >= 1e-6:
            continue
        if token in added:
            print(
                f"note: {token!r} takes the free id {token_id} and its embedding "
                "is all-zero in the checkpoint; SFT has to learn it.",
                file=sys.stderr,
            )
        else:
            print(
                f"warning: token {token!r} (id {token_id}) has an all-zero "
                "embedding in the checkpoint; the vocabulary may not match "
                "the model. The token is still written to the output.",
                file=sys.stderr,
            )


def _resolve_asset(path: Path, label: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise ConversionError(f"{label} does not exist: {resolved}")
    return resolved


def prepare_artifacts(
    state_dict: Mapping[str, torch.Tensor],
    source: Path,
    vocab: VocabInfo,
    chat_template: str,
    max_position_embeddings: int | None,
    model_type: str,
) -> ConversionArtifacts:
    model_config = build_model_config(
        state_dict,
        source,
        # filled below, after tokenizer ids are known
        {},
        max_position_embeddings=max_position_embeddings,
        model_type=model_type,
    )
    tokenizer_config, special_tokens_map, explicit_ids, added_tokens = (
        build_tokenizer_files(
            vocab,
            chat_template,
            model_max_length=model_config["max_position_embeddings"],
        )
    )
    model_config.update(explicit_ids)
    warn_untrained_special_tokens(
        state_dict,
        vocab,
        explicit_ids["eos_token_id"],
        added_tokens,
    )
    validate_chat_template(
        chat_template,
        vocab,
        explicit_ids["eos_token_id"],
        extra_tokens=set(added_tokens),
    )
    extra_eos_token_ids: list[int] = []
    if explicit_ids["pad_token_id"] == 0 and explicit_ids["eos_token_id"] != 0:
        # ChatML: <|endoftext|> (id 0) also ends a document
        extra_eos_token_ids.append(0)
    turn_end_id = vocab.token_ids.get("✿")
    if (
        turn_end_id is not None
        and turn_end_id != explicit_ids["eos_token_id"]
        and "✿" in chat_template
    ):
        # The simple RWKV chat template uses ✿ as the end-of-turn marker.  Stop
        # on it as well as on the document-level <|endoftext|>.
        extra_eos_token_ids.append(turn_end_id)
    generation_config = build_generation_config(
        explicit_ids,
        model_config["max_position_embeddings"],
        extra_eos_token_ids,
        chatml="<|im_start|>" in chat_template,
    )
    converted = map_native_state_dict(state_dict)
    validate_converted_state_dict(converted, model_config)
    return ConversionArtifacts(
        state_dict=converted,
        model_config=model_config,
        generation_config=generation_config,
        tokenizer_config=tokenizer_config,
        special_tokens_map=special_tokens_map,
        added_tokens=added_tokens,
        chat_template=chat_template,
    )


def convert_checkpoint(
    source: Path,
    output_dir: Path,
    *,
    hf_source: Path = DEFAULT_HF_SOURCE,
    vocab_path: Path | None = None,
    chat_template_path: Path | None = None,
    tokenizer_source: Path | None = None,
    modeling_source: Path | None = None,
    kernels_source: Path | None = None,
    config_source: Path | None = None,
    max_shard_size: str = "5GB",
    max_position_embeddings: int | None = None,
    model_type: str = DEFAULT_MODEL_TYPE,
    overwrite: bool = False,
) -> ConversionSummary:
    source = Path(source).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    hf_source = Path(hf_source).expanduser().resolve()

    if not source.is_file():
        raise ConversionError(f"Source checkpoint does not exist: {source}")
    check_output_dir(output_dir, source, overwrite)

    default_template = hf_source / "chat_template.jinja"
    if chat_template_path is not None:
        template_path = _resolve_asset(Path(chat_template_path), "Chat template")
        chat_template = template_path.read_text(encoding="utf-8")
    elif default_template.is_file():
        chat_template = default_template.read_text(encoding="utf-8")
    else:
        chat_template = DEFAULT_CHAT_TEMPLATE

    resolved_vocab = resolve_vocab_path(hf_source, vocab_path)
    vocab = read_vocab_info(resolved_vocab)

    tokenizer_source = _resolve_asset(
        tokenizer_source if tokenizer_source is not None else hf_source / "hf_rwkv_tokenizer.py",
        "Tokenizer implementation",
    )
    modeling_source = _resolve_asset(
        modeling_source if modeling_source is not None else hf_source / "modeling_rwkv7.py",
        "Modeling implementation",
    )
    kernels_source = _resolve_asset(
        kernels_source if kernels_source is not None else hf_source / "rwkv7_kernels.py",
        "Kernel backend",
    )
    config_source = _resolve_asset(
        config_source if config_source is not None else hf_source / "configuration_rwkv7.py",
        "Configuration implementation",
    )

    state_dict = load_native_rwkv7_state_dict(source)
    artifacts = prepare_artifacts(
        state_dict,
        source,
        vocab,
        chat_template,
        max_position_embeddings,
        model_type,
    )

    actual_vocab_size = artifacts.model_config["vocab_size"]
    if vocab.max_token_id >= actual_vocab_size:
        raise ConversionError(
            f"Vocabulary contains token id {vocab.max_token_id}, but the checkpoint "
            f"only has {actual_vocab_size} embedding rows."
        )
    if artifacts.model_config["eos_token_id"] >= actual_vocab_size:
        raise ConversionError(
            f"eos_token_id={artifacts.model_config['eos_token_id']} is outside the "
            f"embedding matrix ({actual_vocab_size} rows)."
        )

    temp_dir = _make_temp_output_dir(output_dir)
    try:
        num_shards = write_weight_files(
            artifacts.state_dict,
            temp_dir,
            max_shard_size,
        )
        shutil.copyfile(resolved_vocab, temp_dir / CANONICAL_VOCAB_NAME)
        (temp_dir / "chat_template.jinja").write_text(
            artifacts.chat_template, encoding="utf-8"
        )
        shutil.copyfile(tokenizer_source, temp_dir / "hf_rwkv_tokenizer.py")
        shutil.copyfile(modeling_source, temp_dir / "modeling_rwkv7.py")
        shutil.copyfile(kernels_source, temp_dir / "rwkv7_kernels.py")
        shutil.copyfile(config_source, temp_dir / "configuration_rwkv7.py")
        (temp_dir / "__init__.py").write_text("", encoding="utf-8")

        write_json(temp_dir / "config.json", artifacts.model_config)
        write_json(temp_dir / "generation_config.json", artifacts.generation_config)
        write_json(temp_dir / "tokenizer_config.json", artifacts.tokenizer_config)
        write_json(temp_dir / "special_tokens_map.json", artifacts.special_tokens_map)
        write_json(temp_dir / "added_tokens.json", artifacts.added_tokens)

        commit_output_dir(temp_dir, output_dir, overwrite)
    except BaseException:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise

    return ConversionSummary(
        source=source,
        output_dir=output_dir,
        num_layers=artifacts.model_config["num_hidden_layers"],
        num_tensors=len(artifacts.state_dict),
        num_shards=num_shards,
        vocab_file=resolved_vocab.name,
        dtype=artifacts.model_config["torch_dtype"],
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert native RWKV7 .pth checkpoints into transformers-compatible "
            "Hugging Face directories."
        )
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Path to a native RWKV7 .pth or .pt checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output HF directory.",
    )
    parser.add_argument(
        "--hf-source",
        type=Path,
        default=DEFAULT_HF_SOURCE,
        help=(
            "Directory containing hf_rwkv_tokenizer.py, modeling_rwkv7.py, "
            "rwkv7_kernels.py, configuration_rwkv7.py, the vocab file and the "
            "chat template."
        ),
    )
    parser.add_argument(
        "--vocab-file",
        type=Path,
        default=None,
        help=(
            "Vocabulary .txt to copy into the output directory. Defaults to "
            "the single rwkv_vocab*.txt found in --hf-source."
        ),
    )
    parser.add_argument(
        "--chat-template",
        type=Path,
        default=None,
        help=(
            "Optional external Jinja chat template. The default is "
            "hf/chat_template.jinja (Qwen3.8 ChatML) "
            "ChatML); the template is validated against the vocabulary."
        ),
    )
    parser.add_argument(
        "--max-shard-size",
        default="5GB",
        help="Maximum safetensors shard size, for example 2GB or 500MB.",
    )
    parser.add_argument(
        "--model-type",
        default=DEFAULT_MODEL_TYPE,
        help=(
            "HF model_type written to config.json. The default 'rwkv7_hf' "
            "avoids FLA's global transformers registration for 'rwkv7'. "
            "Use 'rwkv7' only if you intentionally want FLA's built-in model."
        ),
    )
    parser.add_argument(
        "--max-position-embeddings",
        type=int,
        default=None,
        help=(
            "Optional override for the context length when the filename does "
            "not contain ctxNNNN."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing non-empty output directory.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        summary = convert_checkpoint(
            source=args.source,
            output_dir=args.output_dir,
            hf_source=args.hf_source,
            vocab_path=args.vocab_file,
            chat_template_path=args.chat_template,
            max_shard_size=args.max_shard_size,
            max_position_embeddings=args.max_position_embeddings,
            model_type=args.model_type,
            overwrite=args.overwrite,
        )
    except ConversionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(
        json.dumps(
            {
                "source": str(summary.source),
                "output_dir": str(summary.output_dir),
                "num_layers": summary.num_layers,
                "num_tensors": summary.num_tensors,
                "num_shards": summary.num_shards,
                "vocab_file": summary.vocab_file,
                "dtype": summary.dtype,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
