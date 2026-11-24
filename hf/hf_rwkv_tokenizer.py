# coding=utf-8
"""Hugging Face tokenizer shim for the RWKV world vocabulary.

The RWKV vocabulary is a byte-level greedy vocabulary: every line is

    <token_id> <python literal for the token bytes> <byte length>

Tokens are matched greedily by longest byte sequence.  This file keeps that
algorithm (a small trie) and wraps it in the ``PreTrainedTokenizer`` API so
that ``AutoTokenizer.from_pretrained(..., trust_remote_code=True)`` works.
"""

from __future__ import annotations

from ast import literal_eval
import glob
import os
from pathlib import Path
from typing import List, Optional, Tuple

from transformers.tokenization_utils import AddedToken, PreTrainedTokenizer
from transformers.utils import logging


logger = logging.get_logger(__name__)

# Keep this in sync with convert.py:CANONICAL_VOCAB_NAME.
VOCAB_FILENAME = "rwkv_vocab.txt"
VOCAB_FILES_NAMES = {"vocab_file": VOCAB_FILENAME}

EOS_TOKEN = "<|endoftext|>"


class TRIE:
    __slots__ = ("children", "token_id")

    def __init__(self) -> None:
        self.children: list[TRIE | None] = [None] * 256
        self.token_id: int | None = None

    def add(self, key: bytes, token_id: int) -> None:
        node = self
        for byte in key:
            child = node.children[byte]
            if child is None:
                child = TRIE()
                node.children[byte] = child
            node = child
        node.token_id = token_id

    def find_longest(self, key: bytes, start: int) -> tuple[int, int]:
        """Return ``(end_index, token_id)`` for the longest match at ``start``."""
        node = self
        best_end = -1
        best_id = -1
        index = start
        while index < len(key):
            child = node.children[key[index]]
            if child is None:
                break
            node = child
            index += 1
            if node.token_id is not None:
                best_end = index
                best_id = node.token_id
        if best_end < 0:
            raise ValueError(
                f"RWKV vocabulary cannot tokenize byte 0x{key[start]:02x} at offset {start}."
            )
        return best_end, best_id


class RWKVByteTokenizer:
    """The reference greedy byte-level tokenizer."""

    def __init__(self, vocab_file: str | os.PathLike[str]) -> None:
        self.idx2token: dict[int, bytes] = {}
        self.token2idx: dict[bytes, int] = {}
        max_id = -1
        previous_id = -1

        with open(vocab_file, "r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.rstrip("\n")
                if not line:
                    continue
                first_space = line.find(" ")
                last_space = line.rfind(" ")
                if first_space <= 0 or last_space <= first_space:
                    raise ValueError(
                        f"Malformed RWKV vocabulary line {line_number} in {vocab_file}."
                    )
                token_id = int(line[:first_space])
                if token_id <= previous_id:
                    raise ValueError(
                        f"RWKV vocabulary ids must be strictly increasing "
                        f"(line {line_number} in {vocab_file})."
                    )
                previous_id = token_id
                token = literal_eval(line[first_space:last_space])
                if isinstance(token, str):
                    token = token.encode("utf-8")
                if not isinstance(token, bytes):
                    raise ValueError(
                        f"Unsupported token literal on line {line_number} in {vocab_file}."
                    )
                declared_length = int(line[last_space + 1 :])
                if len(token) != declared_length:
                    raise ValueError(
                        f"Token length mismatch on line {line_number} in {vocab_file}: "
                        f"declared {declared_length}, actual {len(token)}."
                    )
                if token in self.token2idx:
                    raise ValueError(
                        f"Duplicate token bytes on line {line_number} in {vocab_file}."
                    )
                self.idx2token[token_id] = token
                self.token2idx[token] = token_id
                max_id = max(max_id, token_id)

        if not self.idx2token:
            raise ValueError(f"RWKV vocabulary is empty: {vocab_file}")
        self.max_id = max_id

        self.root = TRIE()
        for token, token_id in self.token2idx.items():
            self.root.add(token, token_id)

    def encode_bytes(self, src: bytes) -> list[int]:
        tokens: list[int] = []
        index = 0
        while index < len(src):
            index, token_id = self.root.find_longest(src, index)
            tokens.append(token_id)
        return tokens

    def decode_bytes(self, tokens: list[int]) -> bytes:
        return b"".join(self.idx2token[token] for token in tokens)

    def encode(self, text: str) -> list[int]:
        return self.encode_bytes(text.encode("utf-8"))

    def decode(self, tokens: list[int]) -> str:
        return self.decode_bytes(tokens).decode("utf-8", errors="replace")


def _token_bytes_to_text(token: bytes) -> str:
    """Losslessly represent arbitrary bytes as a Python string.

    ``surrogateescape`` round-trips invalid UTF-8 through the string-based
    tokenizer API without merging or dropping bytes.
    """
    return token.decode("utf-8", errors="surrogateescape")


def _resolve_vocab_file(
    vocab_file: str | os.PathLike[str] | None,
    name_or_path: str | os.PathLike[str] | None,
) -> str:
    if vocab_file is not None:
        path = Path(vocab_file)
        if path.is_file():
            return str(path)

    search_dirs: list[Path] = []
    if vocab_file is not None:
        search_dirs.append(Path(vocab_file).parent)
    if name_or_path and os.path.isdir(name_or_path):
        search_dirs.append(Path(name_or_path))
    search_dirs.append(Path.cwd())

    seen: set[Path] = set()
    for directory in search_dirs:
        directory = directory.resolve()
        if directory in seen:
            continue
        seen.add(directory)
        candidates = sorted(glob.glob(str(directory / "rwkv_vocab*.txt")))
        if len(candidates) == 1:
            return candidates[0]
        preferred = directory / "rwkv_vocab_v20230424.txt"
        if preferred.is_file():
            return str(preferred)
        if len(candidates) > 1:
            names = ", ".join(Path(path).name for path in candidates)
            raise ValueError(
                f"Multiple RWKV vocabulary files found in {directory}: {names}. "
                "Keep exactly one rwkv_vocab*.txt next to the tokenizer."
            )

    raise ValueError(
        "Can't find an RWKV vocabulary file. Expected a file named "
        f"{VOCAB_FILENAME!r} (or any rwkv_vocab*.txt) in the model directory."
    )


class RwkvTokenizer(PreTrainedTokenizer):
    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file: str | os.PathLike[str] | None = None,
        bos_token: str = EOS_TOKEN,
        eos_token: str = EOS_TOKEN,
        unk_token: str | None = None,
        pad_token: str | None = None,
        **kwargs,
    ):
        self.add_bos_token = bool(kwargs.pop("add_bos_token", True))
        resolved_vocab = _resolve_vocab_file(vocab_file, kwargs.get("name_or_path"))

        self.trie_tokenizer = RWKVByteTokenizer(resolved_vocab)
        self._idx2token = self.trie_tokenizer.idx2token
        self._token2idx = self.trie_tokenizer.token2idx

        # Public token<->id maps use strings, as required by the HF API.
        self.encoder = {
            _token_bytes_to_text(token): token_id
            for token, token_id in self._token2idx.items()
        }
        self.decoder = {
            token_id: _token_bytes_to_text(token)
            for token, token_id in self._token2idx.items()
        }

        # RWKV uses id 0 for <|endoftext|>.  The base vocabulary starts at id
        # 1, so reserve id 0 unless the vocabulary already supplies the
        # special token itself.
        self._added_tokens_decoder: dict[int, AddedToken] = {}
        if bos_token not in self.encoder and 0 not in self._idx2token:
            self._added_tokens_decoder[0] = AddedToken(
                str(bos_token), special=True
            )

        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            **kwargs,
        )

    @property
    def vocab_size(self) -> int:
        # Include the reserved id-0 <|endoftext|> token, matching the upstream
        # RWKV tokenizer (65530 entries for the 20230424 vocabulary).
        return len(self.encoder) + len(self.added_tokens_encoder)

    def get_vocab(self) -> dict[str, int]:
        vocab = dict(self.encoder)
        # Do not mutate self.encoder here: the tokenizer treats it as the
        # immutable base vocabulary.
        vocab.update(self.added_tokens_encoder)
        return dict(sorted(vocab.items(), key=lambda item: item[1]))

    def _tokenize(self, text: str, split_special_tokens: bool = False) -> list[str]:
        del split_special_tokens
        return [
            self.decoder[token_id]
            for token_id in self.trie_tokenizer.encode(text)
        ]

    def _convert_token_to_id(self, token):
        if isinstance(token, int):
            return token
        return self.encoder.get(token, self.unk_token_id)

    def _convert_id_to_token(self, index: int) -> str:
        return self.decoder.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens) -> str:
        pieces: list[bytes] = []
        for token in tokens:
            if isinstance(token, bytes):
                pieces.append(token)
            elif isinstance(token, int):
                pieces.append(self._idx2token.get(token, b""))
            else:
                pieces.append(token.encode("utf-8", errors="surrogateescape"))
        return b"".join(pieces).decode("utf-8", errors="replace")

    def save_vocabulary(
        self,
        save_directory: str,
        filename_prefix: Optional[str] = None,
    ) -> Tuple[str]:
        prefix = f"{filename_prefix}-" if filename_prefix else ""
        if os.path.isdir(save_directory):
            vocab_file = os.path.join(save_directory, prefix + VOCAB_FILENAME)
        else:
            vocab_file = prefix + save_directory

        with open(vocab_file, "w", encoding="utf-8") as writer:
            for token_id in sorted(self._idx2token):
                token = self._idx2token[token_id]
                try:
                    literal = repr(token.decode("utf-8"))
                except UnicodeDecodeError:
                    literal = repr(token)
                writer.write(f"{token_id} {literal} {len(token)}\n")
        return (vocab_file,)

    def build_inputs_with_special_tokens(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None,
    ) -> List[int]:
        bos_token_ids = [self.bos_token_id] if self.add_bos_token else []
        output = bos_token_ids + token_ids_0
        if token_ids_1 is None:
            return output
        return output + bos_token_ids + token_ids_1

    def get_special_tokens_mask(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None,
        already_has_special_tokens: bool = False,
    ) -> List[int]:
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0,
                token_ids_1=token_ids_1,
                already_has_special_tokens=True,
            )

        prefix = [1] if self.add_bos_token else []
        if token_ids_1 is None:
            return prefix + ([0] * len(token_ids_0))
        return prefix + ([0] * len(token_ids_0)) + prefix + ([0] * len(token_ids_1))

    def create_token_type_ids_from_sequences(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None,
    ) -> List[int]:
        bos_token_ids = [self.bos_token_id] if self.add_bos_token else []
        output = len(bos_token_ids + token_ids_0) * [0]
        if token_ids_1 is not None:
            output += len(bos_token_ids + token_ids_1) * [1]
        return output
