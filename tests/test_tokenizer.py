"""Golden tests for the upstream RWKV-7 tokenizer/template semantics."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hf.hf_rwkv_tokenizer import RwkvTokenizer  # noqa: E402


def make_tokenizer() -> RwkvTokenizer:
    tokenizer = RwkvTokenizer(vocab_file=ROOT / "hf" / "rwkv_vocab_v20230424.txt")
    tokenizer.chat_template = (ROOT / "hf" / "chat_template.jinja").read_text(
        encoding="utf-8"
    )
    return tokenizer


def main() -> None:
    tokenizer = make_tokenizer()
    assert tokenizer.vocab_size == 65530, tokenizer.vocab_size
    assert len(tokenizer) == 65530, len(tokenizer)
    assert (tokenizer.bos_token_id, tokenizer.eos_token_id) == (0, 0)
    assert tokenizer.pad_token_id is None
    assert tokenizer.unk_token_id is None
    assert tokenizer.encode("hello")[0] == 0
    for text in ("hello\n", "你好，世界！", "é e\u0301 \U0001f600", "a\x00b\t\r\n"):
        ids = tokenizer.encode(text, add_special_tokens=False)
        assert tokenizer.decode(ids) == text, text

    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": "A  \r\n\r\nB\t  \n\n"}],
        tokenize=False,
        add_generation_prompt=True,
    )
    assert rendered == "User✿A\nB✿\nBot✿<think", rendered

    assistant = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": "System  "},
            {"role": "user", "content": "  indented\n\nquestion  "},
            {"role": "assistant", "content": "first  \n\nsecond  "},
        ],
        tokenize=False,
        rwkv_prompt_template="assistant",
    )
    assert assistant == "System: System\n\nUser:   indented\nquestion\n\nAssistant: first\n\nsecond"

    for style, expected in (
        ("bot", "User✿Hi✿\nBot✿"),
        ("assistant", "User: Hi\n\nAssistant: "),
        ("function_calling", "### User\nHi\n### Assistant\n"),
    ):
        for options, suffix in (
            ({}, "<think"),
            ({"fake_cot_mode": False}, "<think"),
            ({"fake_cot_mode": True}, "<think></think"),
        ):
            got = tokenizer.apply_chat_template(
                [{"role": "user", "content": "Hi"}],
                tokenize=False,
                add_generation_prompt=True,
                rwkv_prompt_template=style,
                **options,
            )
            assert got == expected + suffix, (style, options, got)

    tools = [
        {
            "type": "function",
            "function": {
                "name": "weather",
                "description": "Get weather",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
            },
        }
    ]
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Paris?"}],
        tools=tools,
        tokenize=False,
        add_generation_prompt=True,
    )
    assert "### `weather`" in rendered
    assert rendered.endswith("### Assistant\n<think")
    print("PASS: official RWKV tokenizer/template goldens.")


if __name__ == "__main__":
    main()
