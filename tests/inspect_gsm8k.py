#!/usr/bin/env python
"""Inspect RWKV7 responses on GSM8K samples in real time."""

import argparse
import json
import os
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inspect RWKV7 responses on GSM8K samples."
    )
    parser.add_argument(
        "--model",
        default="/home/rwkv/jl/hf_rwkv/models/rwkv7-g1g-1.5b-hf",
        help="Path to the HF model directory.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Number of GSM8K samples to inspect.",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Start from this sample index.",
    )
    parser.add_argument(
        "--output-dir",
        default=".outputs",
        help="Directory to save inspection results.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Max new tokens to generate.",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Model dtype.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device to run on.",
    )
    return parser.parse_args()


def build_prompt(question: str) -> str:
    return f"User✿{question}\n✿Bot✿"


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "gsm8k_inspect.jsonl"

    dtype = getattr(torch, args.dtype)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        dtype=dtype,
    ).to(args.device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.padding_side = "left"

    dataset = load_dataset("openai/gsm8k", "main", split="test")

    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        use_cache=True,
    )

    print("=" * 80)
    print("GSM8K Real-time Response Inspection")
    print(f"Model: {args.model}")
    print(f"Output: {output_file}")
    print("=" * 80)

    with open(output_file, "w", encoding="utf-8", buffering=1) as f:
        for i in range(args.offset, min(args.offset + args.num_samples, len(dataset))):
            item = dataset[i]
            question = item["question"]
            answer = item["answer"]

            prompt = build_prompt(question)
            inputs = tokenizer([prompt], return_tensors="pt").to(model.device)

            with torch.no_grad():
                output = model.generate(**inputs, **gen_kwargs)

            response = tokenizer.decode(output[0][len(inputs.input_ids[0]):], skip_special_tokens=True)

            record = {
                "index": i,
                "question": question,
                "expected": answer,
                "prompt": prompt,
                "response": response.strip(),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())

            print(f"\n--- Sample {i} ---")
            print(f"Question: {question}")
            print(f"Expected: {answer}")
            print(f"Response: {response.strip()}")
            print("-" * 80)

    print(f"\nDone. Results saved to {output_file}")


if __name__ == "__main__":
    main()
