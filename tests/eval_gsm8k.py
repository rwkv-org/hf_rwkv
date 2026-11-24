#!/usr/bin/env python
"""Evaluate a HuggingFace RWKV7 model on GSM8K using lm-evaluation-harness."""

import argparse
import json
from pathlib import Path

from lm_eval import evaluator


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run GSM8K evaluation on a RWKV7 HF model."
    )
    parser.add_argument(
        "--model",
        default="/home/rwkv/jl/hf_rwkv/models/hfrwkv7-0.4b",
        help="Path to the HF model directory.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for evaluation.",
    )
    parser.add_argument(
        "--num-fewshot",
        type=int,
        default=0,
        help="Number of few-shot examples for GSM8K.",
    )
    parser.add_argument(
        "--limit",
        type=float,
        default=None,
        help="Limit the number of examples (e.g. 100 or 0.1 for 10%%).",
    )
    parser.add_argument(
        "--output-dir",
        default="results/gsm8k",
        help="Directory to save evaluation results.",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Model dtype for evaluation.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device to run evaluation on.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    model_args = (
        f"pretrained={args.model},"
        f"dtype={args.dtype},"
        f"trust_remote_code=True"
    )

    print("=" * 60)
    print("GSM8K Evaluation")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Model args: {model_args}")
    print(f"Batch size: {args.batch_size}")
    print(f"Num fewshot: {args.num_fewshot}")
    print(f"Limit: {args.limit}")
    print(f"Device: {args.device}")
    print("=" * 60)

    results = evaluator.simple_evaluate(
        model="hf",
        model_args=model_args,
        tasks=["gsm8k"],
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        device=args.device,
        limit=args.limit,
        confirm_run_unsafe_code=True,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to {output_file}")

    # Print summary
    gsm8k_results = results.get("results", {}).get("gsm8k", {})
    print("\nGSM8K Summary:")
    for metric, value in gsm8k_results.items():
        if isinstance(value, (int, float)):
            print(f"  {metric}: {value:.4f}")
        else:
            print(f"  {metric}: {value}")


if __name__ == "__main__":
    main()
