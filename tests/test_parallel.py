import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Parallel (batched) inference correctness test for RWKV7 HF models."
    )
    parser.add_argument(
        "--model",
        default="/home/rwkv/jl/hf_rwkv/models/hfrwkv7-0.4b",
        help="Path to the HF model directory.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=32,
        help="Number of new tokens to generate.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model_path = args.model
    max_new_tokens = args.max_new_tokens

    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    # Decoder-only models require left padding for batched generation
    tokenizer.padding_side = "left"
    model = model.cuda()
    model.eval()

    prompts = [
        "What is a large language model?",
        "Explain quantum computing:",
        "The capital of France is",
    ]

    # Greedy generation config
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
    )

    print("=" * 60)
    print("Parallel (batched) inference correctness test")
    print(f"Model: {model_path}")
    print("=" * 60)

    # ---- Single inference ----
    single_ids = []
    single_texts = []
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([text], return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(**inputs, **gen_kwargs)
        generated = output[0][len(inputs.input_ids[0]):]
        single_ids.append(generated)
        single_texts.append(tokenizer.decode(generated, skip_special_tokens=True))

    # ---- Batch inference ----
    messages = [[{"role": "user", "content": p}] for p in prompts]
    texts = [tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages]
    inputs = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)
    print("\nBatch inputs debug:")
    print(f"  input_ids shape: {inputs.input_ids.shape}")
    print(f"  attention_mask:\n{inputs.attention_mask}")
    print(f"  lengths: {inputs.attention_mask.sum(dim=1).tolist()}")
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
    print(f"  outputs shape: {outputs.shape}")

    batch_ids = []
    batch_texts = []
    max_input_len = inputs.input_ids.shape[1]
    for i in range(outputs.shape[0]):
        # In left-padded batched generation, all outputs share the same length,
        # so new tokens start at the max input length.
        generated = outputs[i][max_input_len:]
        batch_ids.append(generated)
        batch_texts.append(tokenizer.decode(generated, skip_special_tokens=True))
        print(f"  prompt {i}: max_input_len={max_input_len}, generated_len={len(generated)}, total={outputs.shape[1]}")

    # ---- Compare ----
    print("\nSingle vs Batch consistency check:")
    all_match = True
    for i in range(len(prompts)):
        s_ids = single_ids[i]
        b_ids = batch_ids[i]
        match = torch.equal(s_ids, b_ids)
        all_match &= match
        print(f"\nPrompt {i}: {prompts[i]}")
        print(f"  Single ({len(s_ids)} tokens): {single_texts[i]!r}")
        print(f"  Batch  ({len(b_ids)} tokens): {batch_texts[i]!r}")
        print(f"  Token IDs match: {match}")

    print("\n" + "=" * 60)
    if all_match:
        print("PASS: single and batched greedy decoding produce identical token IDs.")
    else:
        print("FAIL: single and batched greedy decoding differ.")
    print("=" * 60)

    # ---- Also test sampling reproducibility with fixed seed ----
    print("\nSampling reproducibility check:")
    seed = 42
    sample_kwargs = dict(
        max_new_tokens=16,
        do_sample=True,
        temperature=0.8,
        top_p=0.9,
    )
    messages = [{"role": "user", "content": prompts[0]}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generations = []
    for run in range(2):
        torch.manual_seed(seed)
        with torch.no_grad():
            output = model.generate(**inputs, **sample_kwargs)
        generations.append(output[0][len(inputs.input_ids[0]):])

    sample_match = torch.equal(generations[0], generations[1])
    print(f"  Two runs with seed={seed}: {'match' if sample_match else 'differ'}")
    print(f"  Text: {tokenizer.decode(generations[0], skip_special_tokens=True)!r}")


if __name__ == "__main__":
    main()
