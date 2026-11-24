import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = '/home/rwkv/jl/hf_rwkv/models/hfrwkv7-0.4b'
prompt = "What is a large language model?"
max_new_tokens = 256
dtype = torch.bfloat16
num_runs = 5
warmup_runs = 3


def make_inputs(tokenizer, device):
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return tokenizer([text], return_tensors="pt").to(device)


def benchmark_generate(model, inputs, label):
    """Run generation multiple times and report avg time & throughput."""
    device = inputs.input_ids.device
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
    )

    # Warmup
    for i in range(warmup_runs):
        torch.cuda.synchronize(device)
        with torch.no_grad():
            _ = model.generate(**inputs, **gen_kwargs)
        torch.cuda.synchronize(device)
        print(f"  [{label}] warmup {i+1}/{warmup_runs} done")

    # Benchmark
    times = []
    for i in range(num_runs):
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        with torch.no_grad():
            _ = model.generate(**inputs, **gen_kwargs)
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        print(f"  [{label}] run {i+1}/{num_runs}: {elapsed:.3f}s")

    avg = sum(times) / len(times)
    tps = max_new_tokens / avg
    return avg, tps


def main():
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    print("=" * 60)
    print("torch.compile speed comparison for RWKV7")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Dtype: {dtype}")
    print(f"Max new tokens: {max_new_tokens}")
    print(f"Benchmark runs: {num_runs} (warmup: {warmup_runs})")
    print()

    # Eager
    print("Loading eager model...")
    model_eager = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        dtype=dtype,
    ).cuda()
    model_eager.eval()
    inputs = make_inputs(tokenizer, model_eager.device)

    print("Benchmarking eager mode...")
    eager_time, eager_tps = benchmark_generate(model_eager, inputs, "Eager")

    del model_eager
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    # Compile
    print("\nLoading model for torch.compile...")
    model_compile = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        dtype=dtype,
    ).cuda()
    model_compile.eval()
    print("Compiling model (this may take a while on first warmup)...")
    model_compile = torch.compile(model_compile, mode="reduce-overhead")

    print("Benchmarking compiled mode...")
    compile_time, compile_tps = benchmark_generate(model_compile, inputs, "Compile")

    # Report
    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    print(f"Eager:   {eager_time:.3f}s  ({eager_tps:.1f} tokens/sec)")
    print(f"Compile: {compile_time:.3f}s  ({compile_tps:.1f} tokens/sec)")
    if compile_time < eager_time:
        print(f"Speedup: {eager_time / compile_time:.2f}x")
    else:
        print(f"Slowdown: {compile_time / eager_time:.2f}x")


if __name__ == "__main__":
    main()
