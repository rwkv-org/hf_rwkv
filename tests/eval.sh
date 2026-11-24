HF_ENDPOINT="https://hf-mirror.com" lm-eval run --model hf \
    --model_args "pretrained=/home/rwkv/jl/hf_rwkv/models/rwkv7-g1g-1.5b-hf,dtype=bfloat16,trust_remote_code=True" \
    --tasks gsm8k \
    --num_fewshot 0 \
    --device cuda:2 \
    --cache_requests true \
    --output_path .output \
    --log_samples