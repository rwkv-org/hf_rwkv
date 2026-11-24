from transformers import AutoModelForCausalLM, AutoTokenizer
model_path = '/home/rwkv/jl/hf_rwkv/models/rwkv7-g1g-1.5b-hf'
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, dtype="bfloat16")
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = model.cuda() # Supported on Nvidia/AMD/Intel eg. model.xpu()
prompt = "What is a large language model?"
messages = [
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
print(f"Prompt: {prompt}")
print(f"Text: {text}")
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
print(model_inputs)
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=1000,
    do_sample=True,
    temperature=1.0,
    top_p=0.3,
    repetition_penalty=1.2
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]
print(f"Generated IDs: {generated_ids}")
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)[0]
print(response)
