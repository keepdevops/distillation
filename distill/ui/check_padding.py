from mlx_lm import load

model_path = "./airgap_bundle/hf_cache/models--Qwen--Qwen2-1.5B-Instruct/snapshots/ba1cf1846d7df0a0591d6c00649f57e798519da8"

model, tokenizer = load(model_path)

print("After load:")
print("  padding_side:", tokenizer.padding_side)
print("  pad_token:", tokenizer.pad_token)
print("  eos_token:", tokenizer.eos_token)
print("  pad_token_id:", tokenizer.pad_token_id)

# Force it
tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("\nAfter force:")
print("  padding_side:", tokenizer.padding_side)
print("  pad_token:", tokenizer.pad_token)
print("  pad_token_id:", tokenizer.pad_token_id)
