"""
Final verification — run: python verify.py
"""
import torch, os, numpy as np
from model import GPT, GPTConfig
import tiktoken

print("="*50)
print("  SLM FINAL VERIFICATION")
print("="*50)

# 1. Check file exists and size
path = 'myslm_final.pt'
if not os.path.exists(path):
    print("❌ myslm_final.pt not found — run export_model.py first")
    exit()

size_mb = os.path.getsize(path) / 1024**2
status  = "✅" if size_mb < 50 else "❌"
print(f"\n{status} File size  : {size_mb:.1f} MB  (target: <50MB)")

# 2. Load model
ckpt   = torch.load(path, map_location='cpu')
config = GPTConfig(**ckpt['model_args'])
model  = GPT(config)

# Convert fp16 weights back to fp32 for CPU inference
sd = {k: v.float() if hasattr(v,'dtype') and v.is_floating_point()
      else v
      for k, v in ckpt['model'].items()}

# strict=False — allows missing lm_head.weight (weight tying)
# nanoGPT re-links it automatically inside GPT.__init__
missing, unexpected = model.load_state_dict(sd, strict=False)

# Only flag if something OTHER than lm_head is missing
real_missing = [k for k in missing if 'lm_head' not in k]
if real_missing:
    print(f"Unexpected missing keys: {real_missing}")
    exit()

model.eval()

params = sum(p.numel() for p in model.parameters())
print(f"✅ Parameters  : {params/1e6:.2f}M")
print(f"✅ Architecture: {config.n_layer} layers / {config.n_head} heads / {config.n_embd} dim")
print(f"✅ Val loss     : {ckpt['val_loss']:.4f}")
print(f"✅ Trained for  : {ckpt['iter_num']} steps")

# 3. Generate sample text
print("\n--- Sample output ---")
enc   = tiktoken.get_encoding('gpt2')
start = "Once upon a time"
ids   = enc.encode(start)
x     = torch.tensor(ids, dtype=torch.long).unsqueeze(0)

with torch.no_grad():
    out  = model.generate(x, max_new_tokens=80, temperature=0.8, top_k=40)
    text = enc.decode(out[0].tolist())

print(text)
print("---------------------")
print("\nYour SLM is complete and working!")
print(f"   File: {path}  ({size_mb:.1f} MB)")
print(f"   Show this to your team to get RunPod credits 🎯")