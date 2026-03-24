"""
Compute bits-per-byte (bpb) score on validation set.
Run: python eval_bpb.py
"""
import torch
import numpy as np
import os
from model import GPT, GPTConfig

# ── Config ───────────────────────────────────────────
CKPT_PATH = 'out-myslm/ckpt.pt'
VAL_PATH  = 'data/tinystories/val.bin'
N_BATCHES = 200    # how many chunks to evaluate on
DEVICE    = 'cuda' if torch.cuda.is_available() else 'cpu'

# ── Load model ───────────────────────────────────────
print(f"Loading model from {CKPT_PATH} ...")
checkpoint  = torch.load(CKPT_PATH, map_location=DEVICE)
model_args  = checkpoint['model_args']
config      = GPTConfig(**model_args)
model       = GPT(config)
model.load_state_dict(checkpoint['model'])
model.eval()
model.to(DEVICE)

total_params = sum(p.numel() for p in model.parameters())
print(f"Parameters : {total_params/1e6:.2f}M")

# ── Load validation data ─────────────────────────────
print(f"Loading val data from {VAL_PATH} ...")
raw   = np.memmap(VAL_PATH, dtype=np.uint16, mode='r')
data  = torch.from_numpy(np.array(raw, dtype=np.int64))

# ── Evaluate loss ────────────────────────────────────
block_size = config.block_size
total_loss = 0.0

print(f"Evaluating on {N_BATCHES} batches ...")
with torch.no_grad():
    for i in range(N_BATCHES):
        start = i * block_size
        x = data[start : start + block_size].unsqueeze(0).to(DEVICE)
        y = data[start+1 : start + block_size + 1].unsqueeze(0).to(DEVICE)
        _, loss = model(x, y)
        total_loss += loss.item()

avg_loss = total_loss / N_BATCHES

# bpb = avg_loss_in_nats / ln(2) / avg_bytes_per_token
# GPT-2 tokenizer: ~1 token = ~4 bytes on English text
bpb = avg_loss / (np.log(2) * 4.0)

# ── File size ────────────────────────────────────────
ckpt_mb = os.path.getsize(CKPT_PATH) / 1024**2

print("\n" + "="*45)
print(f"  Parameters     : {total_params/1e6:.2f}M")
print(f"  Checkpoint size: {ckpt_mb:.1f} MB  (includes optimizer)")
print(f"  Avg loss (nats): {avg_loss:.4f}")
print(f"  Bits per byte  : {bpb:.4f}")
print(f"  OpenAI leader  : 1.2244  (lower = better)")
print("="*45)