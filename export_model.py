"""
Export clean model — removes duplicate lm_head weight,
forces fp16. Run: python export_model.py
"""
import torch
import os

CKPT_PATH = 'out-myslm/ckpt.pt'
OUT_PATH  = 'myslm_final.pt'

print(f"Loading {CKPT_PATH} ...")
ckpt = torch.load(CKPT_PATH, map_location='cpu')

# Force ALL floating point tensors to fp16
print("Converting ALL weights to fp16 ...")
model_state = {}
for k, v in ckpt['model'].items():
    if hasattr(v, 'dtype') and v.is_floating_point():
        model_state[k] = v.half()
    else:
        model_state[k] = v

# ── THE KEY FIX ───────────────────────────────────────
# lm_head.weight and transformer.wte.weight are the same
# tensor (weight tying) but PyTorch saved them separately.
# Remove lm_head.weight — it gets re-linked automatically
# when GPT.__init__ runs: self.lm_head.weight = self.wte.weight
if 'lm_head.weight' in model_state:
    del model_state['lm_head.weight']
    print("Removed duplicate lm_head.weight (weight tying)")
# ─────────────────────────────────────────────────────

val_loss = ckpt.get('best_val_loss', ckpt.get('val_loss', 0.0))

clean = {
    'model'      : model_state,
    'model_args' : ckpt['model_args'],
    'config'     : ckpt['config'],
    'iter_num'   : ckpt['iter_num'],
    'val_loss'   : val_loss,
}

torch.save(clean, OUT_PATH)

raw_mb   = os.path.getsize(CKPT_PATH) / 1024**2
clean_mb = os.path.getsize(OUT_PATH)  / 1024**2

print(f"\nTraining checkpoint : {raw_mb:.1f} MB")
print(f"Exported model      : {clean_mb:.1f} MB")
print(f"Val loss            : {val_loss:.4f}")

if clean_mb < 50:
    print(f"\nSUCCESS — {clean_mb:.1f}MB is under 50MB target!")
    print(f"   Saved to: {OUT_PATH}")
else:
    print(f"\nStill {clean_mb:.1f}MB — paste output and ask for help")