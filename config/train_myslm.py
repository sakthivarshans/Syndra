# ================================================
#  MY SLM CONFIG
#  Target: ~32MB final model
#  Hardware: RTX 3050 4GB VRAM
# ================================================

# --- Output ---
out_dir                     = 'out-myslm'
eval_interval               = 250
eval_iters                  = 50
log_interval                = 25
always_save_checkpoint      = True
init_from                   = 'scratch'

# --- Data ---
dataset                     = 'tinystories'
batch_size                  = 8      # safe for 4GB VRAM
block_size                  = 256    # context window
gradient_accumulation_steps = 16     # effective batch = 8x16 = 128

# --- Model size (controls your final MB) ---
n_layer                     = 4      # transformer layers
n_head                      = 4      # attention heads
n_embd                      = 256    # vector size — main size knob
dropout                     = 0.1
bias                        = False

# --- Training ---
learning_rate               = 5e-4
max_iters                   = 10000
lr_decay_iters              = 10000
min_lr                      = 5e-5
beta2                       = 0.99
warmup_iters                = 200
weight_decay                = 1e-1
grad_clip                   = 1.0

# --- Hardware ---
device                      = 'cuda'
dtype                       = 'float16'
compile                     = False