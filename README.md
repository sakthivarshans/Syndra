# Syndra — A 30.7MB Language Model Built in One Day

> *"I had zero knowledge about language models. I built one anyway."*

Built from scratch in under 24 hours on an RTX 3050 4GB GPU.
No pretrained weights. No API calls. Just a transformer, a dataset, and a lot of loss going down.

---

## What this is

This is a small language model (SLM) trained entirely from scratch on the TinyStories dataset. It generates grammatically correct English stories, fits in 30.7MB, and was built as a first step toward competing in OpenAI's **Parameter Golf competition** — where the goal is to build the most capable language model under 16MB.

This repository documents the full pipeline: data preparation → tokenization → transformer training → evaluation → export. Every file here was written by hand. The architecture is based on nanoGPT but the config, training setup, evaluation scripts, and export pipeline are custom-built for this size target.

---

## The numbers

| Metric | Value |
|---|---|
| **Final model size** | 30.7 MB |
| **Parameters** | 16.08 million |
| **Architecture** | 4 layers / 4 heads / 256 dim |
| **Val loss** | 1.7955 nats |
| **Training steps** | 10,000 |
| **Training time** | ~6 hours |
| **GPU** | RTX 3050 4GB VRAM |
| **Dataset** | TinyStories (roneneldan/TinyStories) |
| **Tokenizer** | GPT-2 BPE (tiktoken, vocab 50,257) |

---

## Sample output

Prompt: `"Once upon a time"`

```
Once upon a time, in a small village, there lived a little girl named Lily.
She had a small box of toys that she loved very much. Her toys were very
fragile, but she knew that her mommy and daddy would be happy.

One day, Lily's mommy and daddy went for a walk in the forest. They saw
some birds and flowers and butterflies. Lily saw a deer and said...
```

No fine-tuning. No RLHF. Just a transformer that learned story structure
from data — character introduction, setting, conflict, action — purely from
predicting the next token 328 million times.

---

## Project structure

```
nanoGPT/
│
├── model.py                        # transformer architecture (nanoGPT)
├── train.py                        # training loop (nanoGPT)
├── sample.py                       # generation script (nanoGPT)
│
├── config/
│   └── train_myslm.py              # ← my custom config (4L/4H/256D)
│
├── data/
│   └── tinystories/
│       └── prepare.py              # ← download + tokenize dataset
│
├── eval_bpb.py                     # ← compute bits-per-byte score
├── export_model.py                 # ← strip optimizer, convert to fp16
├── generate.py                     # ← clean generation script
└── verify.py                       # ← final model verification
```

Files marked ← were written for this project specifically.
`model.py`, `train.py`, `sample.py` are from karpathy/nanoGPT unchanged.

---

## How to reproduce this

### 1. Clone and install

```bash
git clone https://github.com/YOURUSERNAME/YOURREPO.git
cd YOURREPO

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy transformers datasets tiktoken tqdm
```

### 2. Verify your GPU

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

### 3. Prepare the dataset

```bash
mkdir -p data/tinystories
python data/tinystories/prepare.py
```

This downloads TinyStories from HuggingFace and tokenizes it into
`train.bin` (~400MB) and `val.bin` (~20MB).

### 4. Train

```bash
python train.py config/train_myslm.py
```

Watch the loss drop. Step 0 starts at ~10.9. By step 10,000 it's at ~1.71.
That descent is the model learning English.

On an RTX 3050 4GB this takes about 4 hours. You can stop at 5,000 steps
and it already generates readable text.

If you hit CUDA out of memory, open `config/train_myslm.py` and change:
```python
batch_size = 4                  # was 8
gradient_accumulation_steps = 32  # was 16
```

### 5. Generate text

```bash
python generate.py
```

### 6. Evaluate bpb score

```bash
python eval_bpb.py
```

### 7. Export clean model (~30MB)

```bash
python export_model.py
python verify.py
```

---

## The config that hits 30.7MB

```python
# config/train_myslm.py

n_layer  = 4      # transformer layers
n_head   = 4      # attention heads per layer
n_embd   = 256    # vector dimension — the main size knob

batch_size                  = 8
gradient_accumulation_steps = 16   # effective batch = 128
block_size                  = 256  # context window
learning_rate               = 5e-4
max_iters                   = 10000
device                      = 'cuda'
dtype                       = 'float16'
```

The math: `(vocab × n_embd × 2) + (n_layer × 8 × n_embd²)` ≈ 16M params.
At fp16 (2 bytes per param) that's ~32MB before PyTorch file overhead.
Weight tying (sharing the embedding and output head) saves another ~24MB
that would otherwise be duplicated in the checkpoint.

---

## What I learned building this

**Day 0:** No knowledge of how language models work internally.

**Day 1:** Built a working transformer from scratch.

Along the way:
- How attention actually works — Q, K, V projections, causal masking
- Why residual connections exist (without them, gradients vanish at depth)
- What loss in nats means and how it maps to bits-per-byte
- Why the embedding table dominates parameter count at small vocab sizes
- How weight tying works and why it matters for file size
- The difference between a training checkpoint (184MB) and an inference model (30.7MB)
- Why `batch_size × gradient_accumulation_steps` is what actually matters, not either alone

---

## What comes next

This is version 1. The competition target is **under 16MB** with the best
possible bits-per-byte score on enwik8 (Wikipedia benchmark).

The path from here:

| Step | Change | Expected size |
|---|---|---|
| This repo | n_embd=256, TinyStories | 30.7 MB |
| Next | n_embd=128, enwik8 training | ~8 MB |
| Competition | int8 quant + better tokenizer | <16 MB |

---

## Hardware

Trained on a laptop GPU — RTX 3050 4GB VRAM. No cloud. No expensive setup.
The config is specifically tuned for 4GB VRAM constraints:

- `dtype=float16` halves memory vs float32
- `batch_size=8` keeps activations within 4GB
- `gradient_accumulation_steps=16` recovers the effective batch size

Total VRAM used during training: ~1.2GB of the available 4GB.

---

## Based on

Architecture from [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT).
Dataset: [roneneldan/TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories).
Tokenizer: [tiktoken](https://github.com/openai/tiktoken) gpt2 encoding.

---

