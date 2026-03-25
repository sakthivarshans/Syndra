# Syndra

A 30.7MB language model built from scratch in one week.

No pretrained weights were downloaded. No APIs were called. No cloud was used.
Just a transformer architecture, a dataset, a 4GB GPU, and a training loop
that ran for four hours until the loss came down.

Syndra is a small language model (SLM) trained on the TinyStories dataset.
It generates coherent English stories, understands narrative structure, and
fits in a file smaller than most phone apps. It was built as a foundation
for competing in OpenAI's Parameter Golf competition — where the challenge
is to build the most capable language model that fits under 16MB.

This is version one. It works.

---

## What it does

Syndra is a next-token prediction model. Given a sequence of text, it
predicts what comes next — one token at a time. This simple objective,
trained long enough on enough data, produces a model that can:

- Continue a story from any opening line
- Generate grammatically correct English sentences
- Follow narrative patterns: introduce a character, build a situation, resolve it
- Produce original text that has never existed before

It will not answer questions, follow instructions, or hold a conversation.
It is a base language model — a foundation, not a product. The next version
will be fine-tuned for specific tasks. This version proves the foundation works.

---

## Sample output

**Prompt:** `Once upon a time`

```
Once upon a time, in a small village, there lived a little girl named Lily.
She had a small box of toys that she loved very much. Her toys were very
fragile, but she knew that her mommy and daddy would be happy.

One day, Lily's mommy and daddy went for a walk in the forest. They saw
some birds and flowers and butterflies. Lily saw a deer and said...
```

No templates. No retrieval. The model generated this from a single prompt
by predicting one token at a time, 150 times in a row.

---

## The numbers

| Property | Value |
|---|---|
| File size | **30.7 MB** |
| Parameters | **16.08 million** |
| Architecture | Transformer decoder |
| Layers | 4 |
| Attention heads | 4 |
| Embedding dimension | 256 |
| Context window | 256 tokens |
| Vocabulary size | 50,257 (GPT-2 BPE) |
| Validation loss | 1.7955 nats |
| Training steps | 10,000 |
| Training time | ~4 hours |
| GPU | NVIDIA RTX 3050 4GB VRAM |
| Dataset | TinyStories |
| Tokenizer | tiktoken gpt2 encoding |

---

## Why it exists

OpenAI runs a competition called Parameter Golf. The rules are simple:
build a language model under 16MB with the lowest possible bits-per-byte
score on enwik8 (the first 100MB of Wikipedia). The current leaderboard
leader scores 1.2244 bpb. The goal is to get as close to that as possible
inside a file smaller than most icons.

Syndra at 30.7MB is the proof-of-concept build. It demonstrates that the
full pipeline works: data preparation, tokenization, transformer training,
evaluation, and export. The next version targets 8MB by reducing the
embedding dimension from 256 to 128. The competition version targets 16MB
with better training data and longer runs.

This repository exists to document that process from the beginning —
starting from zero knowledge about language models, building one that works,
and iterating toward a competition-grade result.

---

## Setup

### Requirements

- Python 3.10 or higher
- PyTorch with CUDA (for GPU training)
- 4GB VRAM minimum for training at these settings

### Install

```bash
git clone https://github.com/YOURUSERNAME/syndra.git
cd syndra

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy transformers datasets tiktoken tqdm
```

Verify your GPU is detected:

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

---

## Reproduce from scratch

### Step 1 — Prepare training data

```bash
mkdir -p data/tinystories
python data/tinystories/prepare.py
```

Downloads TinyStories from HuggingFace and tokenizes it into binary files.
Produces `train.bin` (~400MB) and `val.bin` (~20MB).
Takes 5–10 minutes depending on internet speed.

### Step 2 — Train

```bash
python train.py config/train_myslm.py
```

Trains from scratch. Loss starts at ~10.9 (random) and descends to ~1.71
over 10,000 steps. On an RTX 3050 4GB this takes approximately four hours.

Watch for this output — it means training is working:

```
step 0:     train loss 10.932, val loss 10.918
step 250:   train loss 3.215,  val loss 3.301
step 1000:  train loss 2.401,  val loss 2.512
step 5000:  train loss 1.887,  val loss 2.063
step 10000: train loss 1.712,  val loss 1.941
```

You can stop early at 5,000 steps. The model already generates readable
stories at that point. Checkpoints are saved automatically every 250 steps.

If training is interrupted:

```bash
python train.py config/train_myslm.py --init_from=resume
```

### Step 3 — Generate text

```bash
python sample.py --out_dir=out-myslm --start="Once upon a time" --num_samples=3 --max_new_tokens=150 --device=cuda
```

### Step 4 — Evaluate

```bash
python eval_bpb.py
```

Computes the bits-per-byte score on the validation set and reports model
size, parameter count, and average loss.

### Step 5 — Export

```bash
python export_model.py
python verify.py
```

Strips optimizer state, converts weights to fp16, removes the duplicate
lm_head tensor (weight tying), and saves the clean inference model.
The training checkpoint is ~184MB. The exported model is 30.7MB.

---

## Config — what controls the size

```python
# config/train_myslm.py

n_layer  = 4      # transformer blocks stacked
n_head   = 4      # parallel attention heads per block
n_embd   = 256    # vector dimension — the primary size knob

batch_size                  = 8
gradient_accumulation_steps = 16   # effective batch size = 128
block_size                  = 256  # context window in tokens
learning_rate               = 5e-4
max_iters                   = 10000
device                      = 'cuda'
dtype                       = 'float16'
```

`n_embd` is the single most important number. Halving it from 256 to 128
reduces the model from ~30MB to ~8MB. Doubling it to 512 produces a ~120MB
model. Everything else follows from this one value.

The config is tuned for 4GB VRAM. If you have more, increase `batch_size`
and reduce `gradient_accumulation_steps` proportionally for faster training.

---

## How the model works

Syndra is a transformer decoder — the same architecture that underlies GPT.

Text enters as a sequence of token IDs produced by the GPT-2 BPE tokenizer.
Each token ID is looked up in an embedding table to produce a 256-dimensional
vector. Position embeddings are added so the model knows token order.

The combined embeddings pass through 4 transformer blocks. Each block runs
causal self-attention (every token looks at all previous tokens and decides
which are relevant) followed by a feed-forward network (each token processes
independently). Residual connections and layer normalization wrap each
operation for stable training.

After 4 blocks, a final linear projection maps the 256-dimensional
representations to logits over the 50,257-token vocabulary. The token with
appropriate probability under the distribution is sampled and appended to
the sequence. Repeat until done.

Training optimizes cross-entropy loss: at every position, the model's
probability distribution over next tokens is penalized for assigning low
probability to the token that actually came next. Over 10,000 steps and
328 million tokens, this pushes the weights toward configurations that
model English well.

---

## Repository structure

```
syndra/
├── model.py                   transformer architecture
├── train.py                   training loop
├── sample.py                  text generation
├── config/
│   └── train_myslm.py         model and training configuration
├── data/
│   └── tinystories/
│       └── prepare.py         dataset download and tokenization
├── eval_bpb.py                bits-per-byte evaluation
├── export_model.py            clean model export (strips optimizer)
├── verify.py                  final verification and sample generation
└── myslm_final.pt             trained model weights (not in repo)
```

`model.py` and `train.py` are from karpathy/nanoGPT, used as the
architectural foundation. All other files are written for this project.

---

## What comes next

| Version | Target size | Change from current |
|---|---|---|
| v1 (this) | 30.7 MB | baseline |
| v2 | ~8 MB | reduce n_embd to 128 |
| competition | <16 MB | enwik8 training + int8 quantization |

The competition build will train on enwik8 rather than TinyStories,
use a smaller vocabulary, apply quantization after training, and run
for significantly more steps on better hardware via RunPod.

---

## Credits

The transformer architecture and training code in `model.py` and `train.py`
are from **Andrej Karpathy's nanoGPT** — github.com/karpathy/nanoGPT

Karpathy built nanoGPT as a clean, readable implementation of the GPT
architecture specifically for learning. Every design decision in the code
is made for clarity over production optimization, which made it the right
foundation for someone building their first language model. The codebase
is small enough to read in an afternoon and well-structured enough to teach
every important concept in transformer training.

If you want to understand how language models work at the implementation
level — not the diagram level, the actual code level — start with nanoGPT.

Dataset: **TinyStories** by Ronen Eldan and Yuanzhi Li — huggingface.co/datasets/roneneldan/TinyStories

Tokenizer: **tiktoken** by OpenAI — github.com/openai/tiktoken

---

*Syndra is a work in progress. Version one proves the pipeline.
Version two targets the competition.*
