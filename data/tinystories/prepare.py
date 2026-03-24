"""
prepare.py — TinyStories dataset preparation for nanoGPT (16M SLM)

Outputs (saved next to this file in data/tinystories/):
    train.bin   tokenized training tokens (uint16, flat array)
    val.bin     tokenized validation tokens (uint16, flat array)
    meta.pkl    vocab_size + encoding info (read by train.py)

Run:
    python data/tinystories/prepare.py
"""

import os
import pickle
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
# Keep num_proc=1 on Windows — avoids the multiprocessing RuntimeError.
# On Linux/Mac you can safely set this to 4 for faster tokenization.
NUM_PROC       = 1
NUM_PROC_LOAD  = 1

enc = tiktoken.get_encoding("gpt2")   # GPT-2 BPE, vocab_size = 50257


def process(example):
    """Tokenize one story and append the end-of-text token."""
    ids = enc.encode_ordinary(example["text"])  # no special tokens
    ids.append(enc.eot_token)                   # 50256 = <|endoftext|>
    return {"ids": ids, "len": len(ids)}


# ── All execution inside __name__ guard ───────────────────────────────────────
# Required on Windows (spawn method). Without this you get:
#   RuntimeError: An attempt has been made to start a new process before
#   the current process has finished its bootstrapping phase.
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # 1. Download dataset ──────────────────────────────────────────────────────
    print("Downloading TinyStories dataset from HuggingFace...")
    dataset = load_dataset("roneneldan/TinyStories", num_proc=NUM_PROC_LOAD)
    print(f"Downloaded. Splits: {list(dataset.keys())}")

    # TinyStories has a 'validation' split; nanoGPT's get_batch() expects 'val'
    if "validation" in dataset:
        dataset["val"] = dataset.pop("validation")
    if "train" not in dataset or "val" not in dataset:
        raise KeyError(f"Expected 'train' and 'val' splits, got: {list(dataset.keys())}")

    for split, ds in dataset.items():
        print(f"  {split}: {len(ds):,} examples")

    # 2. Tokenize ──────────────────────────────────────────────────────────────
    print("\nTokenizing splits (num_proc=1, safe for Windows)...")
    tokenized = dataset.map(
        process,
        remove_columns=["text"],
        desc="Tokenizing",
        num_proc=NUM_PROC,
    )

    # 3. Write .bin files ──────────────────────────────────────────────────────
    # train.py reads these with:
    #   np.memmap('train.bin', dtype=np.uint16, mode='r')
    out_dir = os.path.dirname(os.path.abspath(__file__))

    for split, dset in tokenized.items():
        total_tokens = int(np.sum(dset["len"], dtype=np.uint64))
        out_path = os.path.join(out_dir, f"{split}.bin")
        print(f"\nWriting {split}.bin  ({total_tokens:,} tokens)  →  {out_path}")

        arr = np.memmap(out_path, dtype=np.uint16, mode="w+", shape=(total_tokens,))
        total_batches = min(1024, len(dset))   # shard into chunks for fast writes

        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f"  {split}.bin"):
            batch = dset.shard(
                num_shards=total_batches,
                index=batch_idx,
                contiguous=True,
            ).with_format("numpy")
            chunk = np.concatenate(batch["ids"])
            arr[idx : idx + len(chunk)] = chunk
            idx += len(chunk)

        arr.flush()
        size_mb = os.path.getsize(out_path) / 1e6
        print(f"  Done — {size_mb:.1f} MB")

    # 4. Save meta.pkl ─────────────────────────────────────────────────────────
    # train.py checks for this file to determine vocab_size.
    # Without it, train.py falls back to 50304 (harmless but imprecise).
    meta = {
        "vocab_size": enc.n_vocab,   # 50257
        "encoding":   "gpt2",
        "tokenizer":  "tiktoken",
    }
    meta_path = os.path.join(out_dir, "meta.pkl")
    with open(meta_path, "wb") as f:
        pickle.dump(meta, f)
    print(f"\nSaved meta.pkl  (vocab_size = {meta['vocab_size']})")

    # 5. Summary ───────────────────────────────────────────────────────────────
    print("\n✅ All files ready:")
    for fname in ["train.bin", "val.bin", "meta.pkl"]:
        fpath = os.path.join(out_dir, fname)
        if os.path.exists(fpath):
            print(f"   {fname}  ({os.path.getsize(fpath) / 1e6:.2f} MB)")
