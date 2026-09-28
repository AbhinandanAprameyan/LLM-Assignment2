"""
train_mini_gpt.py

Assignment 2: Building a Small-Scale Foundation Model from Scratch
====================================================================

Trains the `MiniGPT` model (see `mini_gpt.py`) from scratch on next-token
prediction, reusing Assignment 1's cleaning + tokenization pipeline
(`data_collection_preprocessing.py`) so the "preprocessed dataset from
Assignment 1" requirement is satisfied by construction rather than by a
separate, disconnected data-loading step.

WHAT THIS SCRIPT DOES
----------------------
  1. Imports Assignment 1's `collect_real_datasets` / `collect_demo_corpus`,
     `clean_and_deduplicate`, and `build_tokenizer` functions and reuses
     them unchanged, so the exact same cleaning rules and tokenizer choice
     from Assignment 1 carry over here.
  2. Re-tokenizes the cleaned corpus into ONE long flat stream of token
     ids (rather than Assignment 1's independent 512-token blocks), then
     slices that stream into fixed-length (`--block-size`, 32-128 tokens)
     (input, target) pairs for next-token prediction -- this is the
     standard way to prepare a token stream for causal language modeling.
  3. Trains `MiniGPT` with a standard PyTorch loop: forward pass -> cross
     entropy loss -> backward pass -> optimizer step, tracking average
     train/validation loss and perplexity (exp(loss)) every epoch.
  4. Saves a checkpoint (`mini_gpt_checkpoint.pt`) and immediately reloads
     it into a fresh model instance to verify save/load correctness.
  5. Optionally (`--sweep`) runs several short training trials across a
     small grid of hyperparameters (learning rate, batch size, number of
     layers, embedding size) and logs the results for the report's
     "hyperparameter experiments" section.

USAGE
-----
    # One-time: copy Assignment 1's script into this folder (or point
    # --a1-script at wherever it lives).
    python train_mini_gpt.py --target-mb 20 --epochs 5

    # Offline smoke-test (no internet, reuses Assignment 1's --demo corpus):
    python train_mini_gpt.py --demo --epochs 2

    # Hyperparameter sweep (in addition to the main run):
    python train_mini_gpt.py --target-mb 20 --epochs 5 --sweep --sweep-epochs 2
"""

import argparse
import importlib.util
import json
import math
import os
import random
import time
from dataclasses import asdict
from typing import List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader

from mini_gpt import MiniGPT, MiniGPTConfig


# ===========================================================================
# Reuse Assignment 1's pipeline
# ===========================================================================

def import_assignment1(path: str):
    """Dynamically import Assignment 1's data_collection_preprocessing.py by file path."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Could not find Assignment 1's script at '{path}'. Copy "
            f"data_collection_preprocessing.py into this folder, or pass "
            f"--a1-script /path/to/data_collection_preprocessing.py"
        )
    spec = importlib.util.spec_from_file_location("a1_preprocessing", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_flat_token_stream(a1, args) -> Tuple[List[int], int, int]:
    """
    Reuse Assignment 1's collection + cleaning + tokenization stages, then
    flatten every cleaned document's tokens into one long stream (with an
    end-of-document separator token between documents) suitable for
    slicing into fixed-length next-token-prediction training examples.
    """
    use_demo = args.demo or not a1.HF_DATASETS_AVAILABLE
    if use_demo:
        docs = a1.collect_demo_corpus(repeat=args.demo_repeat)
    else:
        target_bytes = int(args.target_mb * 1024 * 1024)
        docs = a1.collect_real_datasets(target_bytes)

    # Match Assignment 1's own demo-mode relaxation of the length filter.
    effective_min_words = args.min_words
    if use_demo and args.min_words == 50:
        effective_min_words = 15
    clean_docs = a1.clean_and_deduplicate(docs, min_words=effective_min_words, lowercase=args.lowercase)
    if not clean_docs:
        raise RuntimeError("No documents survived cleaning -- try a larger --target-mb or --demo-repeat.")

    tokenizer = a1.build_tokenizer(args.tokenizer)
    if tokenizer is None:  # Assignment 1's fallback tokenizer needs a vocab built from this corpus
        tokenizer = a1.FallbackTokenizer()
        tokenizer.build_vocab(d.text for d in clean_docs)

    eos_id = getattr(tokenizer, "eos_token_id", None)
    if eos_id is None:
        eos_id = getattr(tokenizer, "pad_token_id", 0) or 0

    flat_ids: List[int] = []
    for doc in clean_docs:
        flat_ids.extend(tokenizer.encode(doc.text))
        flat_ids.append(eos_id)  # simple document separator

    vocab_size = len(tokenizer) if hasattr(tokenizer, "__len__") else (max(flat_ids) + 1)
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = 0

    print(
        f"Built flat token stream: {len(flat_ids):,} tokens from {len(clean_docs):,} "
        f"cleaned documents (vocab_size={vocab_size})"
    )
    return flat_ids, vocab_size, pad_id


# ===========================================================================
# Dataset
# ===========================================================================

class SequenceDataset(Dataset):
    """
    Wraps a flat list of token ids into fixed-length (block_size) (input,
    target) pairs for next-token prediction: target[t] = input[t+1].
    Non-overlapping windows are used (simple and fast); a sliding window
    with stride < block_size would yield more training examples at the
    cost of more redundancy, and is a straightforward extension.
    """

    def __init__(self, token_ids: List[int], block_size: int):
        self.data = token_ids
        self.block_size = block_size

    def __len__(self) -> int:
        return max(0, (len(self.data) - 1) // self.block_size)

    def __getitem__(self, i: int):
        start = i * self.block_size
        chunk = self.data[start: start + self.block_size + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


# ===========================================================================
# Training / evaluation
# ===========================================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_perplexity(loss: float) -> float:
    """exp(loss) can overflow for a very poorly trained / early model; cap it for readability."""
    return math.exp(min(loss, 20.0))


@torch.no_grad()
def evaluate(model: MiniGPT, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total_loss, n_batches = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        _, loss = model(x, targets=y)
        total_loss += loss.item()
        n_batches += 1
    model.train()
    return total_loss / n_batches if n_batches else float("nan")


def train_one_config(args, flat_ids, vocab_size, pad_id, device, tag: str, verbose: bool = True):
    n_val = max(args.block_size + 1, int(0.1 * len(flat_ids)))
    train_ids, val_ids = flat_ids[:-n_val], flat_ids[-n_val:]

    train_loader = DataLoader(
        SequenceDataset(train_ids, args.block_size),
        batch_size=args.batch_size, shuffle=True, drop_last=True,
    )
    val_loader = DataLoader(
        SequenceDataset(val_ids, args.block_size),
        batch_size=args.batch_size, shuffle=False, drop_last=True,
    )
    if len(train_loader) == 0 or len(val_loader) == 0:
        raise RuntimeError(
            "Not enough tokens to form even one training/validation batch "
            f"at block_size={args.block_size}, batch_size={args.batch_size}. "
            "Increase --target-mb / --demo-repeat, or decrease --block-size / --batch-size."
        )

    config = MiniGPTConfig(
        vocab_size=vocab_size, block_size=args.block_size, n_layer=args.n_layer,
        n_head=args.n_head, n_embd=args.n_embd, dropout=args.dropout, pad_token_id=pad_id,
    )
    model = MiniGPT(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        total_loss, n_batches = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            _, loss = model(x, targets=y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        train_loss = total_loss / n_batches
        val_loss = evaluate(model, val_loader, device)
        entry = {
            "tag": tag,
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_ppl": safe_perplexity(train_loss),
            "val_ppl": safe_perplexity(val_loss),
            "elapsed_seconds": round(time.time() - t0, 2),
        }
        history.append(entry)
        if verbose:
            print(
                f"[{tag}] epoch {epoch}/{args.epochs} | "
                f"train_loss {train_loss:.4f} (ppl {entry['train_ppl']:.2f}) | "
                f"val_loss {val_loss:.4f} (ppl {entry['val_ppl']:.2f}) | "
                f"{entry['elapsed_seconds']:.1f}s"
            )
    return model, config, history


# ===========================================================================
# Checkpoint save / load
# ===========================================================================

def save_checkpoint(path: str, model: MiniGPT, config: MiniGPTConfig, history, extra: dict = None) -> None:
    payload = {
        "model_state_dict": model.state_dict(),
        "config": asdict(config),
        "history": history,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_checkpoint(path: str, device: torch.device) -> Tuple[MiniGPT, dict]:
    payload = torch.load(path, map_location=device)
    config = MiniGPTConfig(**payload["config"])
    model = MiniGPT(config).to(device)
    model.load_state_dict(payload["model_state_dict"])
    return model, payload


# ===========================================================================
# Main
# ===========================================================================

SWEEP_GRID = [
    {"lr": 1e-3, "batch_size": 16, "n_layer": 1, "n_embd": 64},
    {"lr": 1e-3, "batch_size": 32, "n_layer": 2, "n_embd": 128},
    {"lr": 5e-4, "batch_size": 32, "n_layer": 2, "n_embd": 128},
    {"lr": 5e-4, "batch_size": 64, "n_layer": 1, "n_embd": 256},
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--a1-script", type=str, default="./data_collection_preprocessing.py",
                   help="Path to Assignment 1's data_collection_preprocessing.py")
    p.add_argument("--demo", action="store_true", help="Use Assignment 1's offline synthetic corpus instead of real data.")
    p.add_argument("--demo-repeat", type=int, default=20, help="Repeat factor for the demo corpus (more repeats = more tokens).")
    p.add_argument("--target-mb", type=float, default=20.0, help="Target raw text size in MB for the real collection run.")
    p.add_argument("--tokenizer", type=str, default="gpt2")
    p.add_argument("--min-words", type=int, default=50)
    p.add_argument("--lowercase", action="store_true")

    p.add_argument("--block-size", type=int, default=64, help="Sequence length per training example (assignment: 32-128).")
    p.add_argument("--n-layer", type=int, default=2, help="Number of transformer blocks (assignment: 1-2).")
    p.add_argument("--n-head", type=int, default=4, help="Number of attention heads (assignment: 2-4).")
    p.add_argument("--n-embd", type=int, default=128, help="Embedding dimension (assignment: 64-256).")
    p.add_argument("--dropout", type=float, default=0.1)

    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--sweep", action="store_true", help="Also run a small hyperparameter sweep.")
    p.add_argument("--sweep-epochs", type=int, default=2, help="Epochs per sweep trial (kept short).")

    p.add_argument("--out-dir", type=str, default="./output2")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    a1 = import_assignment1(args.a1_script)
    flat_ids, vocab_size, pad_id = build_flat_token_stream(a1, args)

    # ---- Optional hyperparameter sweep -------------------------------
    if args.sweep:
        sweep_results = []
        for i, cfg in enumerate(SWEEP_GRID):
            tag = f"sweep_{i}"
            sweep_args = argparse.Namespace(**vars(args))
            sweep_args.lr, sweep_args.batch_size = cfg["lr"], cfg["batch_size"]
            sweep_args.n_layer, sweep_args.n_embd = cfg["n_layer"], cfg["n_embd"]
            sweep_args.epochs = args.sweep_epochs
            model, config, history = train_one_config(args=sweep_args, flat_ids=flat_ids, vocab_size=vocab_size,
                                                        pad_id=pad_id, device=device, tag=tag)
            final = history[-1]
            sweep_results.append({
                **cfg,
                "epochs": args.sweep_epochs,
                "final_train_loss": final["train_loss"],
                "final_val_loss": final["val_loss"],
                "final_val_ppl": final["val_ppl"],
                "num_parameters": model.num_parameters(),
            })
        sweep_path = os.path.join(args.out_dir, "hyperparam_sweep.json")
        with open(sweep_path, "w") as f:
            json.dump(sweep_results, f, indent=2)
        print(f"Sweep results written to {sweep_path}")

    # ---- Main training run --------------------------------------------
    model, config, history = train_one_config(
        args=args, flat_ids=flat_ids, vocab_size=vocab_size, pad_id=pad_id, device=device, tag="main",
    )

    ckpt_path = os.path.join(args.out_dir, "mini_gpt_checkpoint.pt")
    save_checkpoint(ckpt_path, model, config, history, extra={
        "flat_token_stream_length": len(flat_ids),
        "vocab_size": vocab_size,
    })
    print(f"Saved checkpoint to {ckpt_path}")

    # ---- Verify save/load correctness ----------------------------------
    reloaded_model, payload = load_checkpoint(ckpt_path, device)
    val_loader = DataLoader(
        SequenceDataset(flat_ids[-max(args.block_size + 1, int(0.1 * len(flat_ids))):], args.block_size),
        batch_size=args.batch_size, shuffle=False, drop_last=True,
    )
    reloaded_val_loss = evaluate(reloaded_model, val_loader, device)
    print(
        f"Checkpoint reload check: val_loss immediately after reload = {reloaded_val_loss:.4f} "
        f"(should match the final logged val_loss {history[-1]['val_loss']:.4f})"
    )

    metrics_path = os.path.join(args.out_dir, "metrics_log.json")
    with open(metrics_path, "w") as f:
        json.dump({
            "history": history,
            "config": asdict(config),
            "num_parameters": model.num_parameters(),
            "flat_token_stream_length": len(flat_ids),
            "device": str(device),
            "reload_check_val_loss": reloaded_val_loss,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
        }, f, indent=2)
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
