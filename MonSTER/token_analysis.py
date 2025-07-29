#!/usr/bin/env python
"""token_analysis.py

Utility script to inspect a saved Hierarchical Reasoning Model (or any
PyTorch checkpoint) and report the size of its token‑embedding table
(number of tokens × embedding dimension).

Usage (from repo root):
    python token_analysis.py checkpoints/HRM-checkpoint-ARC-2

Optional flags:
    --ckpt-file   Name of the checkpoint file inside the directory
                  (default: "checkpoint").

The script is intentionally lightweight: it does **not** build the full
model architecture.  It simply loads the checkpoint in CPU memory,
searches the state‑dict for the first 2‑D tensor whose name contains
"emb" (common for token embeddings), and prints its shape.

If your project uses a different naming convention, adjust the
`_is_embedding()` helper below.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Tuple

import torch


def _is_embedding(name: str, tensor: torch.Tensor) -> bool:
    """Heuristic to decide whether a parameter is a token‑embedding table."""
    return tensor.dim() == 2 and "emb" in name.lower()


def _find_embedding(state_dict: dict[str, torch.Tensor]) -> Tuple[str, torch.Tensor] | Tuple[None, None]:
    """Return (param_name, weight) for the first embedding found, else (None, None)."""
    for name, weight in state_dict.items():
        if _is_embedding(name, weight):
            return name, weight
    return None, None


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect token embedding dimensions in a checkpoint.")
    parser.add_argument("ckpt_dir", type=Path, help="Directory that contains the checkpoint file (and all_config.yaml)")
    parser.add_argument("--ckpt-file", default="checkpoint", help="Checkpoint filename inside ckpt_dir (default: 'checkpoint')")
    args = parser.parse_args()

    ckpt_path = args.ckpt_dir / args.ckpt_file
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"Loading checkpoint → {ckpt_path} (CPU)…")
    ckpt = torch.load(ckpt_path, map_location="cpu")

    # Determine the state‑dict.
    if isinstance(ckpt, dict):
        if "model" in ckpt and isinstance(ckpt["model"], dict):
            state_dict = ckpt["model"]
        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:  # fallback: maybe the whole object *is* the state‑dict
            state_dict = ckpt  # type: ignore[assignment]
    else:
        raise RuntimeError("Unsupported checkpoint format – expected a dict with 'model' or 'state_dict'.")

    name, emb_weight = _find_embedding(state_dict)
    if emb_weight is None:
        print("❌ No 2‑D embedding tensor found. Adjust `_is_embedding()` for your project’s naming scheme.")
        return

    vocab_size, emb_dim = emb_weight.shape
    print("\n✅ Embedding tensor found →", name)
    print(f"   Tokens (vocab size) : {vocab_size}")
    print(f"   Embedding dimension : {emb_dim}\n")


if __name__ == "__main__":
    main()
