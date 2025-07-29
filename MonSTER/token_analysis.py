#!/usr/bin/env python
"""
token_analysis.py  —  inspect every token‑embedding matrices in an HRM checkpoint
───────────────────────────────────────────────────────────────────────────────
ARC‑HRM uses multiple vocabularies:

• grid‑cell / colour tokens            → embed_tokens.embedding_weight
• puzzle‑ID (task) tokens              → puzzle_emb.weights
• optional special or project‑specific → e.g. halt_emb.weight, pos_emb.weight

This script loads the checkpoint on CPU, finds **every** 2‑D tensor that
looks like an embedding table, prints shape + a short guess at its role,
and then dumps the actual values (rounded) for the 12 vocab tokens and
12 random puzzle tokens, in rows of 32 dimensions each, formatted with
two decimal places and explicit +/– signs.
"""

from __future__ import annotations

import argparse
import random
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch

###############################################################################
# Helper rules to label common HRM embedding tensors.
###############################################################################
NAME_RULES: List[Tuple[re.Pattern[str], str]] = [
    (re.compile(r"puzzle[_.]?emb", re.I), "Puzzle‑ID tokens"),
    (re.compile(r"embed_tokens|grid|colour", re.I), "Grid/colour tokens"),
    (re.compile(r"halt[_.]?emb", re.I), "Halt‑flag tokens"),
    (re.compile(r"pos[_.]?emb|position", re.I), "Positional embeddings"),
]


def classify(name: str) -> str:
    for pat, label in NAME_RULES:
        if pat.search(name):
            return label
    return "Unclassified‑embedding"


def load_state_dict(ckpt_path: Path) -> Dict[str, torch.Tensor]:
    obj = torch.load(ckpt_path, map_location="cpu")
    if isinstance(obj, dict):
        for key in ("model", "state_dict", "module"):
            if key in obj and isinstance(obj[key], dict):
                return obj[key]
        return obj
    raise RuntimeError("Unsupported checkpoint format (expected dict).")


def find_embeddings(
    state_dict: Dict[str, torch.Tensor]
) -> List[Tuple[str, torch.Tensor]]:
    out: List[Tuple[str, torch.Tensor]] = []
    for name, tensor in state_dict.items():
        if tensor.ndim == 2 and any(tag in name.lower() for tag in ("emb", "token")):
            out.append((name, tensor))
    return out


def print_embeddings(
    emb: torch.Tensor,
    puzzle_emb: torch.Tensor,
    chunk: int = 32,
    num_random_puzzle: int = 12,
    seed: int = 0
) -> None:
    """
    Print each of the 12 vocab embeddings and `num_random_puzzle` random
    puzzle embeddings, rounding to 2 dp with explicit +/– signs and slicing
    into lines of `chunk` dims each.
    """
    # build labels for the 12 grid tokens:
    labels = {0: "PAD", 1: "EOS", **{i+2: f"Colour {i}" for i in range(10)}}
    random.seed(seed)

    print("\n=== Vocabulary embeddings (12 tokens) ===")
    for idx, name in labels.items():
        vec = emb[idx].cpu().tolist()
        print(f"\nToken {idx:2d} → {name}")
        for i in range(0, len(vec), chunk):
            line = vec[i : i + chunk]
            print(" ".join(f"{v:+.2f}" for v in line))

    print(f"\n=== {num_random_puzzle} Random puzzle embeddings ===")
    max_id = puzzle_emb.shape[0]
    picks = random.sample(range(max_id), num_random_puzzle)
    for idx in picks:
        vec = puzzle_emb[idx].cpu().tolist()
        print(f"\nPuzzle token {idx}")
        for i in range(0, len(vec), chunk):
            line = vec[i : i + chunk]
            print(" ".join(f"{v:+.2f}" for v in line))


def main() -> None:
    ap = argparse.ArgumentParser(
        description="List and dump token‑embedding matrices in an HRM checkpoint."
    )
    ap.add_argument(
        "ckpt_dir", type=Path, help="Folder that contains the checkpoint file"
    )
    ap.add_argument(
        "--ckpt-file",
        default="checkpoint",
        help="Checkpoint filename inside ckpt_dir (default: 'checkpoint')",
    )
    ap.add_argument(
        "--dump",
        action="store_true",
        help="If set, print the formatted values for vocab + puzzle embeddings",
    )
    args = ap.parse_args()

    ckpt_path = args.ckpt_dir / args.ckpt_file
    if not ckpt_path.exists():
        sys.exit(f"❌  Checkpoint file not found: {ckpt_path}")

    print(f"🔍  Loading checkpoint → {ckpt_path} (CPU)…", file=sys.stderr)
    sd = load_state_dict(ckpt_path)
    tables = find_embeddings(sd)
    if not tables:
        sys.exit("❌  No embedding tensors found. Adjust the search heuristic.")

    # summary table
    print("\n════════════════════════════════════════════════════════════════════")
    print(f"{'TABLE NAME':60} │ {'TOKENS':>7} × {'DIM':<4} │ ROLE")
    print("════════════════════════════════════════════════════════════════════")
    for name, w in tables:
        n_tok, dim = w.shape
        role = classify(name)
        print(f"{name:60} │ {n_tok:7d} × {dim:<4d} │ {role}")
    print("════════════════════════════════════════════════════════════════════")
    print(f"Found {len(tables)} embedding table(s).")

    # if requested, dump actual values for grid + puzzle embeddings
    if args.dump:
        grid = next((w for n, w in tables if classify(n) == "Grid/colour tokens"), None)
        puzzle = next((w for n, w in tables if classify(n) == "Puzzle‑ID tokens"), None)
        if grid is None or puzzle is None:
            sys.exit("❌  Could not find both grid/colour and puzzle‑ID embeddings.")
        print_embeddings(grid, puzzle)


if __name__ == "__main__":
    main()
