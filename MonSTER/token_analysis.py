#!/usr/bin/env python
"""
token_analysis.py  —  inspect every token‑embedding matrix in an HRM checkpoint
───────────────────────────────────────────────────────────────────────────────
ARC‑HRM uses multiple vocabularies:

• grid‑cell / colour tokens            → embed_tokens.embedding_weight
• puzzle‑ID (task) tokens              → puzzle_emb.weight
• optional special or project‑specific → e.g. halt_emb.weight, pos_emb.weight

This script loads the checkpoint on CPU, finds **every** 2‑D tensor that
looks like an embedding table, and prints shape + a short guess at its role.

Usage (from repo root):

    uv run -m MonSTER.token_analysis \
        checkpoints/HRM-checkpoint-ARC-2 [--ckpt-file checkpoint]

The script makes no assumptions about file names beyond containing
“emb”, “token”, or “weight”.  Adjust `NAME_RULES` if your codebase uses
different conventions.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, Tuple, List

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
        # common wrappers
        for key in ("model", "state_dict", "module"):
            if key in obj and isinstance(obj[key], dict):
                return obj[key]
        return obj  # the checkpoint *is* the state‑dict
    raise RuntimeError("Unsupported checkpoint format (expected dict).")


def find_embeddings(
    state_dict: Dict[str, torch.Tensor]
) -> List[Tuple[str, torch.Tensor]]:
    out: List[Tuple[str, torch.Tensor]] = []
    for name, tensor in state_dict.items():
        if tensor.ndim == 2 and any(tag in name.lower() for tag in ("emb", "token")):
            out.append((name, tensor))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description="List every token‑embedding matrix in an HRM checkpoint."
    )
    ap.add_argument(
        "ckpt_dir", type=Path, help="Folder that contains the checkpoint file"
    )
    ap.add_argument(
        "--ckpt-file",
        default="checkpoint",
        help="Checkpoint filename inside ckpt_dir (default: 'checkpoint')",
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

    print("\n════════════════════════════════════════════════════════════════════")
    print(f"{'TABLE NAME':60} │ {'TOKENS':>7} × {'DIM':<4} │ ROLE")
    print("════════════════════════════════════════════════════════════════════")
    for name, w in tables:
        n_tok, dim = w.shape
        role = classify(name)
        print(f"{name:60} │ {n_tok:7d} × {dim:<4d} │ {role}")
    print("════════════════════════════════════════════════════════════════════")
    print(f"Found {len(tables)} embedding table(s).")


if __name__ == "__main__":
    main()
