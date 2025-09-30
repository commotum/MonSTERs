#!/usr/bin/env python3
"""
Analyze ARC-style datasets for example counts.

What it does per dataset directory passed as an argument:
- Recursively finds JSON files that contain a top-level "train" list.
- Counts ONLY training pairs: number of items in the "train" list per file.
- Ignores any "test" entries completely.
- Prints min/max and a histogram of counts across tasks.

Usage:
  python3 dataset/raw-data/analyze_examples.py <DATASET_DIR> [<DATASET_DIR> ...]

Examples:
  python3 dataset/raw-data/analyze_examples.py \
      dataset/raw-data/ARC-ALL \
      dataset/raw-data/ARC-AGI \
      dataset/raw-data/ARC-AGI-2 \
      dataset/raw-data/ConceptARC

Notes on specs:
- Original ARC (2019) typically has 3–5 training pairs.
- ARC-AGI / ARC-AGI-2 public specs indicate 2–10 training pairs.
  This script reports what is present in the files you provide.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def find_task_files(root: Path):
    # Recursively scan for JSON files
    # We'll filter by content when loading to ensure they have a train list
    for p in root.rglob("*.json"):
        yield p


def load_train_len(path: Path) -> int | None:
    try:
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
    except Exception:
        return None
    train = obj.get("train")
    if not isinstance(train, list):
        return None
    return len(train)


def analyze_dataset_dir(dataset_dir: Path):
    counts: list[int] = []
    items: list[tuple[str, int]] = []  # (file stem, train_len)
    for p in find_task_files(dataset_dir):
        n = load_train_len(p)
        if n is not None:
            counts.append(n)
            items.append((p.stem, n))
    if not counts:
        return None
    c = Counter(counts)
    min_n = min(c)
    max_n = max(c)
    return len(counts), min_n, max_n, c, items


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Analyze ARC-style datasets for training example counts.")
    parser.add_argument("dirs", nargs="+", type=Path, help="One or more dataset directories to analyze")
    parser.add_argument("--list-over", dest="list_over", type=int, default=None,
                        help="List task IDs (file stems) with more than N training examples")
    args = parser.parse_args(argv)

    exit_code = 0
    for d in args.dirs:
        if not d.exists():
            print(f"[WARN] Skipping missing directory: {d}")
            exit_code = 1
            continue
        result = analyze_dataset_dir(d)
        if result is None:
            print(f"[INFO] No task JSON files with a 'train' list found in: {d}")
            continue
        total, min_n, max_n, hist, items = result
        print(f"\nDataset: {d}")
        print("- Counting ONLY training pairs (len(train)); test is ignored")
        print(f"- Tasks scanned: {total}")
        print(f"- Min examples per task: {min_n}")
        print(f"- Max examples per task: {max_n}")
        print("- Histogram (num_examples -> num_tasks):")
        for n in range(min_n, max_n + 1):
            print(f"  {n} -> {hist.get(n, 0)}")

        if args.list_over is not None:
            over = sorted([stem for stem, n in items if n > args.list_over])
            print(f"- Files with > {args.list_over} training examples (stem only):")
            if over:
                for stem in over:
                    print(f"  {stem}")
            else:
                print("  (none)")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
