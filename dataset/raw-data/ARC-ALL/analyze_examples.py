#!/usr/bin/env python3
"""
Analyze ARC-ALL tasks to compute:
1) Max number of examples (input/output grid pairs) in any task
2) Min number of examples in any task
3) Count of tasks by number of examples from min..max

Notes:
- Only counts training examples (i.e., pairs in the "train" array)
- Ignores/does not use any "test" entries
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path


def iter_task_files(root: Path):
    data_dir = root / "data"
    if not data_dir.exists():
        raise SystemExit(f"Data directory not found: {data_dir}")

    # Include both training and evaluation splits
    for split in ("training", "evaluation"):
        split_dir = data_dir / split
        if not split_dir.exists():
            continue
        for p in split_dir.glob("*.json"):
            yield p


def count_train_examples(task_path: Path) -> int:
    try:
        with task_path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        train = obj.get("train", [])
        # Ensure each item has an input/output pair, but we only count pairs
        return int(len(train))
    except Exception as e:
        print(f"Warning: failed to parse {task_path}: {e}", file=sys.stderr)
        return 0


def main() -> int:
    root = Path(__file__).resolve().parent
    counts = []
    files = list(iter_task_files(root))
    if not files:
        print("No task JSON files found.")
        return 1

    for p in files:
        n = count_train_examples(p)
        # Treat tasks with zero train pairs as zero (still included)
        counts.append(n)

    if not counts:
        print("No counts found.")
        return 1

    c = Counter(counts)
    min_n = min(c)
    max_n = max(c)

    print("ARC-ALL Training Example Counts (per task)")
    print(f"- Tasks scanned: {len(files)}")
    print(f"- Min examples per task: {min_n}")
    print(f"- Max examples per task: {max_n}")
    print("- Histogram (num_examples -> num_tasks):")
    for n in range(min_n, max_n + 1):
        print(f"  {n} -> {c.get(n, 0)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

