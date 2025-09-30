#!/usr/bin/env python3
"""
Analyze ConceptARC tasks to compute:
1) Max number of examples (input/output grid pairs) in any task
2) Min number of examples in any task
3) Count of tasks by number of examples from min..max

Notes:
- Counts only training examples (pairs in the "train" array)
- Ignores any "test" entries
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path


def iter_task_files(root: Path):
    # ConceptARC stores tasks under corpus/* and MinimalTasks/*. Recurse and filter by content.
    for p in root.rglob("*.json"):
        yield p


def count_train_examples(task_path: Path) -> int | None:
    try:
        with task_path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
    except Exception:
        return None

    train = obj.get("train")
    if not isinstance(train, list):
        return None
    return len(train)


def main() -> int:
    root = Path(__file__).resolve().parent
    counts = []
    files = []
    for p in iter_task_files(root):
        n = count_train_examples(p)
        if n is not None:
            counts.append(n)
            files.append(p)

    if not counts:
        print("No task JSON files with a 'train' field found.")
        return 1

    c = Counter(counts)
    min_n = min(c)
    max_n = max(c)

    print("ConceptARC Training Example Counts (per task)")
    print(f"- Tasks scanned: {len(files)}")
    print(f"- Min examples per task: {min_n}")
    print(f"- Max examples per task: {max_n}")
    print("- Histogram (num_examples -> num_tasks):")
    for n in range(min_n, max_n + 1):
        print(f"  {n} -> {c.get(n, 0)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

