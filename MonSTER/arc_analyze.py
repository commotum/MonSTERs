#!/usr/bin/env python3
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any
import argparse
import json
import sys
import csv
from collections import defaultdict, Counter

ARC_DEFAULT_TARGETS = [4, 6, 8, 10, 12, 15, 20, 24, 30]

def grid_size(grid: List[List[int]]) -> Tuple[int, int]:
    """Return (W, H) for a grid (list of list)."""
    if grid is None:
        return (0, 0)
    h = len(grid)
    w = len(grid[0]) if h > 0 else 0
    return (w, h)

def safe_max_pair(*pairs: Optional[Tuple[int, int]]) -> Tuple[int, int]:
    """Componentwise max over (W,H) pairs, ignoring None."""
    max_w, max_h = 0, 0
    for p in pairs:
        if p is None:
            continue
        w, h = p
        if w > max_w:
            max_w = w
        if h > max_h:
            max_h = h
    return (max_w, max_h)

def is_arc_task_json(obj: Any) -> bool:
    """Heuristic: has 'train' and 'test' lists with dict items containing 'input'."""
    if not isinstance(obj, dict):
        return False
    if "train" not in obj or "test" not in obj:
        return False
    if not isinstance(obj["train"], list) or not isinstance(obj["test"], list):
        return False
    # At least one example with 'input'
    def has_input(lst):
        for item in lst:
            if isinstance(item, dict) and "input" in item:
                return True
        return False
    return has_input(obj["train"]) or has_input(obj["test"])

def ceil_to_targets(x: int, targets: List[int]) -> int:
    for t in targets:
        if x <= t:
            return t
    return targets[-1]

def assign_bucket(W: int, H: int, w_targets: List[int], h_targets: List[int], canonicalize: bool) -> Tuple[int, int, bool, int, int]:
    """Return (Wb, Hb, rotated, W_eff, H_eff). If canonicalize and W<H, swap."""
    rotated = False
    W_eff, H_eff = W, H
    if canonicalize and W < H:
        W_eff, H_eff = H, W
        rotated = True
    Wb = ceil_to_targets(W_eff, w_targets)
    Hb = ceil_to_targets(H_eff, h_targets)
    return Wb, Hb, rotated, W_eff, H_eff

def discover_task_files(roots: List[Path]) -> List[Path]:
    """Recursively find ARC-like .json task files under the given roots.
    
    Typical ARC layouts:
      data/
        training/*.json
        evaluation/*.json
    """
    found: List[Path] = []
    seen = set()
    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob("*.json"):
            try:
                # lightweight prefilter: file size < 1MB typical for ARC
                if p.stat().st_size > 2_000_000:
                    continue
                key = (p.resolve())
                if key in seen:
                    continue
                with p.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                if is_arc_task_json(data):
                    found.append(p)
                    seen.add(key)
            except Exception:
                # ignore unreadable / non-JSON
                continue
    return found

def analyze_task(path: Path, max_side: int, assume_test_out_equals_in: bool = False) -> Dict[str, Any]:
    """Parse a single task JSON and compute per-example/test maxima and per-task canvas."""
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    train = data.get("train", [])
    test = data.get("test", [])
    details = []
    # Examples (train)
    e_max_pairs = []
    for idx, ex in enumerate(train, start=1):
        inp = ex.get("input")
        out = ex.get("output")
        in_size = grid_size(inp)
        out_size = grid_size(out) if out is not None else None
        pair_max = safe_max_pair(in_size, out_size)
        e_max_pairs.append(pair_max)
        details.append({
            "kind": "E",
            "idx": idx,
            "input_size": in_size,
            "output_size": out_size,
            "pair_max": pair_max
        })
    # Tests
    t_max_pairs = []
    for idx, ex in enumerate(test, start=1):
        inp = ex.get("input")
        out = ex.get("output", None)
        in_size = grid_size(inp)
        if out is None and assume_test_out_equals_in:
            out_size = in_size
        else:
            out_size = grid_size(out) if out is not None else None
        pair_max = safe_max_pair(in_size, out_size)
        t_max_pairs.append(pair_max)
        details.append({
            "kind": "T",
            "idx": idx,
            "input_size": in_size,
            "output_size": out_size,
            "pair_max": pair_max
        })
    # Task canvas: componentwise max over all pair_max
    all_pairs = e_max_pairs + t_max_pairs
    if not all_pairs:
        W, H = 0, 0
    else:
        W, H = safe_max_pair(*all_pairs)
    # clip to max_side
    W = min(W, max_side)
    H = min(H, max_side)
    return {
        "path": str(path),
        "task_id": path.stem,
        "W": W,
        "H": H,
        "examples": len(train),
        "tests": len(test),
        "details": details,
    }

def write_csv(rows: List[Dict[str, Any]], csv_path: Path) -> None:
    if not rows:
        return
    fieldnames = ["task_id", "path", "W", "H", "bucket_W", "bucket_H", "rotated", "W_eff", "H_eff", "examples", "tests", "area"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            row = {
                "task_id": r["task_id"],
                "path": r["path"],
                "W": r["W"],
                "H": r["H"],
                "bucket_W": r.get("bucket_W"),
                "bucket_H": r.get("bucket_H"),
                "rotated": r.get("rotated"),
                "W_eff": r.get("W_eff"),
                "H_eff": r.get("H_eff"),
                "examples": r["examples"],
                "tests": r["tests"],
                "area": r["W"] * r["H"],
            }
            w.writerow(row)

def print_task_details(task: Dict[str, Any]) -> None:
    print(f"Task {task['task_id']}:")
    for d in task["details"]:
        kind = d["kind"]
        idx = d["idx"]
        iw, ih = d["input_size"]
        if d["output_size"] is not None:
            ow, oh = d["output_size"]
            out_str = f"{ow}x{oh}"
        else:
            out_str = "N/A"
        mw, mh = d["pair_max"]
        print(f"  {kind}{idx}: Input - {iw}x{ih}, Output - {out_str}, Max ({mw}x{mh})")
    print(f"  Largest/Max Total: ({task['W']}x{task['H']})\n")

def main(argv=None):
    p = argparse.ArgumentParser(description="Analyze ARC task JSONs for per-task max grid sizes and bucket them.")
    p.add_argument("roots", nargs="*", default=["."], help="Root directories to search (recursively) for ARC JSON task files.")
    p.add_argument("--max-side", type=int, default=30, help="Max side length to clip canvas sizes (default: 30).")
    p.add_argument("--assume-test-output-equals-input", action="store_true",
                   help="If set, assume test outputs (when missing) have same size as inputs.")
    p.add_argument("--canonicalize", action="store_true", help="Rotate tasks so width>=height for bucketing only.")
    p.add_argument("--w-targets", type=int, nargs="+", default=ARC_DEFAULT_TARGETS, help="Target widths for bucket ceiling.")
    p.add_argument("--h-targets", type=int, nargs="+", default=ARC_DEFAULT_TARGETS, help="Target heights for bucket ceiling.")
    p.add_argument("--csv", type=str, help="Optional path to write a CSV summary.")
    p.add_argument("--print-details", action="store_true", help="Print per-example/test details like E1/T1 lines.")
    p.add_argument("--show-buckets", action="store_true", help="Print bucket distribution summary.")
    args = p.parse_args(argv)

    roots = [Path(r).expanduser().resolve() for r in args.roots]
    task_files = discover_task_files(roots)
    if not task_files:
        print("No ARC-like task JSON files found under:", ", ".join(str(r) for r in roots))
        print("Tip: Point me at the directory that contains 'training' and/or 'evaluation' folders from ARC.")
        return 2

    results = []
    buckets = Counter()
    bucket_examples = defaultdict(list)

    for path in sorted(task_files):
        task = analyze_task(path, max_side=args.max_side, assume_test_out_equals_in=args.assume_test_output_equals_input)
        W, H = task["W"], task["H"]
        Wb, Hb, rotated, W_eff, H_eff = assign_bucket(W, H, args.w_targets, args.h_targets, args.canonicalize)
        task_row = {
            **task,
            "bucket_W": Wb,
            "bucket_H": Hb,
            "rotated": rotated,
            "W_eff": W_eff,
            "H_eff": H_eff,
        }
        results.append(task_row)
        buckets[(Wb, Hb)] += 1
        bucket_examples[(Wb, Hb)].append(task_row)

        if args.print_details:
            print_task_details(task)

    # Summary
    total = len(results)
    print(f"Found {total} ARC task files.")
    # Top-line stats
    max_W = max(r["W"] for r in results) if results else 0
    max_H = max(r["H"] for r in results) if results else 0
    print(f"Max canvas across all tasks: ({max_W}x{max_H}) (clipped to {args.max_side})")

    if args.show_buckets:
        print("\nBucket distribution (Wb x Hb : count):")
        for (wb, hb), c in sorted(buckets.items(), key=lambda x: (x[0][1]*x[0][0], x[0][1], x[0][0])):
            print(f"  {wb:>2}x{hb:<2} : {c}")

    if args.csv:
        csv_path = Path(args.csv).expanduser().resolve()
        write_csv(results, csv_path)
        print(f"\nWrote CSV summary to: {csv_path}")

    # Return code 0 on success
    return 0

if __name__ == "__main__":
    sys.exit(main())

