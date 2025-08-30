#!/usr/bin/env python3
"""
ARC Task Bucketing & Size Analysis
---------------------------------
Given a directory of ARC-style JSON tasks, compute per-task max canvases and
bucket them on a 2D lattice of target widths and heights.

Schema expected (standard ARC):
{
  "train": [
    {"input": [[...], ...], "output": [[...], ...]},
    ...
  ],
  "test": [
    {"input": [[...], ...]}  # "output" may be present in some corpora, absent in others
  ]
}

This script:
  1) For each *example* (input/output) and each *test* pair, computes the per-pair
     max width and max height across its available grids (input and, if present, output).
  2) For each *task*, takes the max of those per-pair maxima ⇒ (W_task, H_task).
  3) Optionally caps sizes (default 30×30).
  4) Assigns each task to a bucket via axis-wise ceiling to target sets W_T, H_T.
  5) Prints a compact distribution table + saves CSV summaries.
  6) Provides a second analysis with multiples-of-4 bucketing.

Usage examples:
  python arc_analyze.py /path/to/tasks_dir
  python arc_analyze.py /path/to/tasks_dir --w-targets 4,6,8,10,12,15,20,24,30 --h-targets 4,6,8,10,12,15,20,24,30
  python arc_analyze.py /path/to/train --canonicalize --pixel-budget 20000 --top 25 --csv-prefix out/arc

Outputs:
  - CSV #1: <csv_prefix>_tasks.csv : one row per task with sizes and bucket info
  - CSV #2: <csv_prefix>_buckets.csv : one row per bucket with aggregated stats
  - CSV #3: <csv_prefix>_multiples4.csv : one row per bucket with multiples-of-4 bucketing
  - CSV #4: <csv_prefix>_square_multiples8.csv : one row per bucket with square multiples-of-8 bucketing
  - CSV #5: <csv_prefix>_square_multiples4.csv : one row per bucket with square multiples-of-4 bucketing

No third-party deps required (stdlib only).
"""

from __future__ import annotations
import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from collections import Counter, defaultdict

# ---------------------------
# Core helpers
# ---------------------------

@dataclass
class TaskSummary:
    task_id: str
    path: Path
    W: int
    H: int
    Wb: int
    Hb: int
    rotated: bool
    area: int
    b_area: int
    overhead: float  # bucket_area / area
    fill_deficit: float  # relative to cap^2 (if --cap provided)
    # Multiples of 4 bucketing
    Wb4: int
    Hb4: int
    b_area4: int
    overhead4: float
    # Square multiples of 8 bucketing
    Wb8: int
    Hb8: int
    b_area8: int
    overhead8: float
    # Square multiples of 4 bucketing
    Wb4sq: int
    Hb4sq: int
    b_area4sq: int
    overhead4sq: float


def grid_dims(grid: List[List[int]]) -> Tuple[int, int]:
    """Return (W, H) for a grid (list of lists)."""
    if not isinstance(grid, list):
        return (0, 0)
    H = len(grid)
    W = max((len(row) for row in grid), default=0)
    return (W, H)


def pair_max_dims(example: Dict) -> Tuple[int, int]:
    """For an example/test dict, return per-pair max (W, H) over input and output (if present)."""
    # Some corpora have "output" for tests, some do not.
    maxW = 0
    maxH = 0
    for key in ("input", "output"):
        if key in example:
            W, H = grid_dims(example[key])
            maxW = max(maxW, W)
            maxH = max(maxH, H)
    return (maxW, maxH)


def task_canvas(task: Dict, cap: Optional[int]) -> Tuple[int, int]:
    """Compute the max canvas (W, H) for a task across all train+test pairs."""
    Ws, Hs = [], []
    for sect in ("train", "test"):
        for ex in task.get(sect, []) or []:
            w, h = pair_max_dims(ex)
            Ws.append(w)
            Hs.append(h)
    W = max(Ws, default=0)
    H = max(Hs, default=0)
    if cap is not None:
        W = min(W, cap)
        H = min(H, cap)
    return (W, H)


def parse_targets(s: Optional[str], default: List[int]) -> List[int]:
    if not s:
        return default
    try:
        vals = [int(x.strip()) for x in s.split(',') if x.strip()]
    except ValueError:
        raise argparse.ArgumentTypeError("Targets must be a comma-separated list of integers.")
    if not vals:
        return default
    vals = sorted(set(vals))
    return vals


def ceil_to_targets(x: int, targets: List[int]) -> int:
    for t in targets:
        if x <= t:
            return t
    return targets[-1]


def assign_bucket(W: int, H: int, W_T: List[int], H_T: List[int], canonicalize: bool) -> Tuple[int, int, bool]:
    rotated = False
    w, h = W, H
    if canonicalize and w < h:
        w, h = h, w
        rotated = True
    Wb = ceil_to_targets(w, W_T)
    Hb = ceil_to_targets(h, H_T)
    return (Wb, Hb, rotated)


def assign_multiples4_bucket(W: int, H: int, canonicalize: bool) -> Tuple[int, int, bool]:
    """Assign to buckets that are multiples of 4: 4x4, 4x8, 4x12, 8x8, 8x12, etc."""
    rotated = False
    w, h = W, H
    if canonicalize and w < h:
        w, h = h, w
        rotated = True
    
    # Round up to next multiple of 4
    Wb4 = ((w + 3) // 4) * 4
    Hb4 = ((h + 3) // 4) * 4
    
    return (Wb4, Hb4, rotated)


def assign_square_multiples8_bucket(W: int, H: int) -> Tuple[int, int, bool]:
    """Assign to square buckets that are multiples of 8: 8x8, 16x16, 24x24, 32x32."""
    # Take the larger dimension and round up to next multiple of 8
    max_dim = max(W, H)
    bucket_size = ((max_dim + 7) // 8) * 8
    
    # Ensure we don't exceed the cap (if any)
    if bucket_size > 32:  # Default cap is 30, so 32 is reasonable max
        bucket_size = 32
    
    return (bucket_size, bucket_size, False)  # Always square, no rotation needed


def assign_square_multiples4_bucket(W: int, H: int) -> Tuple[int, int, bool]:
    """Assign to square buckets that are multiples of 4: 4x4, 8x8, 12x12, 16x16, 20x20, 24x24, 28x28, 32x32."""
    # Take the larger dimension and round up to next multiple of 4
    max_dim = max(W, H)
    bucket_size = ((max_dim + 3) // 4) * 4
    
    # Ensure we don't exceed the cap (if any)
    if bucket_size > 32:  # Default cap is 30, so 32 is reasonable max
        bucket_size = 32
    
    return (bucket_size, bucket_size, False)  # Always square, no rotation needed


def aspect_category(W: int, H: int, tol: float = 0.1) -> str:
    """Label as 'square', 'wide', or 'tall'. tol=0.1 means ~±10% from square counts as square."""
    if H == 0 or W == 0:
        return "empty"
    r = W / H
    if 1 - tol <= r <= 1 + tol:
        return "square"
    return "wide" if r > 1 else "tall"


# ---------------------------
# Scanning & loading
# ---------------------------

def iter_task_paths(root: Path) -> Iterable[Path]:
    if root.is_file() and root.suffix.lower() == ".json":
        yield root
        return
    for p in sorted(root.rglob("*.json")):
        yield p


def load_task(path: Path) -> Optional[Dict]:
    try:
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        # Basic schema sanity
        if not any(k in obj for k in ("train", "test")):
            return None
        return obj
    except Exception:
        return None


# ---------------------------
# Aggregation
# ---------------------------

def summarize_tasks(
    root: Path,
    cap: Optional[int],
    W_T: List[int],
    H_T: List[int],
    canonicalize: bool,
) -> Tuple[List[TaskSummary], Counter, Counter, Counter, Counter]:
    records: List[TaskSummary] = []
    bucket_counts: Counter = Counter()
    bucket_counts4: Counter = Counter()
    bucket_counts8: Counter = Counter()
    bucket_counts4sq: Counter = Counter()

    for path in iter_task_paths(root):
        task = load_task(path)
        if task is None:
            continue
        W, H = task_canvas(task, cap)
        Wb, Hb, rotated = assign_bucket(W, H, W_T, H_T, canonicalize)
        Wb4, Hb4, rotated4 = assign_multiples4_bucket(W, H, canonicalize)
        Wb8, Hb8, rotated8 = assign_square_multiples8_bucket(W, H)
        Wb4sq, Hb4sq, rotated4sq = assign_square_multiples4_bucket(W, H)
        
        area = max(1, W * H)
        b_area = max(1, Wb * Hb)
        b_area4 = max(1, Wb4 * Hb4)
        b_area8 = max(1, Wb8 * Hb8)
        b_area4sq = max(1, Wb4sq * Hb4sq)
        
        overhead = b_area / area
        overhead4 = b_area4 / area
        overhead8 = b_area8 / area
        overhead4sq = b_area4sq / area
        
        cap_area = (cap * cap) if cap else None
        fill_deficit = 1.0 - (area / cap_area) if cap_area else float('nan')
        
        rec = TaskSummary(
            task_id=path.stem,
            path=path,
            W=W,
            H=H,
            Wb=Wb,
            Hb=Hb,
            rotated=rotated,
            area=area,
            b_area=b_area,
            overhead=overhead,
            fill_deficit=fill_deficit,
            Wb4=Wb4,
            Hb4=Hb4,
            b_area4=b_area4,
            overhead4=overhead4,
            Wb8=Wb8,
            Hb8=Hb8,
            b_area8=b_area8,
            overhead8=overhead8,
            Wb4sq=Wb4sq,
            Hb4sq=Hb4sq,
            b_area4sq=b_area4sq,
            overhead4sq=overhead4sq,
        )
        records.append(rec)
        bucket_counts[(Wb, Hb)] += 1
        bucket_counts4[(Wb4, Hb4)] += 1
        bucket_counts8[(Wb8, Hb8)] += 1
        bucket_counts4sq[(Wb4sq, Hb4sq)] += 1

    return records, bucket_counts, bucket_counts4, bucket_counts8, bucket_counts4sq


def aggregate_bucket_metrics(records: List[TaskSummary]) -> Dict[Tuple[int, int], Dict]:
    agg: Dict[Tuple[int, int], Dict] = defaultdict(lambda: {
        "count": 0,
        "mean_overhead": 0.0,
        "mean_area": 0.0,
        "examples": [],
    })
    for r in records:
        k = (r.Wb, r.Hb)
        a = agg[k]
        a["count"] += 1
        a["mean_overhead"] += r.overhead
        a["mean_area"] += r.area
        if len(a["examples"]) < 5:
            a["examples"].append(r.task_id)
    for k, a in agg.items():
        if a["count"]:
            a["mean_overhead"] /= a["count"]
            a["mean_area"] /= a["count"]
    return agg


def aggregate_multiples4_bucket_metrics(records: List[TaskSummary]) -> Dict[Tuple[int, int], Dict]:
    agg: Dict[Tuple[int, int], Dict] = defaultdict(lambda: {
        "count": 0,
        "mean_overhead": 0.0,
        "mean_area": 0.0,
        "examples": [],
    })
    for r in records:
        k = (r.Wb4, r.Hb4)
        a = agg[k]
        a["count"] += 1
        a["mean_overhead"] += r.overhead4
        a["mean_area"] += r.area
        if len(a["examples"]) < 5:
            a["examples"].append(r.task_id)
    for k, a in agg.items():
        if a["count"]:
            a["mean_overhead"] /= a["count"]
            a["mean_area"] /= a["count"]
    return agg


def aggregate_square_multiples8_bucket_metrics(records: List[TaskSummary]) -> Dict[Tuple[int, int], Dict]:
    agg: Dict[Tuple[int, int], Dict] = defaultdict(lambda: {
        "count": 0,
        "mean_overhead": 0.0,
        "mean_area": 0.0,
        "examples": [],
    })
    for r in records:
        k = (r.Wb8, r.Hb8)
        a = agg[k]
        a["count"] += 1
        a["mean_overhead"] += r.overhead8
        a["mean_area"] += r.area
        if len(a["examples"]) < 5:
            a["examples"].append(r.task_id)
    for k, a in agg.items():
        if a["count"]:
            a["mean_overhead"] /= a["count"]
            a["mean_area"] /= a["count"]
    return agg


def aggregate_square_multiples4_bucket_metrics(records: List[TaskSummary]) -> Dict[Tuple[int, int], Dict]:
    agg: Dict[Tuple[int, int], Dict] = defaultdict(lambda: {
        "count": 0,
        "mean_overhead": 0.0,
        "mean_area": 0.0,
        "examples": [],
    })
    for r in records:
        k = (r.Wb4sq, r.Hb4sq)
        a = agg[k]
        a["count"] += 1
        a["mean_overhead"] += r.overhead4sq
        a["mean_area"] += r.area
        if len(a["examples"]) < 5:
            a["examples"].append(r.task_id)
    for k, a in agg.items():
        if a["count"]:
            a["mean_overhead"] /= a["count"]
            a["mean_area"] /= a["count"]
    return agg


# ---------------------------
# Pretty printing
# ---------------------------

def print_header(title: str):
    print("\n" + title)
    print("=" * len(title))


def print_bucket_table(agg: Dict[Tuple[int, int], Dict], total: int, top: int, title: str = "Buckets"):
    rows = []
    for (Wb, Hb), a in agg.items():
        rows.append((Wb * Hb, Wb, Hb, a["count"], 100.0 * a["count"] / max(1, total), a["mean_overhead"], a["mean_area"], a["examples"]))
    rows.sort(key=lambda x: (-x[3], x[0], x[1], x[2]))  # by count desc, then area asc

    print(f"{title} - Top {min(top, len(rows))} buckets by count (of {total} tasks):")
    print(f"{'bucket':>9}  {'count':>5}  {'%':>6}  {'area':>6}  {'mean_over':>9}  examples")
    for i, (area, Wb, Hb, cnt, pct, mean_over, mean_area, ex) in enumerate(rows[:top], 1):
        print(f"{Wb:>2}x{Hb:<2}  {cnt:>5}  {pct:>6.2f}  {area:>6}  {mean_over:>9.3f}  {', '.join(ex)}")


def print_aspect_breakdown(records: List[TaskSummary]):
    cats = Counter(aspect_category(r.W, r.H) for r in records)
    total = max(1, sum(cats.values()))
    print("Aspect breakdown (task canvas):")
    for k in ("wide", "square", "tall", "empty"):
        if cats.get(k):
            print(f"  {k:7}: {cats[k]:>5}  ({100.0*cats[k]/total:5.2f}%)")


# ---------------------------
# CSV output
# ---------------------------

def write_task_csv(records: List[TaskSummary], out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "task_id", "path", "W", "H", "Wb", "Hb", "rotated",
            "area", "bucket_area", "overhead", "fill_deficit",
            "Wb4", "Hb4", "bucket_area4", "overhead4",
            "Wb8", "Hb8", "bucket_area8", "overhead8",
            "Wb4sq", "Hb4sq", "bucket_area4sq", "overhead4sq"
        ])
        for r in records:
            w.writerow([
                r.task_id,
                str(r.path),
                r.W, r.H,
                r.Wb, r.Hb,
                int(r.rotated),
                r.area, r.b_area,
                f"{r.overhead:.6f}",
                (f"{r.fill_deficit:.6f}" if not math.isnan(r.fill_deficit) else ""),
                r.Wb4, r.Hb4,
                r.b_area4,
                f"{r.overhead4:.6f}",
                r.Wb8, r.Hb8,
                r.b_area8,
                f"{r.overhead8:.6f}",
                r.Wb4sq, r.Hb4sq,
                r.b_area4sq,
                f"{r.overhead4sq:.6f}"
            ])


def write_bucket_csv(agg: Dict[Tuple[int, int], Dict], out_path: Path, total: int):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Wb", "Hb", "count", "percent", "bucket_area", "mean_overhead", "mean_area", "examples"]) 
        for (Wb, Hb), a in sorted(agg.items(), key=lambda kv: (-kv[1]["count"], kv[0][0]*kv[0][1])):
            pct = 100.0 * a["count"] / max(1, total)
            w.writerow([Wb, Hb, a["count"], f"{pct:.6f}", Wb*Hb, f"{a['mean_overhead']:.6f}", f"{a['mean_area']:.6f}", ";".join(a["examples"])])


def write_multiples4_csv(agg: Dict[Tuple[int, int], Dict], out_path: Path, total: int):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Wb4", "Hb4", "count", "percent", "bucket_area", "mean_overhead", "mean_area", "examples"]) 
        for (Wb4, Hb4), a in sorted(agg.items(), key=lambda kv: (-kv[1]["count"], kv[0][0]*kv[0][1])):
            pct = 100.0 * a["count"] / max(1, total)
            w.writerow([Wb4, Hb4, a["count"], f"{pct:.6f}", Wb4*Hb4, f"{a['mean_overhead']:.6f}", f"{a['mean_area']:.6f}", ";".join(a["examples"])])


def write_square_multiples8_csv(agg: Dict[Tuple[int, int], Dict], out_path: Path, total: int):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Wb8", "Hb8", "count", "percent", "bucket_area", "mean_overhead", "mean_area", "examples"]) 
        for (Wb8, Hb8), a in sorted(agg.items(), key=lambda kv: (-kv[1]["count"], kv[0][0]*kv[0][1])):
            pct = 100.0 * a["count"] / max(1, total)
            w.writerow([Wb8, Hb8, a["count"], f"{pct:.6f}", Wb8*Hb8, f"{a['mean_overhead']:.6f}", f"{a['mean_area']:.6f}", ";".join(a["examples"])])


def write_square_multiples4_csv(agg: Dict[Tuple[int, int], Dict], out_path: Path, total: int):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Wb4sq", "Hb4sq", "count", "percent", "bucket_area", "mean_overhead", "mean_area", "examples"]) 
        for (Wb4sq, Hb4sq), a in sorted(agg.items(), key=lambda kv: (-kv[1]["count"], kv[0][0]*kv[0][1])):
            pct = 100.0 * a["count"] / max(1, total)
            w.writerow([Wb4sq, Hb4sq, a["count"], f"{pct:.6f}", Wb4sq*Hb4sq, f"{a['mean_overhead']:.6f}", f"{a['mean_area']:.6f}", ";".join(a["examples"])])


# ---------------------------
# Main
# ---------------------------

def main():
    parser = argparse.ArgumentParser(description="Analyze ARC task sizes and 2D buckets.")
    parser.add_argument("root", type=str, help="Path to a task JSON file or a directory containing JSON tasks (recursively scanned)")
    parser.add_argument("--cap", type=int, default=30, help="Cap each side to this value (default: 30). Use 0 to disable.")
    parser.add_argument("--canonicalize", action="store_true", help="Rotate canvases to landscape (W>=H) for bucketing only.")
    parser.add_argument("--w-targets", type=str, default="4,6,8,10,12,15,20,24,30", help="Comma-separated bucket width targets.")
    parser.add_argument("--h-targets", type=str, default="4,6,8,10,12,15,20,24,30", help="Comma-separated bucket height targets.")
    parser.add_argument("--top", type=int, default=25, help="How many top buckets to print.")
    parser.add_argument("--csv-prefix", type=str, default="arc_analysis", help="Prefix for CSV outputs (tasks and buckets).")
    parser.add_argument("--pixel-budget", type=int, default=0, help="If >0, print suggested batch size per bucket as floor(pixel_budget / (Wb*Hb)).")
    parser.add_argument("--output-dir", type=str, default="/home/jake/Developer/MonSTERs/dataset/raw-data/ARC-ALL/data/analysis", help="Directory to save CSV outputs.")

    args = parser.parse_args()
    root = Path(args.root)
    if not root.exists():
        print(f"Path not found: {root}")
        return 1

    W_T = parse_targets(args.w_targets, [4,6,8,10,12,15,20,24,30])
    H_T = parse_targets(args.h_targets, [4,6,8,10,12,15,20,24,30])
    cap = None if args.cap and args.cap <= 0 else args.cap

    records, bucket_counts, bucket_counts4, bucket_counts8, bucket_counts4sq = summarize_tasks(root, cap, W_T, H_T, args.canonicalize)
    total = len(records)
    if total == 0:
        print("No valid tasks found.")
        return 0

    agg = aggregate_bucket_metrics(records)
    agg4 = aggregate_multiples4_bucket_metrics(records)
    agg8 = aggregate_square_multiples8_bucket_metrics(records)
    agg4sq = aggregate_square_multiples4_bucket_metrics(records)

    print_header("ARC Task 2D Bucketing Summary")
    print(f"Scanned: {total} tasks from {root}")
    print(f"Cap: {cap if cap else 'none'}, Canonicalize: {args.canonicalize}")
    print(f"Targets (W): {W_T}")
    print(f"Targets (H): {H_T}")
    print()

    print_aspect_breakdown(records)
    print()

    # Quick bucket orientation summary
    def count_orientations(agg_dict):
        square = sum(1 for (w, h) in agg_dict.keys() if w == h)
        wide = sum(1 for (w, h) in agg_dict.keys() if w > h)
        tall = sum(1 for (w, h) in agg_dict.keys() if w < h)
        return square, wide, tall
    
    std_sq, std_wide, std_tall = count_orientations(agg)
    mul4_sq, mul4_wide, mul4_tall = count_orientations(agg4)
    mul8_sq, mul8_wide, mul8_tall = count_orientations(agg8)
    mul4sq_sq, mul4sq_wide, mul4sq_tall = count_orientations(agg4sq)
    
    print(f"Standard buckets: {std_sq} square, {std_wide} wide, {std_tall} tall")
    print(f"Multiples-of-4:  {mul4_sq} square, {mul4_wide} wide, {mul4_tall} tall")
    print(f"Square multiples-of-8: {mul8_sq} square, {mul8_wide} wide, {mul8_tall} tall")
    print(f"Square multiples-of-4: {mul4sq_sq} square, {mul4sq_wide} wide, {mul4sq_tall} tall")
    print()

    print_bucket_table(agg, total, args.top, "Standard Buckets")
    print()
    print_bucket_table(agg4, total, args.top, "Multiples-of-4 Buckets")
    print()
    print_bucket_table(agg8, total, args.top, "Square Multiples-of-8 Buckets")
    print()
    print_bucket_table(agg4sq, total, args.top, "Square Multiples-of-4 Buckets")

    if args.pixel_budget and args.pixel_budget > 0:
        print("\nSuggested batch sizes (pixel budget / bucket area):")
        for (Wb, Hb), a in sorted(agg.items(), key=lambda kv: (kv[0][0]*kv[0][1])):
            area = Wb * Hb
            bs = max(1, args.pixel_budget // max(1, area))
            print(f"  {Wb:>2}x{Hb:<2} : area={area:>4} -> batch_size≈{bs}")

    # Write CSV outputs
    output_dir = Path(args.output_dir)
    tasks_csv = output_dir / f"{args.csv_prefix}_tasks.csv"
    buckets_csv = output_dir / f"{args.csv_prefix}_buckets.csv"
    multiples4_csv = output_dir / f"{args.csv_prefix}_multiples4.csv"
    square_multiples8_csv = output_dir / f"{args.csv_prefix}_square_multiples8.csv"
    square_multiples4_csv = output_dir / f"{args.csv_prefix}_square_multiples4.csv"
    
    write_task_csv(records, tasks_csv)
    write_bucket_csv(agg, buckets_csv, total)
    write_multiples4_csv(agg4, multiples4_csv, total)
    write_square_multiples8_csv(agg8, square_multiples8_csv, total)
    write_square_multiples4_csv(agg4sq, square_multiples4_csv, total)

    print("\nCSV written:")
    print(f"  Tasks              -> {tasks_csv}")
    print(f"  Buckets            -> {buckets_csv}")
    print(f"  Multiples4         -> {multiples4_csv}")
    print(f"  Square Multiples8  -> {square_multiples8_csv}")
    print(f"  Square Multiples4  -> {square_multiples4_csv}")

    # Example: print a few sample tasks from the most popular bucket
    ((top_Wb, top_Hb), _) = max(agg.items(), key=lambda kv: kv[1]["count"]) if agg else (((0,0),{"count":0}))
    examples = agg.get((top_Wb, top_Hb), {}).get("examples", [])
    if examples:
        print(f"\nMost common standard bucket: {top_Wb}x{top_Hb}, examples: {', '.join(examples)}")

    # Example: print a few sample tasks from the most popular multiples-of-4 bucket
    ((top_Wb4, top_Hb4), _) = max(agg4.items(), key=lambda kv: kv[1]["count"]) if agg4 else (((0,0),{"count":0}))
    examples4 = agg4.get((top_Wb4, top_Hb4), {}).get("examples", [])
    if examples4:
        print(f"Most common multiples-of-4 bucket: {top_Wb4}x{top_Hb4}, examples: {', '.join(examples4)}")

    # Example: print a few sample tasks from the most popular square multiples-of-8 bucket
    ((top_Wb8, top_Hb8), _) = max(agg8.items(), key=lambda kv: kv[1]["count"]) if agg8 else (((0,0),{"count":0}))
    examples8 = agg8.get((top_Wb8, top_Hb8), {}).get("examples", [])
    if examples8:
        print(f"Most common square multiples-of-8 bucket: {top_Wb8}x{top_Hb8}, examples: {', '.join(examples8)}")

    # Example: print a few sample tasks from the most popular square multiples-of-4 bucket
    ((top_Wb4sq, top_Hb4sq), _) = max(agg4sq.items(), key=lambda kv: kv[1]["count"]) if agg4sq else (((0,0),{"count":0}))
    examples4sq = agg4sq.get((top_Wb4sq, top_Hb4sq), {}).get("examples", [])
    if examples4sq:
        print(f"Most common square multiples-of-4 bucket: {top_Wb4sq}x{top_Hb4sq}, examples: {', '.join(examples4sq)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
