#!/usr/bin/env python3
"""
generate_graphs.py -- produces the graphs required by Spec section 27
from results/processed/benchmark_summary.csv:

  1. Threads vs Execution Time   (OpenMP, fixed dataset size)
  2. Threads vs Speedup          (OpenMP, fixed dataset size)
  3. Threads vs Efficiency       (OpenMP, fixed dataset size)
  4. Dataset Size vs Execution Time (best-performing mode per size)
  5. Sequential vs OpenMP vs MPI vs Hybrid (grouped bar, fixed size)
  6. Exact vs Approximate vs Alignment execution time comparison

Run after analysis/analyze_results.py. Output PNGs go to graphs/.
"""
import argparse
import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def to_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def pick_dataset_size(rows, algorithm=None):
    """Picks the largest dataset size present, so graphs show the most
    meaningful (least noise-dominated) regime by default. When algorithm
    is given, only considers rows for that algorithm -- otherwise a
    longer-named dataset size from a DIFFERENT algorithm (e.g. an
    alignment batch like "align_500recs") could get picked for an exact/
    approximate-matching graph just because its string sorts "larger",
    even though that dataset size has no exact/approximate rows at all."""
    if algorithm is not None:
        rows = [r for r in rows if r["algorithm"] == algorithm]
    sizes = sorted({r["dataset_size"] for r in rows},
                   key=lambda s: (len(s), s))
    return sizes[-1] if sizes else None


def graph_threads_vs(rows, dsize, algo, metric, ylabel, title, outpath):
    pts = [(int(r["threads"]), to_float(r[metric]))
           for r in rows
           if r["mode"] == "openmp" and r["dataset_size"] == dsize
           and r["algorithm"] == algo and to_float(r[metric]) is not None]
    pts.sort()
    if not pts:
        print(f"  (skip) {outpath}: no OpenMP data for {algo}/{dsize}")
        return
    xs, ys = zip(*pts)
    plt.figure(figsize=(6, 4))
    plt.plot(xs, ys, marker="o")
    if metric == "speedup":
        plt.plot(xs, xs, linestyle="--", color="gray", label="ideal linear speedup")
        plt.legend()
    plt.xlabel("Number of OpenMP Threads")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()
    print(f"  wrote {outpath}")


def graph_dataset_size_vs_time(rows, algo, outpath):
    by_size = defaultdict(list)
    for r in rows:
        if r["algorithm"] == algo and r["mode"] != "sequential":
            t = to_float(r["time"])
            if t is not None:
                by_size[r["dataset_size"]].append(t)
    if not by_size:
        print(f"  (skip) {outpath}: no data")
        return
    sizes = sorted(by_size.keys(), key=lambda s: (len(s), s))
    best_times = [min(by_size[s]) for s in sizes]
    plt.figure(figsize=(6, 4))
    plt.plot(sizes, best_times, marker="o")
    plt.xlabel("Dataset Size")
    plt.ylabel("Best Execution Time (s)")
    plt.title(f"Dataset Size vs Execution Time ({algo})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()
    print(f"  wrote {outpath}")


def graph_mode_comparison(rows, dsize, algo, outpath):
    best_by_mode = {}
    for r in rows:
        if r["dataset_size"] == dsize and r["algorithm"] == algo:
            t = to_float(r["time"])
            if t is None:
                continue
            mode = r["mode"]
            if mode not in best_by_mode or t < best_by_mode[mode]:
                best_by_mode[mode] = t
    order = ["sequential", "openmp", "mpi", "hybrid"]
    modes = [m for m in order if m in best_by_mode]
    if not modes:
        print(f"  (skip) {outpath}: no data")
        return
    times = [best_by_mode[m] for m in modes]
    plt.figure(figsize=(6, 4))
    plt.bar(modes, times, color=["#888", "#4c72b0", "#55a868", "#c44e52"][:len(modes)])
    plt.ylabel("Best Execution Time (s)")
    plt.title(f"Sequential vs OpenMP vs MPI vs Hybrid ({algo}, {dsize})")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()
    print(f"  wrote {outpath}")


def graph_algorithm_comparison(rows, dsize, outpath):
    best_by_algo = {}
    for r in rows:
        if r["dataset_size"] == dsize:
            t = to_float(r["time"])
            if t is None:
                continue
            algo = r["algorithm"]
            if algo not in best_by_algo or t < best_by_algo[algo]:
                best_by_algo[algo] = t
    if not best_by_algo:
        print(f"  (skip) {outpath}: no data")
        return
    algos = list(best_by_algo.keys())
    times = [best_by_algo[a] for a in algos]
    plt.figure(figsize=(6, 4))
    plt.bar(algos, times, color="#4c72b0")
    plt.ylabel("Best Execution Time (s)")
    plt.title(f"Exact vs Approximate vs Alignment ({dsize})")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()
    print(f"  wrote {outpath}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="results/processed/benchmark_summary.csv")
    ap.add_argument("--outdir", default="graphs")
    ap.add_argument("--algorithm", default="exact",
                     help="Primary algorithm for the thread-scaling graphs (1-3)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    rows = load(args.input)
    if not rows:
        print(f"No rows in {args.input} -- run run_benchmarks.sh and analyze_results.py first.")
        return

    dsize = pick_dataset_size(rows, algorithm=args.algorithm)
    print(f"Using dataset size '{dsize}' for fixed-size graphs (largest available).")

    graph_threads_vs(rows, dsize, args.algorithm, "time", "Execution Time (s)",
                      f"Threads vs Execution Time ({args.algorithm}, {dsize})",
                      os.path.join(args.outdir, "01_threads_vs_time.png"))
    graph_threads_vs(rows, dsize, args.algorithm, "speedup", "Speedup S(p)",
                      f"Threads vs Speedup ({args.algorithm}, {dsize})",
                      os.path.join(args.outdir, "02_threads_vs_speedup.png"))
    graph_threads_vs(rows, dsize, args.algorithm, "efficiency", "Efficiency E(p)",
                      f"Threads vs Efficiency ({args.algorithm}, {dsize})",
                      os.path.join(args.outdir, "03_threads_vs_efficiency.png"))
    graph_dataset_size_vs_time(rows, args.algorithm, os.path.join(args.outdir, "04_dataset_size_vs_time.png"))
    graph_mode_comparison(rows, dsize, args.algorithm, os.path.join(args.outdir, "05_mode_comparison.png"))
    graph_algorithm_comparison(rows, dsize, os.path.join(args.outdir, "06_algorithm_comparison.png"))


if __name__ == "__main__":
    main()
