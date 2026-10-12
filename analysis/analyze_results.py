#!/usr/bin/env python3
"""
analyze_results.py -- turns results/raw/benchmark_results.csv (one row per
repetition) into results/processed/benchmark_summary.csv (one row per
configuration, with median time / speedup / efficiency), matching the
schema given in Spec section 32:

    algorithm,mode,dataset_size,mpi_ranks,threads,time,speedup,efficiency

Speedup S(p) = T1 / Tp, Efficiency E(p) = S(p) / p, where:
  - T1 is the median SEQUENTIAL time for that (algorithm, dataset_size)
  - p is "workers" = mpi_ranks * threads (Spec section 18's "up to 16
    execution contexts" accounting -- this is how hybrid configurations
    are compared on the same p-axis as pure OpenMP/MPI ones)
"""
import argparse
import csv
import statistics
from collections import defaultdict


def load_raw(path):
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["mpi_ranks"] = int(row["mpi_ranks"])
            row["threads"] = int(row["threads"])
            row["time"] = float(row["time"])
            rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="results/raw/benchmark_results.csv")
    ap.add_argument("--output", default="results/processed/benchmark_summary.csv")
    args = ap.parse_args()

    rows = load_raw(args.input)

    # group repetitions -> median time per (algorithm, mode, dataset_size, ranks, threads)
    groups = defaultdict(list)
    for r in rows:
        key = (r["algorithm"], r["mode"], r["dataset_size"], r["mpi_ranks"], r["threads"])
        if r["time"] > 0:
            groups[key].append(r["time"])

    medians = {k: statistics.median(v) for k, v in groups.items() if v}

    # sequential baseline time T1, per (algorithm, dataset_size)
    baseline = {}
    for (algo, mode, dsize, ranks, threads), t in medians.items():
        if mode == "sequential":
            baseline[(algo, dsize)] = t

    out_rows = []
    for (algo, mode, dsize, ranks, threads), t in sorted(medians.items()):
        p = max(1, ranks * threads) if mode != "sequential" else 1
        t1 = baseline.get((algo, dsize))
        speedup = (t1 / t) if (t1 and t > 0) else ""
        efficiency = (speedup / p) if isinstance(speedup, float) else ""
        out_rows.append({
            "algorithm": algo, "mode": mode, "dataset_size": dsize,
            "mpi_ranks": ranks, "threads": threads, "workers": p,
            "time": round(t, 6),
            "speedup": round(speedup, 4) if isinstance(speedup, float) else "",
            "efficiency": round(efficiency, 4) if isinstance(efficiency, float) else "",
        })

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "algorithm", "mode", "dataset_size", "mpi_ranks", "threads",
            "workers", "time", "speedup", "efficiency"])
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"Wrote {len(out_rows)} summarized configurations to {args.output}")
    missing_baselines = {(a, d) for (a, m, d, r, t) in medians if m != "sequential"} - set(baseline)
    if missing_baselines:
        print("WARNING: no sequential baseline found for:", missing_baselines,
              "-- speedup/efficiency left blank for those rows.")


if __name__ == "__main__":
    main()
