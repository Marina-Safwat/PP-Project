#!/usr/bin/env python3
"""
generate_dataset.py -- synthetic DNA dataset generator (Spec section 20A).

Generates random FASTA sequences of a requested size, optionally injecting
a known query pattern at known, reproducible positions (with a fixed
random seed) and/or known-mismatch variants, so that automated tests and
benchmarks have ground truth to check against.

Examples:
    # Plain random genome
    python3 generate_dataset.py --size 100000000 --output data/synthetic/100MB.fasta

    # Genome with a known query injected 50 times (for correctness checks)
    python3 generate_dataset.py --size 10000000 --output data/synthetic/10MB.fasta \
        --inject-query ACGTAGCTAG --inject-count 50 --seed 42

    # Multiple sequences (records) instead of one long one, e.g. for
    # Smith-Waterman batch-alignment (MPI) testing
    python3 generate_dataset.py --size 1000 --records 200 \
        --output data/synthetic/sw_batch_db.fasta --seed 7
"""
import argparse
import random
import sys

BASES = "ACGT"


def random_sequence(rng: random.Random, length: int) -> str:
    return "".join(rng.choice(BASES) for _ in range(length))


def inject_pattern(rng: random.Random, seq: list, pattern: str, count: int):
    """Injects `pattern` at `count` random non-overlapping-ish positions.
    Returns the sorted list of 0-based positions used, for verification.
    """
    n = len(seq)
    m = len(pattern)
    if m == 0 or m > n or count <= 0:
        return []
    positions = set()
    attempts = 0
    max_attempts = count * 50 + 1000
    while len(positions) < count and attempts < max_attempts:
        pos = rng.randrange(0, n - m + 1)
        # avoid overlapping an already-placed injection
        if not any(abs(pos - p) < m for p in positions):
            positions.add(pos)
            seq[pos:pos + m] = list(pattern)
        attempts += 1
    return sorted(positions)


def write_fasta(path: str, records, line_width: int = 70):
    with open(path, "w") as f:
        for header, seq in records:
            f.write(f">{header}\n")
            for i in range(0, len(seq), line_width):
                f.write(seq[i:i + line_width] + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--size", type=int, required=True,
                     help="Total bases per record (or total genome length if --records 1)")
    ap.add_argument("--records", type=int, default=1,
                     help="Number of FASTA records to generate (default 1 = one long genome)")
    ap.add_argument("--output", required=True, help="Output FASTA path")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed (reproducibility)")
    ap.add_argument("--inject-query", default=None,
                     help="If set, injects this exact pattern into the FIRST record")
    ap.add_argument("--inject-count", type=int, default=0,
                     help="How many times to inject --inject-query")
    ap.add_argument("--positions-out", default=None,
                     help="Optional file to write the ground-truth injected positions to")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    records = []
    for r in range(args.records):
        seq = list(random_sequence(rng, args.size))
        header = f"synthetic_seq_{r+1}"
        if r == 0 and args.inject_query and args.inject_count > 0:
            positions = inject_pattern(rng, seq, args.inject_query, args.inject_count)
            if args.positions_out:
                with open(args.positions_out, "w") as pf:
                    pf.write("\n".join(str(p) for p in positions) + "\n")
            print(f"Injected {len(positions)} copies of '{args.inject_query}' "
                  f"into {header} (requested {args.inject_count})", file=sys.stderr)
        records.append((header, "".join(seq)))

    write_fasta(args.output, records)
    total_bases = sum(len(s) for _, s in records)
    print(f"Wrote {args.output}: {args.records} record(s), {total_bases:,} total bases", file=sys.stderr)


if __name__ == "__main__":
    main()
