# Parallel DNA Sequence Matching and Alignment Engine

A hybrid MPI + OpenMP system for exact matching, mutation-tolerant
(Hamming-distance) matching, and Smith–Waterman local alignment over
large DNA datasets — built to investigate parallel-computing techniques
(task/data decomposition, synchronization, scaling) using genomics as a
realistic computational workload. See `report/design_decisions.md` for
the reasoning behind the architecture; this file is build/run
instructions.

## 1. Requirements

```
g++ >= 9 (C++17, OpenMP support)
OpenMPI or MPICH (mpic++, mpirun)
Python 3.8+ with pandas/matplotlib (for analysis/graphs)
```

On Ubuntu/Debian:
```bash
sudo apt-get install build-essential libopenmpi-dev openmpi-bin python3-pip
pip3 install matplotlib pandas
```

## 2. Build

```bash
make all      # builds bin/dna_seq, bin/dna_openmp, bin/dna_mpi, bin/dna_hybrid
make test     # builds + runs tests/test_all.cpp (2,700+ correctness checks)
make clean    # removes bin/ and benchmark outputs
```

## 3. Quick start

```bash
# Generate a synthetic dataset with known injected matches
python3 scripts/generate_dataset.py --size 10000000 \
    --output data/synthetic/10MB.fasta \
    --inject-query ACGTAGCTAG --inject-count 50 --seed 42

# Sequential
./bin/dna_seq --input data/synthetic/10MB.fasta --query ACGTAGCTAG --algorithm exact

# OpenMP (shared-memory)
./bin/dna_openmp --input data/synthetic/10MB.fasta --query ACGTAGCTAG \
    --algorithm exact --threads 8

# MPI (distributed-memory)
mpirun -np 4 ./bin/dna_mpi --input data/synthetic/10MB.fasta --query ACGTAGCTAG --algorithm exact

# Hybrid MPI + OpenMP
mpirun -np 4 ./bin/dna_hybrid --input data/synthetic/10MB.fasta --query ACGTAGCTAG \
    --algorithm exact --threads 4
```

All four binaries share the identical `--input/--query/--algorithm` CLI.
`--algorithm` is one of `exact | approximate | align`.
`approximate` additionally takes `--mismatches K`.
`align` additionally takes `--match/--mismatch/--gap` scoring parameters
(defaults 2/-1/-2). In `align` mode, every record in `--input` is aligned
against `--query`, and the best-scoring match is reported — this is
consistent across all four binaries (`dna_seq`/`dna_openmp` loop over
records sequentially with OpenMP wavefront-parallelizing each alignment;
`dna_mpi`/`dna_hybrid` additionally distribute records across ranks —
see design_decisions.md, "Why Smith-Waterman is not wavefront-
parallelized across MPI ranks"). Keeping this identical across all four
modes is what makes align-mode speedup/efficiency numbers meaningful.

## 4. Running the test/validation suite

```bash
make test                              # unit-level correctness (2,697 checks)
./scripts/validate_all_modes.sh        # end-to-end: seq == openmp == mpi == hybrid (exact/approximate)
./scripts/validate_alignment.sh        # end-to-end: same, for Smith-Waterman
```

These are exactly the checks required by the spec's Section 21
(correctness validation): identical results across all four execution
modes are a prerequisite for any benchmark numbers meaning anything.

## 5. Benchmarking (Sections 22–27)

```bash
# Edit the config arrays at the top of this script to match YOUR
# hardware's core count / cluster node count before running.
./scripts/run_benchmarks.sh

python3 analysis/analyze_results.py     # -> results/processed/benchmark_summary.csv
python3 analysis/generate_graphs.py     # -> graphs/*.png (the 6 required graphs)
```

### ⚠️ Important: run benchmarks on real multi-core hardware

The correctness tests above are meaningful anywhere. **Performance
numbers are only meaningful on a machine with more than one physical
core** (and MPI benchmarks are only meaningful with more than one node,
or at least more cores than ranks). If you run `run_benchmarks.sh` on a
single-core machine or oversubscribed VM, you will see flat-to-negative
"speedup" — that's expected, not a bug (thread/rank counts beyond the
physical core count just add scheduling overhead). Run it on your
laptop, lab machine, or university cluster to get numbers worth putting
in the report. Never edit the benchmark numbers by hand — Section 42 of
the spec explicitly calls this out, and it's good practice regardless.

### Results already included

`results/raw/benchmark_results.csv`, `results/processed/benchmark_summary.csv`,
and all 18 graphs under `graphs/` in this repo are real output from a full
sweep run on a dedicated 32-logical-core node of MTSU's Babbage HPC
cluster (Slurm, `research-cpu` partition) — not placeholders. Peak
speedups at 32 workers: exact matching 25.2x (MPI), approximate matching
22.0x (MPI), Smith-Waterman batch alignment 12.9x (MPI). Two real bugs
were found and fixed using this data before it was finalized:

1. An OpenMP `schedule(dynamic, 4096)` clause in `approximate_match.hpp`
   on a loop with only `nthreads` iterations was silently handing the
   entire loop to a single thread — real hardware runs showed ~1.0x
   "speedup" at every thread count until this was caught and fixed to
   `schedule(dynamic, 1)` (see the comment at that line for the full
   diagnosis).
2. An MPI deadlock in `mpi_engine.hpp`, where non-root ranks read an
   uninitialized `MPI_Reduce` result — fixed with an explicit
   `MPI_Bcast` of the winning rank (see comment at that line).

Re-running `./scripts/run_benchmarks.sh` and
`./scripts/run_alignment_benchmarks.sh` on any other machine will
overwrite these CSVs with that machine's own numbers; the graphs can
then be regenerated with `analysis/generate_graphs.py` as described
above.

## 6. Project structure

```
parallel-dna-engine/
├── include/                  Shared algorithm headers (used by all 4 binaries)
│   ├── dna_common.hpp        FASTA parsing, dataset gen, timers, CLI args
│   ├── exact_match.hpp       Naive + KMP exact matching, partitioned w/ halo
│   ├── approximate_match.hpp Hamming-distance mutation-tolerant matching
│   └── smith_waterman.hpp    Wavefront-parallel local alignment + traceback
├── src/
│   ├── common/
│   │   ├── engine_main.hpp   Shared CLI driver for dna_seq / dna_openmp
│   │   └── mpi_engine.hpp    Shared CLI driver for dna_mpi / dna_hybrid
│   ├── sequential/main.cpp   Built WITHOUT -fopenmp
│   ├── openmp/main.cpp       Built WITH -fopenmp
│   ├── mpi/main.cpp          Built with mpic++ only
│   └── hybrid/main.cpp       Built with mpic++ -fopenmp
├── scripts/
│   ├── generate_dataset.py   Synthetic FASTA generator w/ known-match injection
│   ├── run_benchmarks.sh     Full benchmark sweep -> results/raw/*.csv
│   ├── validate_all_modes.sh   End-to-end correctness (exact/approximate)
│   └── validate_alignment.sh   End-to-end correctness (Smith-Waterman)
├── analysis/
│   ├── analyze_results.py    Raw reps -> median/speedup/efficiency CSV
│   └── generate_graphs.py    The 6 graphs from Spec section 27
├── tests/test_all.cpp        Unit-level correctness suite (make test)
├── report/design_decisions.md  Architecture rationale, scope decisions
└── Makefile
```

## 7. Scope notes (what's implemented vs. deferred)

Implemented (spec's "Core / Required" + most of "Strongly Desired",
Section 37):
- FASTA parsing, synthetic dataset generation w/ known-match injection
- Exact matching (naive + KMP), approximate (Hamming) matching, Smith–Waterman
- Sequential, OpenMP, MPI, and Hybrid implementations of all three algorithms
- Correctness validation across all four modes (automated)
- Automated benchmark sweep + speedup/efficiency analysis + all 6 graphs
- Scheduling comparison support (`schedule(static)`/`schedule(dynamic)` — see
  approximate_match.hpp; static vs dynamic vs guided can be swapped via a
  one-line pragma edit and re-benchmarked, documented in design_decisions.md)

Deferred as documented, out-of-core-scope decisions (Section 37's
"Optional Advanced Work", or explicitly noted trade-offs):
- Per-diagonal MPI halo-exchange for single-pair SW (batch distribution
  used instead — see design_decisions.md for why)
- Memory-optimized (rolling-row) Smith–Waterman
- CUDA/SIMD, edit-distance search, real multi-node HPC cluster run
  (code supports it via mpirun's hostfile; simply not exercised here)
- Real genomic (NCBI) dataset demonstration — the generator supports
  arbitrary sizes; drop a real FASTA file into `data/real/` and point
  `--input` at it, no code changes needed

None of these are required for a complete, correct, and benchmarkable
core system per the spec's own priority ordering (Section 42).
