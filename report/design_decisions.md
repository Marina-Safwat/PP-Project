# Design Decisions

This document explains the *why* behind the architecture, so it can be
lifted fairly directly into the final report's "Parallel Design" /
"Implementation" / "Limitations" sections (Spec section 11, phase 11).

## 1. One shared algorithm implementation for sequential + OpenMP

`include/{exact_match,approximate_match,smith_waterman}.hpp` contain a
single implementation of each algorithm, guarded by `#pragma omp`
directives. `src/sequential/main.cpp` is compiled *without* `-fopenmp`;
`src/openmp/main.cpp` is compiled *with* it. Without the flag, every
`#pragma omp` line is a no-op (the compiler emits a harmless "ignoring
unknown pragma" warning) and the code runs single-threaded.

This is a deliberate correctness-by-construction choice: it is
*impossible* for the sequential and OpenMP binaries to implement
different algorithmic logic, because they are the same source file. The
only question benchmarking has to answer is "how much does adding
threads help", not "did we implement the same thing twice, correctly,
both times." The same reasoning extends to `dna_mpi`/`dna_hybrid`
(`src/mpi/main.cpp` vs `src/hybrid/main.cpp`, differing only in whether
`-fopenmp` is added to the `mpic++` build).

## 2. Partitioning strategy: exact vs. approximate matching

Both use the same "contiguous block per worker" decomposition
(`partition_bounds()` in `mpi_engine.hpp`, mirrored at the OpenMP level
inside each algorithm header): start positions `[0, num_start_positions)`
are split into `p` equal-ish contiguous ranges.

**Exact matching needs a halo.** A `TAGCTA` match starting at the last
few bases of worker A's range extends into worker B's territory. Per
Spec section 11, each worker's local buffer is extended by
`pattern_length - 1` bases past its owned range, and only matches whose
*start* position falls inside the worker's own owned range are kept
(this prevents double-counting when both A and B could otherwise "see"
the same match from their overlapping halos).

**Approximate (Hamming) matching does not need a halo for boundary
matches at the OpenMP level**, because it evaluates fixed-length windows
whose start position defines the entire window — but it *does* still
need the same halo formula at the MPI level, because a window itself
can straddle two ranks' locally-held buffers if the buffers were split
without the extra `pattern_length - 1` bases. `mpi_engine.hpp` gives
both algorithms the same halo treatment via one shared
`local_len = owned_len + (pattern_length - 1)` calculation — simpler to
maintain and it costs almost nothing (halo is O(pattern length), which
is tiny compared to genome/partition size).

## 3. Why Smith-Waterman is not wavefront-parallelized across MPI ranks

The OpenMP implementation parallelizes Smith-Waterman *within* a single
alignment via anti-diagonal wavefronts (Spec section 15): cells on
diagonal `d` depend only on diagonals `d-1` and `d-2`, so a diagonal's
cells are computed with a `#pragma omp parallel for`, with an implicit
barrier before the next diagonal starts.

A "textbook-complete" MPI version would extend this by partitioning
matrix *rows* across ranks and exchanging one boundary row via
`MPI_Sendrecv` after every diagonal. This is algorithmically correct but
was deliberately **not** implemented, for a concrete, quantifiable
reason: for an alignment of length `N`, it requires on the order of `N`
separate point-to-point message exchanges between neighboring ranks —
one per diagonal — each carrying only a few bytes. Message *latency*
(not bandwidth) dominates at that granularity, so this design would
demonstrate communication overhead overwhelming useful work long before
any real speedup shows up, for any alignment short enough to be
practical to benchmark repeatedly.

Instead, `mpi_engine.hpp` implements what the spec's own Section 15
identifies as the standard real-world MPI approach for this exact
problem: **"parallelizing multiple alignments"** — MPI distributes a
*batch* of database sequences (all aligned against one query) across
ranks (inter-sequence / embarrassingly-parallel decomposition), and each
rank optionally still uses OpenMP to wavefront-parallelize *each
individual alignment it owns* (intra-sequence parallelism) when running
as `dna_hybrid`. This gives a genuinely two-level parallel hierarchy
(exactly Spec section 18's diagram) without the message-rate problem
above, and is directly usable for realistic workloads like "search a
database of candidate sequences for the best match to this query."

**Trade-off to state plainly in the report:** this means MPI/hybrid
Smith-Waterman speedup is bounded by the number of *database sequences*,
not by the length of any single alignment. A database of 4 sequences run
across 8 MPI ranks will not use more than 4 ranks productively. This is
a real, known limitation of batch-style parallelism and worth an
explicit sentence in the report's Limitations section — it is not a bug.

## 4. Correctness validation strategy (Spec section 21)

Two layers:

1. **`tests/test_all.cpp`** checks the *algorithm headers* directly
   against independent brute-force references, across
   `nthreads ∈ {1,2,3,4,5,7,8,16}`. This isolates "is the parallel
   algorithm logic correct" from "does the CLI/MPI plumbing work",
   and it is fast enough to run on every change (`make test`).
2. **`scripts/validate_all_modes.sh` / `validate_alignment.sh`** run the
   actual compiled `dna_seq`/`dna_openmp`/`dna_mpi`/`dna_hybrid` binaries
   end-to-end on the same input and diff their reported results. This
   is the layer that would have caught the one real bug found during
   development (see below) — a bug purely in MPI plumbing, invisible to
   unit tests of the algorithm headers alone.

**Bug found and fixed during development:** the original MPI batch
Smith-Waterman code checked `rank == global_ml.rank` (the MAXLOC result
from `MPI_Reduce`) to decide which rank should send its winning
alignment to rank 0. `MPI_Reduce` only populates its receive buffer on
the *root* rank — on every other rank, `global_ml` was uninitialized
garbage, so the "am I the winner" check was comparing against garbage
memory on all non-root ranks. This caused a deadlock: rank 0 waited
forever on an `MPI_Recv` from the true winning rank, which never knew it
was supposed to send. Fixed by adding an `MPI_Bcast` of the winning
rank's ID immediately after the reduce, so every rank agrees on who the
winner is before the point-to-point send/receive happens. This is a
useful, concrete example for the report's "problems encountered" /
"synchronization pitfalls" discussion (Spec section 34).

## 5. Scheduling comparison (Spec section 26)

`approximate_match.hpp`'s OpenMP loop uses `schedule(dynamic, 4096)`
(chunks of 4096 windows) because Hamming-distance work per window is
data-dependent (early-exit on the first mismatch beyond the allowed
threshold means some windows finish much faster than others — this is
exactly the load-imbalance scenario Spec section 26 asks you to
investigate). `exact_match.hpp`'s KMP/naive loop uses
`schedule(static)` because each partition's work is nearly uniform
(proportional to partition length). To run the static-vs-dynamic-vs-
guided experiment called for in Section 26, change the `schedule(...)`
clause in the relevant header, rebuild, and re-run
`run_benchmarks.sh` — the CSV/graph pipeline needs no changes, just tag
the `mode` column manually if you want to compare multiple scheduling
policies in the same summary file.

## 6. What "Hybrid" measures

For search algorithms (exact/approximate), "hybrid" = MPI partitions the
genome across ranks (as in section 2 above), and within each rank's
local buffer, OpenMP further partitions the work across `--threads`
threads (same partition-with-halo logic, applied one level down). For
alignment, "hybrid" = MPI partitions the *database* across ranks (section
3 above), and OpenMP wavefront-parallelizes each rank's individual
alignments. In both cases, "workers" for speedup/efficiency purposes
(Spec section 23) is `mpi_ranks × threads` — this is how
`analysis/analyze_results.py` computes efficiency for hybrid
configurations on the same axis as pure-OpenMP or pure-MPI ones.
