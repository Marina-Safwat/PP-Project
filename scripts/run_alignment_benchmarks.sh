#!/usr/bin/env bash
# run_alignment_benchmarks.sh -- benchmark sweep for Smith-Waterman batch
# alignment (Spec sections 14-16, 22), appending to the SAME raw CSV
# schema as run_benchmarks.sh so analyze_results.py / generate_graphs.py
# can summarize exact/approximate/align together (this is what fills in
# graph 6, "Exact vs Approximate vs Alignment").
#
# Unlike exact/approximate (one long genome, partitioned across
# workers), Smith-Waterman here uses batch alignment: a database of many
# short sequences, aligned against one query, distributed across
# ranks/threads (see report/design_decisions.md section 3 for why).
# "dataset_size" in the CSV is therefore labeled by record count, not
# bytes, and there is no throughput (MB/s) figure for this algorithm --
# the CSV's throughput_mb_s column is left as 0 for these rows.
#
# Usage:
#   ./scripts/run_alignment_benchmarks.sh
#   REPS=5 RECORD_COUNTS="200 500" ./scripts/run_alignment_benchmarks.sh
set -euo pipefail
cd "$(dirname "$0")/.."

BIN=bin
OUT=results/raw/benchmark_results.csv
QUERY=${QUERY:-ACGTTAGCTAGGCATCGTAACGTTAGCTAGG}
REPS=${REPS:-5}
RECORD_COUNTS=${RECORD_COUNTS:-"200 500"}
RECORD_LEN=${RECORD_LEN:-2000}

OMP_THREADS=${OMP_THREADS:-"1 2 4 8 16 32"}
MPI_RANKS=${MPI_RANKS:-"2 4 8 16 32"}
HYBRID_CONFIGS=${HYBRID_CONFIGS:-"2x2 2x4 2x8 4x2 4x4 4x8 8x2 8x4 16x2"}

if [ "$(id -u)" = "0" ]; then
    export OMPI_ALLOW_RUN_AS_ROOT=1
    export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
fi

mkdir -p results/raw data/synthetic

# Append to the existing CSV if present (from run_benchmarks.sh), else
# create it fresh with the shared header.
if [ ! -f "$OUT" ]; then
    echo "algorithm,mode,dataset_size,mpi_ranks,threads,rep,time,throughput_mb_s" > "$OUT"
fi

extract_time() {
    grep -E "^Execution Time" | grep -oE "[0-9]+\.[0-9]+" | head -1
}

run_and_log() {
    local mode="$1" dsize="$2" ranks="$3" threads="$4"; shift 4
    for rep in $(seq 1 "$REPS"); do
        local t
        t=$("$@" | extract_time)
        [ -z "$t" ] && t=0
        echo "align,$mode,$dsize,$ranks,$threads,$rep,$t,0" >> "$OUT"
        echo "  [align/$mode ds=$dsize ranks=$ranks threads=$threads rep=$rep] time=${t}s"
    done
}

for nrec in $RECORD_COUNTS; do
    dsize="align_${nrec}recs"
    db_path="data/synthetic/align_db_${nrec}.fasta"
    if [ ! -f "$db_path" ]; then
        echo ">> Generating alignment database: $nrec records x $RECORD_LEN bases ..."
        python3 scripts/generate_dataset.py --size "$RECORD_LEN" --records "$nrec" \
            --output "$db_path" --seed 42
    fi

    # Sequential/OpenMP align the SAME full batch as MPI/hybrid (see the
    # fix in src/common/engine_main.hpp), so all four modes' timings are
    # directly comparable for speedup/efficiency purposes.
    run_and_log sequential "$dsize" 1 1 \
        "$BIN/dna_seq" --input "$db_path" --query "$QUERY" --algorithm align

    for t in $OMP_THREADS; do
        run_and_log openmp "$dsize" 1 "$t" \
            "$BIN/dna_openmp" --input "$db_path" --query "$QUERY" --algorithm align --threads "$t"
    done

    for r in $MPI_RANKS; do
        run_and_log mpi "$dsize" "$r" 1 \
            mpirun --oversubscribe -np "$r" "$BIN/dna_mpi" --input "$db_path" --query "$QUERY" --algorithm align
    done

    for cfg in $HYBRID_CONFIGS; do
        r=${cfg%x*}; t=${cfg#*x}
        run_and_log hybrid "$dsize" "$r" "$t" \
            mpirun --oversubscribe -np "$r" "$BIN/dna_hybrid" --input "$db_path" --query "$QUERY" --algorithm align --threads "$t"
    done
done

echo ""
echo ">> Alignment benchmark rows appended to $OUT"
echo ">> Next: python3 analysis/analyze_results.py && python3 analysis/generate_graphs.py"
