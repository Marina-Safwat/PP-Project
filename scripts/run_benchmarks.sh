#!/usr/bin/env bash
# run_benchmarks.sh -- full benchmark sweep (Spec section 22), writing
# results/raw/benchmark_results.csv in the schema from Spec section 32:
#   algorithm,mode,dataset_size,mpi_ranks,threads,time,speedup,efficiency
#
# Run this on the REAL multi-core hardware / cluster you intend to report
# numbers from -- NOT inside a 1-core sandbox, where "speedup" is
# meaningless (see README "Benchmarking caveat"). Edit the arrays below
# to match what your machine can actually support.
#
# Usage:
#   ./scripts/run_benchmarks.sh
#   REPS=5 DATASETS="1MB 10MB 100MB" ./scripts/run_benchmarks.sh
set -euo pipefail
cd "$(dirname "$0")/.."

BIN=bin
OUT=results/raw/benchmark_results.csv
QUERY=${QUERY:-ACGTAGCTAGCTAACGT}
MISMATCHES=${MISMATCHES:-2}
REPS=${REPS:-5}

# ---- Configurations to sweep (edit for your hardware, Spec section 22) ----
DATASETS=${DATASETS:-"1MB 10MB 100MB"}
declare -A DATASET_BYTES=( ["1MB"]=1000000 ["10MB"]=10000000 ["100MB"]=100000000
                           ["500MB"]=500000000 ["1GB"]=1000000000 )

OMP_THREADS=${OMP_THREADS:-"1 2 4 8"}
MPI_RANKS=${MPI_RANKS:-"2 4"}
HYBRID_CONFIGS=${HYBRID_CONFIGS:-"2x2 2x4 4x2"} # MPIxOpenMP

if [ "$(id -u)" = "0" ]; then
    export OMPI_ALLOW_RUN_AS_ROOT=1
    export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
fi

mkdir -p results/raw data/synthetic
echo "algorithm,mode,dataset_size,mpi_ranks,threads,rep,time,throughput_mb_s" > "$OUT"

median() {
    # reads numbers on stdin, prints the median
    python3 -c "import sys,statistics; xs=[float(x) for x in sys.stdin]; print(statistics.median(xs))"
}

extract_time() {
    grep -E "^Execution Time" | grep -oE "[0-9]+\.[0-9]+"
}

run_and_log() {
    local algorithm="$1" mode="$2" dsize="$3" ranks="$4" threads="$5"; shift 5
    for rep in $(seq 1 "$REPS"); do
        local t
        t=$("$@" | extract_time)
        [ -z "$t" ] && t=0
        local mb=$(python3 -c "print(${DATASET_BYTES[$dsize]}/1e6/${t:-1} if ${t:-0} > 0 else 0)")
        echo "$algorithm,$mode,$dsize,$ranks,$threads,$rep,$t,$mb" >> "$OUT"
        echo "  [$algorithm/$mode ds=$dsize ranks=$ranks threads=$threads rep=$rep] time=${t}s"
    done
}

for dsize in $DATASETS; do
    bytes=${DATASET_BYTES[$dsize]}
    dataset_path="data/synthetic/bench_${dsize}.fasta"
    if [ ! -f "$dataset_path" ]; then
        echo ">> Generating $dsize dataset ..."
        python3 scripts/generate_dataset.py --size "$bytes" --output "$dataset_path" \
            --inject-query "$QUERY" --inject-count 50 --seed 42
    fi

    for algorithm in exact approximate; do
        extra_args=()
        [ "$algorithm" = "approximate" ] && extra_args=(--mismatches "$MISMATCHES")

        # Sequential baseline (threads=1, ranks=1)
        run_and_log "$algorithm" sequential "$dsize" 1 1 \
            "$BIN/dna_seq" --input "$dataset_path" --query "$QUERY" --algorithm "$algorithm" "${extra_args[@]}"

        # OpenMP sweep
        for t in $OMP_THREADS; do
            run_and_log "$algorithm" openmp "$dsize" 1 "$t" \
                "$BIN/dna_openmp" --input "$dataset_path" --query "$QUERY" --algorithm "$algorithm" --threads "$t" "${extra_args[@]}"
        done

        # MPI sweep
        for r in $MPI_RANKS; do
            run_and_log "$algorithm" mpi "$dsize" "$r" 1 \
                mpirun --oversubscribe -np "$r" "$BIN/dna_mpi" --input "$dataset_path" --query "$QUERY" --algorithm "$algorithm" "${extra_args[@]}"
        done

        # Hybrid sweep
        for cfg in $HYBRID_CONFIGS; do
            r=${cfg%x*}; t=${cfg#*x}
            run_and_log "$algorithm" hybrid "$dsize" "$r" "$t" \
                mpirun --oversubscribe -np "$r" "$BIN/dna_hybrid" --input "$dataset_path" --query "$QUERY" --algorithm "$algorithm" --threads "$t" "${extra_args[@]}"
        done
    done
done

echo ""
echo ">> Raw results written to $OUT"
echo ">> Next: python3 analysis/analyze_results.py"
