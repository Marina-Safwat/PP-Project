#!/usr/bin/env bash
# Validates Spec section 21's core invariant end-to-end, against the
# actual compiled binaries (not just the shared header logic, which
# tests/test_all.cpp already covers): for identical input,
#   Sequential result == OpenMP result == MPI result == Hybrid result
#
# Usage: ./scripts/validate_all_modes.sh [path/to/dataset.fasta]
# Defaults to generating a fresh synthetic dataset if none is given.
set -euo pipefail
cd "$(dirname "$0")/.."

BIN=bin
DATA=${1:-}
QUERY=${QUERY:-ACGTAGCTAG}
MISMATCHES=${MISMATCHES:-2}

# Allow this script to also run correctly inside containers where MPI
# refuses to run as root; harmless on normal machines/clusters.
if [ "$(id -u)" = "0" ]; then
    export OMPI_ALLOW_RUN_AS_ROOT=1
    export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
fi

if [ -z "$DATA" ]; then
    DATA=data/synthetic/validation_dataset.fasta
    echo ">> No dataset given, generating one at $DATA"
    python3 scripts/generate_dataset.py --size 2000000 --output "$DATA" \
        --inject-query "$QUERY" --inject-count 25 --seed 123
fi

echo ">> Dataset: $DATA"
echo ">> Query:   $QUERY"
echo ""

run() {
    local label="$1"; shift
    echo "--- $label ---"
    "$@"
}

extract_matches() {
    grep "^Matches found:" | awk '{print $3}'
}

pass=0
fail=0

check_equal() {
    local name="$1" a="$2" b="$3"
    if [ "$a" == "$b" ]; then
        echo "  [OK]   $name ($a == $b)"
        pass=$((pass+1))
    else
        echo "  [FAIL] $name ($a != $b)"
        fail=$((fail+1))
    fi
}

echo "### Exact matching ###"
seq_out=$($BIN/dna_seq --input "$DATA" --query "$QUERY" --algorithm exact)
omp_out=$($BIN/dna_openmp --input "$DATA" --query "$QUERY" --algorithm exact --threads 4)
mpi_out=$(mpirun --oversubscribe -np 4 $BIN/dna_mpi --input "$DATA" --query "$QUERY" --algorithm exact)
hyb_out=$(mpirun --oversubscribe -np 2 $BIN/dna_hybrid --input "$DATA" --query "$QUERY" --algorithm exact --threads 2)

seq_n=$(echo "$seq_out" | extract_matches)
omp_n=$(echo "$omp_out" | extract_matches)
mpi_n=$(echo "$mpi_out" | extract_matches)
hyb_n=$(echo "$hyb_out" | extract_matches)

check_equal "exact: seq vs openmp" "$seq_n" "$omp_n"
check_equal "exact: seq vs mpi"    "$seq_n" "$mpi_n"
check_equal "exact: seq vs hybrid" "$seq_n" "$hyb_n"

echo ""
echo "### Approximate matching (<= $MISMATCHES mismatches) ###"
seq_out=$($BIN/dna_seq --input "$DATA" --query "$QUERY" --algorithm approximate --mismatches $MISMATCHES)
omp_out=$($BIN/dna_openmp --input "$DATA" --query "$QUERY" --algorithm approximate --mismatches $MISMATCHES --threads 4)
mpi_out=$(mpirun --oversubscribe -np 4 $BIN/dna_mpi --input "$DATA" --query "$QUERY" --algorithm approximate --mismatches $MISMATCHES)
hyb_out=$(mpirun --oversubscribe -np 2 $BIN/dna_hybrid --input "$DATA" --query "$QUERY" --algorithm approximate --mismatches $MISMATCHES --threads 2)

seq_n=$(echo "$seq_out" | grep -E "^Total Matches" | grep -oE "[0-9]+$")
omp_n=$(echo "$omp_out" | grep -E "^Total Matches" | grep -oE "[0-9]+$")
mpi_n=$(echo "$mpi_out" | extract_matches)
hyb_n=$(echo "$hyb_out" | extract_matches)

check_equal "approx: seq vs openmp" "$seq_n" "$omp_n"
check_equal "approx: seq vs mpi"    "$seq_n" "$mpi_n"
check_equal "approx: seq vs hybrid" "$seq_n" "$hyb_n"

echo ""
echo "=================================================="
echo "  $pass passed, $fail failed"
echo "=================================================="
exit $fail
