#!/usr/bin/env bash
# Validates Smith-Waterman correctness across all four modes:
#   - Single-pair alignment: dna_seq vs dna_openmp must agree on score
#     (they share the exact same wavefront code, just single- vs
#     multi-threaded -- see include/smith_waterman.hpp).
#   - Batch alignment: dna_mpi vs dna_hybrid must agree on which
#     database sequence is the best match and its score (inter-sequence
#     parallelism across ranks, see src/common/mpi_engine.hpp).
set -euo pipefail
cd "$(dirname "$0")/.."

BIN=bin
if [ "$(id -u)" = "0" ]; then
    export OMPI_ALLOW_RUN_AS_ROOT=1
    export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
fi

DB=data/synthetic/align_validation_db.fasta
QUERY=ACGTTAGCTAGGCATCG

if [ ! -f "$DB" ]; then
    echo ">> Generating alignment validation database ..."
    python3 scripts/generate_dataset.py --size 500 --records 40 --output "$DB" --seed 55
fi

pass=0; fail=0
check_equal() {
    local name="$1" a="$2" b="$3"
    if [ "$a" == "$b" ]; then echo "  [OK]   $name ($a == $b)"; pass=$((pass+1));
    else echo "  [FAIL] $name ($a != $b)"; fail=$((fail+1)); fi
}

echo "### Single-pair alignment: sequential vs OpenMP (best match across the small DB) ###"
seq_score=$($BIN/dna_seq --input "$DB" --query "$QUERY" --algorithm align | grep "^Best match" | grep -oE "score = [0-9-]+" | grep -oE "[0-9-]+")
omp_score=$($BIN/dna_openmp --input "$DB" --query "$QUERY" --algorithm align --threads 4 | grep "^Best match" | grep -oE "score = [0-9-]+" | grep -oE "[0-9-]+")
check_equal "SW score: seq vs openmp" "$seq_score" "$omp_score"

echo ""
echo "### Batch alignment: MPI vs Hybrid (best-matching record across DB) ###"
mpi_line=$(mpirun --oversubscribe -np 4 $BIN/dna_mpi --input "$DB" --query "$QUERY" --algorithm align | grep "^Best match")
hyb_line=$(mpirun --oversubscribe -np 3 $BIN/dna_hybrid --input "$DB" --query "$QUERY" --algorithm align --threads 2 | grep "^Best match")
check_equal "SW batch best-match line: mpi vs hybrid" "$mpi_line" "$hyb_line"

echo ""
echo "=================================================="
echo "  $pass passed, $fail failed"
echo "=================================================="
exit $fail
