// OpenMP shared-memory executable (dna_openmp).
// Same driver as the sequential binary; built with -fopenmp so the
// #pragma omp for directives in exact_match.hpp / approximate_match.hpp /
// smith_waterman.hpp actually spawn a thread team of size --threads.
#include "engine_main.hpp"

int main(int argc, char** argv) {
    return dna::run_engine_cli(argc, argv);
}
