// Sequential baseline executable (dna_seq).
// Deliberately built without -fopenmp so every #pragma omp in the shared
// headers is inert -- this guarantees the sequential baseline runs the
// exact same algorithmic code path as the OpenMP binary, just single
// threaded, which is what makes the correctness-validation step
// (Spec section 21) meaningful.
#include "engine_main.hpp"

int main(int argc, char** argv) {
    return dna::run_engine_cli(argc, argv);
}
