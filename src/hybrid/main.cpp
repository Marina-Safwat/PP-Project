// Hybrid MPI + OpenMP executable (dna_hybrid).
// Identical driver to dna_mpi (same mpi_engine.hpp), but built with
// `mpic++ -fopenmp`, so each rank ALSO spawns an OpenMP thread team
// (size given by --threads) for the local work it owns -- MPI ranks
// give distributed-memory parallelism across nodes, OpenMP threads give
// shared-memory parallelism within each rank's share of the work
// (Spec section 18).
#include <mpi.h>
#include "mpi_engine.hpp"

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rc = dna::run_mpi_cli(argc, argv);
    MPI_Finalize();
    return rc;
}
