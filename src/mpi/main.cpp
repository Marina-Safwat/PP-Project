// Pure MPI distributed-memory executable (dna_mpi).
// Built with mpic++ only (no -fopenmp), so each rank runs single
// threaded: parallelism here comes entirely from process-level
// distribution across ranks (see mpi_engine.hpp for the two
// partitioning strategies used).
#include <mpi.h>
#include "mpi_engine.hpp"

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rc = dna::run_mpi_cli(argc, argv);
    MPI_Finalize();
    return rc;
}
