#include "RMHD_converter.hpp"

int myrank, nprocs;

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    RMHD_converter *bla = new RMHD_converter(
            atoi(argv[1]), atoi(argv[2]), atoi(argv[3]),
            atoi(argv[4]), atoi(argv[5]), atoi(argv[6]),
            2);

    bla->convert("Kdata0", "Kdata1", "Rdata");
    delete bla;

    // clean up
    fftwf_mpi_cleanup();
    MPI_Finalize();
    return EXIT_SUCCESS;
}

