#include "RMHD_converter.hpp"
#include <iostream>

int myrank, nprocs;

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    int n, N, nfiles;
    if (argc == 4)
    {
        n = atoi(argv[1]);
        N = atoi(argv[2]);
        nfiles = atoi(argv[3]);
    }
    else
    {
        std::cerr <<
            "not enough (or too many) parameters.\naborting." <<
            std::endl;
        MPI_Finalize();
        return EXIT_SUCCESS;
    }
    RMHD_converter *bla = new RMHD_converter(
            (n/2+1), n, n,
            N, N, N,
            nfiles);
    bla->convert("Kdata0", "Kdata1", "Rdata");

    delete bla;

    // clean up
    fftwf_mpi_cleanup();
    MPI_Finalize();
    return EXIT_SUCCESS;
}

