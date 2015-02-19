#include "RMHD_converter.hpp"
#include <iostream>

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
//            (1364/2+1), 1364, 1364,
//            2048, 2048, 2048,
//            64);
    bla->convert("Kdata0", "Kdata1", "Rdata");

    //bla->convert("K138000QNP002",
    //             "K138000QNP003",
    //             "U138000");
    //bla->convert("K138000QNP005",
    //             "K138000QNP006",
    //             "B138000");
    delete bla;

    // clean up
    fftwf_mpi_cleanup();
    MPI_Finalize();
    return EXIT_SUCCESS;
}

