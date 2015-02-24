#include "RMHD_converter.hpp"
#include <iostream>

int myrank, nprocs;
const int iter0 = 138000;

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    int n, N, nfiles;
    n = 1364;
    N = 2048;
    nfiles = 64;
    char iname0[100], iname1[100], oname[100];
    RMHD_converter *bla = new RMHD_converter(
            (n/2+1), n, n,
            N, N, N,
            nfiles);
 //   bla->convert("Kdata0", "Kdata1", "Rdata");

    int iteration = 138000;
    sprintf(iname0, "K%.6dQNP002", iteration - iter0);
    sprintf(iname1, "K%.6dQNP003", iteration - iter0);
    sprintf(oname, "u_t%.3x", iteration - iter0);
    bla->convert(iname0, iname1, oname);
    sprintf(iname0, "K%.6dQNP005", iteration - iter0);
    sprintf(iname1, "K%.6dQNP006", iteration - iter0);
    sprintf(oname, "b_t%.3x", iteration - iter0);
    bla->convert(iname0, iname1, oname);
    delete bla;

    // clean up
    fftwf_mpi_cleanup();
    MPI_Finalize();
    return EXIT_SUCCESS;
}

