#include "p3DFFT_to_iR.hpp"
#include "Morton_shuffler.hpp"
#include <iostream>

int myrank, nprocs;

const int iter0 = 138000;

int get_RMHD_names(
        int iteration,
        bool velocity,
        char **fname)
{
    if (velocity)
    {
        sprintf(fname[0], "K%.6dQNP002", iteration);
        sprintf(fname[1], "K%.6dQNP003", iteration);
    }
    else
    {
        sprintf(fname[0], "K%.6dQNP005", iteration);
        sprintf(fname[1], "K%.6dQNP006", iteration);
    }
    return EXIT_SUCCESS;
}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    const int n = 1364;
    const int N = 2048;
    int nfiles = nprocs % 64;
    int iteration;
    if (argc == 2)
    {
        iteration = atoi(argv[1]);
    }
    else
    {
        std::cerr <<
            "not enough (or too many) parameters.\naborting." <<
            std::endl;
        MPI_Finalize();
        return EXIT_SUCCESS;
    }
    p3DFFT_to_iR *r = new p3DFFT_to_iR(
            (n/2+1), n, n,
            N, N, N,
            2);
    Morton_shuffler *s = new Morton_shuffler(
            N, N, N, 2, nfiles);

    // initialize file names
    char *ifile[2];
    for (int i=0; i<2; i++)
        ifile[i] = (char*)malloc(sizeof(char)*100);

    // velocity
    get_RMHD_names(iteration, true, ifile);
    r->read(ifile);
    sprintf(ifile[0], "u_t%.3x", iteration - iter0);
    s->shuffle(r->r3, ifile[0]);
    // magnetic
    get_RMHD_names(iteration, false, ifile);
    r->read(ifile);
    sprintf(ifile[0], "b_t%.3x", iteration - iter0);
    s->shuffle(r->r3, ifile[0]);

    //free file names
    for (int i=0; i<2; i++)
        free(ifile[i]);

    delete s;
    delete r;

    // clean up
    fftwf_mpi_cleanup();
    MPI_Finalize();
    return EXIT_SUCCESS;
}

