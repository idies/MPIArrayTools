#include "p3DFFT_to_iR.hpp"
#include "Morton_shuffler.hpp"
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
    p3DFFT_to_iR *r = new p3DFFT_to_iR(
            (n/2+1), n, n,
            N, N, N,
            2);

    // initialize file names
    char **ifile;
    ifile = (char**)malloc(2*sizeof(char*));
    for (int i; i<2; i++)
    {
        ifile[i] = (char*)malloc(100*sizeof(char));
        sprintf(ifile[i], "Kdata%d", i);
    }

    //read
    r->read(ifile);

    //free file names
    for (int i; i<2; i++)
        free(ifile[i]);
    free(ifile);
    Morton_shuffler *s = new Morton_shuffler(
            N, N, N, 2, nfiles);
    s->shuffle(r->r3, "Rdata");

    delete s;
    delete r;

    // clean up
    fftwf_mpi_cleanup();
    MPI_Finalize();
    return EXIT_SUCCESS;
}

