#include <stdio.h>
#include <stdlib.h>
#include "field_descriptor.hpp"

int myrank, nprocs;

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    int n[3];
    field_descriptor *f0, *f1;

    switch(argc)
    {
        case 3:
            n[0] = atoi(argv[1]);
            n[1] = atoi(argv[2]);
            break;
        case 4:
            n[0] = atoi(argv[1]);
            n[1] = atoi(argv[2]);
            n[2] = atoi(argv[3]);
            break;
        default:
            printf("you messed up the parameters, I'm not doing anything.\n");
            MPI_Finalize();
            return EXIT_SUCCESS;
            break;
    }
    if (myrank == 0)
        printf( "transposing %dD array from \"data0\" into \"data1\""
                " with %d processes.\n",
                argc - 1,
                nprocs);
    f0 = new field_descriptor(argc - 1, n, MPI_FLOAT);
    f1 = f0->get_transpose();

    float *a0, *a1;
    a0 = fftwf_alloc_real(f0->local_size);
    a1 = fftwf_alloc_real(f1->local_size);
    f0->read("data0", (void*)a0);
    f0->transpose(a0, a1);
    f1->write("data1", (void*)a1);
    fftw_free(a0);
    fftw_free(a1);

    delete f0;
    delete f1;
    MPI_Finalize();
    return EXIT_SUCCESS;
}

