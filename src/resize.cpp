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
    field_descriptor *f0, *f1, *f2;

    if (argc != 4)
    {
        MPI_Finalize();
        printf("wrong number of parameters.\n");
        return EXIT_SUCCESS;
    }

    n[0] = atoi(argv[1]);
    n[1] = atoi(argv[2]);
    n[2] = atoi(argv[3]);
    f0 = new field_descriptor(3, n, MPI_COMPLEX8);
    n[0] = 2*atoi(argv[1]);
    n[1] = 2*atoi(argv[2]);
    n[2] = 2*(atoi(argv[3])-1)+1;
    f1 = new field_descriptor(3, n, MPI_COMPLEX8);
    n[0] = f1->sizes[0];
    n[1] = f1->sizes[1];
    n[2] = 2*f1->sizes[2];
    f2 = new field_descriptor(3, n, MPI_REAL4);

    fftwf_complex *a0, *a1;
    float *a2;
    a0 = fftwf_alloc_complex(f0->local_size);
    a1 = fftwf_alloc_complex(f1->local_size);
    a2 = fftwf_alloc_real(f2->local_size);

    f0->read("data0", (void*)a0);

    fftwf_copy_complex_array(
            f0, a0,
            f1, a1);

    fftwf_plan c2r = fftwf_mpi_plan_dft_c2r_3d(
            f2->sizes[0], f2->sizes[1], f2->sizes[2]-2,
            a1, a2,
            MPI_COMM_WORLD,
            FFTW_ESTIMATE);
    fftwf_execute(c2r);
    f2->write("data2", (void*)a2);
    fftw_free(a0);
    fftw_free(a1);
    fftw_free(a2);

    delete f0;
    delete f1;
    delete f2;
    MPI_Finalize();
    return EXIT_SUCCESS;
}

