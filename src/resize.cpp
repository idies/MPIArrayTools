#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include "field_descriptor.hpp"
#include "fftwf_tools.hpp"

int myrank, nprocs;

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    fftwf_mpi_init();

    field_descriptor *f0r=NULL, *f0c=NULL;
    field_descriptor *f1r=NULL, *f1c=NULL;

    if (argc != 4)
    {
        MPI_Finalize();
        printf("wrong number of parameters.\n");
        return EXIT_SUCCESS;
    }

    fftwf_get_descriptors_3D(
            atoi(argv[1]), atoi(argv[2]), atoi(argv[3]),
            &f0r, &f0c);
    fftwf_get_descriptors_3D(
            2*atoi(argv[1]), 2*atoi(argv[2]), 2*atoi(argv[3]),
            &f1r, &f1c);

    fftwf_complex *a0, *a1c;
    float *a1r;
    a0 = fftwf_alloc_complex(f0c->local_size);
    a1c = fftwf_alloc_complex(f1c->local_size);
    a1r = fftwf_alloc_real(2*f1c->local_size);

    f0c->read("data0c", (void*)a0);

    fftwf_plan c2r = fftwf_mpi_plan_dft_c2r_3d(
            f0r->sizes[0], f0r->sizes[1], f0r->sizes[2],
            a0, a1r,
            MPI_COMM_WORLD,
            FFTW_ESTIMATE);
    fftwf_execute(c2r);
    fftwf_destroy_plan(c2r);
    fftwf_clip_zero_padding(f0r, a1r);
    f0r->write("data0r", (void*)a1r);

    fftwf_copy_complex_array(
            f0c, a0,
            f1c, a1c);

    c2r = fftwf_mpi_plan_dft_c2r_3d(
            f1r->sizes[0], f1r->sizes[1], f1r->sizes[2],
            a1c, a1r,
            MPI_COMM_WORLD,
            FFTW_ESTIMATE);
    fftwf_execute(c2r);
    fftwf_destroy_plan(c2r);

    fftwf_clip_zero_padding(f1r, a1r);
    f1r->write("data1r", (void*)a1r);
    fftw_free(a0);
    fftw_free(a1c);
    fftw_free(a1r);
    fftwf_mpi_cleanup();

    delete f0r;
    delete f0c;
    delete f1r;
    delete f1c;
    MPI_Finalize();
    return EXIT_SUCCESS;
}

