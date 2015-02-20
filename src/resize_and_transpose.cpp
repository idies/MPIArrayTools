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

    field_descriptor *f0c = NULL; // 2D input descriptor, we need to transpose it
    field_descriptor *f1c = NULL; // descriptor for 3D fully transposed input
    field_descriptor *f2c = NULL, *f2r=NULL; // descriptors for output

    if (argc != 7)
    {
        MPI_Finalize();
        printf("wrong number of parameters.\n");
        return EXIT_SUCCESS;
    }

    // first 3 arguments are dimensions for input array
    // i.e. actual dimensions for the Fourier representation.
    // NOT real space grid dimensions
    int ni[3];

    // the input array is read in as a 2D array,
    // since the first dimension must be a multiple of nprocs
    // (which is generally an even number)
    ni[0] = atoi(argv[1])*atoi(argv[2]);
    ni[1] = atoi(argv[3]);
    f0c = new field_descriptor(2, ni, MPI_COMPLEX8, MPI_COMM_WORLD);
    // f1c will be pointing at the input array after it has been
    // transposed, therefore we have this correspondence:
    // f0c->sizes[0] = f1c->sizes[1]*f1c->sizes[2]
    ni[0] = atoi(argv[3]);
    ni[1] = atoi(argv[1]);
    ni[2] = atoi(argv[2]);
    f1c = new field_descriptor(3, ni, MPI_COMPLEX8, MPI_COMM_WORLD);

    fftwf_complex *c0, *c1, *c2;
    c0 = fftwf_alloc_complex(f0c->local_size);
    f0c->read("data0c", (void*)c0);
    c1 = fftwf_alloc_complex(f1c->local_size);
    f0c->transpose(c0, c1);
    // we don't need c0 anymore
    fftwf_free(c0);

    // transpose last two dimensions here
    f1c->transpose(c1);

    // clear f1c, and put in the description for the fully
    // transposed field
    delete f1c;
    ni[0] = atoi(argv[3]);
    ni[1] = atoi(argv[2]);
    ni[2] = atoi(argv[1]);
    f1c = new field_descriptor(3, ni, MPI_COMPLEX8, MPI_COMM_WORLD);

    // following 3 arguments are dimensions for output array
    // i.e. real space grid dimensions
    // f2r and f2c will be allocated in this call
    fftwf_get_descriptors_3D(
            atoi(argv[4]), atoi(argv[5]), atoi(argv[6]),
            &f2r, &f2c);

    c2 = fftwf_alloc_complex(f2c->local_size);

    // pad input array with zeros
    // or call it Fourier interpolation if you like
    fftwf_copy_complex_array(
            f1c, c1,
            f2c, c2);

    // we don't need c1 anymore
    fftwf_free(c1);

    float *r2;
    r2 = fftwf_alloc_real(2*f2c->local_size);

    fftwf_plan c2r = fftwf_mpi_plan_dft_c2r_3d(
            f2r->sizes[0], f2r->sizes[1], f2r->sizes[2],
            c2, r2,
            MPI_COMM_WORLD,
            FFTW_ESTIMATE);
    fftwf_execute(c2r);
    fftwf_destroy_plan(c2r);

    // we don't need c2 anymore
    fftw_free(c2);

    fftwf_clip_zero_padding(f2r, r2);
    f2r->write("data2r", (void*)r2);

    // clean up
    fftw_free(r2);
    fftwf_mpi_cleanup();
    delete f0c;
    delete f1c;
    delete f2r;
    delete f2c;
    MPI_Finalize();
    return EXIT_SUCCESS;
}

