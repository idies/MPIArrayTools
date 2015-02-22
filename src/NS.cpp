#include "Morton_shuffler.hpp"
#include <iostream>

int myrank, nprocs;

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);


    Morton_shuffler *s = new Morton_shuffler(
            1024, 1024, 1024,
            3, 8);
    int n[] = {1024, 1024, 1024};
    field_descriptor *scalar = new field_descriptor(
            3, n, MPI_REAL4, MPI_COMM_WORLD);

    float *data0 = fftwf_alloc_real(s->dinput->local_size);
    float *data1 = fftwf_alloc_real(s->dinput->local_size);
    scalar->read("NS_ux_t100", data0);
    scalar->read("NS_uy_t100", data0 +   scalar->local_size);
    scalar->read("NS_uz_t100", data0 + 2*scalar->local_size);
    scalar->interleave(data0, data1, 3);
    s->shuffle(data1, data0, "NS_u_t100");

    fftwf_free(data1);
    delete s;
    s = new Morton_shuffler(
            1024, 1024, 1024,
            1, 8);
    data1 = data0 + scalar->local_size;
    scalar->read("NS_p_t100", data0);
    s->shuffle(data0, data1, "NS_p_t100");

    fftwf_free(data0);
    delete s;

    // clean up
    fftwf_mpi_cleanup();
    MPI_Finalize();
    return EXIT_SUCCESS;
}

