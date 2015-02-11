#include "field_descriptor.hpp"

int myrank, nprocs;

int transpose(
        field_descriptor *f,
        float *input,
        float *output)
{
    // IMPORTANT NOTE:
    // for 3D transposition, the input data is messed up
    fftwf_plan tplan;
    ptrdiff_t dim1;
    switch (f->ndims)
    {
        case 2:
            dim1 = f->sizes[1];
            break;
        case 3:
            // transpose the two local dimensions 1 and 2
            float *atmp;
            atmp = (float*)malloc(f->sizes[1]*f->sizes[2]*sizeof(float));
            for (int k = 0; k < f->subsizes[0]; k++)
            {
                // put transposed slice in atmp
                for (int j = 0; j < f->sizes[1]; j++)
                    for (int i = 0; i < f->sizes[2]; i++)
                    {
                        atmp[i*f->sizes[1] + j] =
                            input[(k*f->sizes[1] + j)*f->sizes[2] + i];
                    }
                // copy back transposed slice
                for (int j = 0; j < f->sizes[2]; j++)
                    for (int i = 0; i < f->sizes[1]; i++)
                    {
                        input[(k*f->sizes[2] + j)*f->sizes[1] + i] =
                            atmp[j*f->sizes[1] + i];
                    }
            }
            free(atmp);
            dim1 = f->sizes[1]*f->sizes[2];
            break;
        default:
            return -1;
            break;
    }
    tplan = fftwf_mpi_plan_transpose(
            f->sizes[0], dim1,
            input, output,
            MPI_COMM_WORLD,
            FFTW_ESTIMATE);
    fftwf_execute(tplan);
    fftwf_destroy_plan(tplan);
    return EXIT_SUCCESS;
}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    int n[3];
    field_descriptor f0, f1;

    switch(argc)
    {
        case 3:
            if (myrank == 0)
                printf("transposing 2D array from \"data0\" into \"data1\" with %d processes.\n", nprocs);
            // dimensions
            n[0] = atoi(argv[1]);
            n[1] = atoi(argv[2]);
            f0.initialize(2, n, MPI_FLOAT);
            n[0] = atoi(argv[2]);
            n[1] = atoi(argv[1]);
            f1.initialize(2, n, MPI_FLOAT);
            break;
        case 4:
            if (myrank == 0)
                printf("transposing 3D array from \"data0\" into \"data1\" with %d processes.\n", nprocs);
            // dimensions
            n[0] = atoi(argv[1]);
            n[1] = atoi(argv[2]);
            n[2] = atoi(argv[3]);
            f0.initialize(3, n, MPI_FLOAT);
            n[0] = atoi(argv[3]);
            n[1] = atoi(argv[2]);
            n[2] = atoi(argv[1]);
            f1.initialize(3, n, MPI_FLOAT);
            break;
        default:
            printf("you messed up the parameters, I'm not doing anything.\n");
            f0.finalize();
            f1.finalize();
            MPI_Finalize();
            break;
    }

    float *a0, *a1;
    a0 = (float*)malloc(f0.local_size*sizeof(float));
    a1 = (float*)malloc(f1.local_size*sizeof(float));
    f0.read("data0", (void*)a0);
    transpose(&f0, a0, a1);
    f1.write("data1", (void*)a1);
    free(a0);
    free(a1);

    MPI_Finalize();
    return EXIT_SUCCESS;
}

