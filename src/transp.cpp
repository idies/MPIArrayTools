#include "field_descriptor.hpp"

int myrank, nprocs;

int transpose_2D(
        int n[],
        const char *fname0,
        const char *fname1)
{
    // generate field descriptor objects
    int ntmp[2];
    ntmp[0] = n[0];
    ntmp[1] = n[1];
    field_descriptor f0, f1;
    f0.initialize(2, ntmp, MPI_FLOAT);
    ntmp[0] = n[1];
    ntmp[1] = n[0];
    f1.initialize(2, ntmp, MPI_FLOAT);

    // allocate arrays
    float *a0, *a1;
    a0 = (float*)malloc(f0.local_size*sizeof(float));
    a1 = (float*)malloc(f1.local_size*sizeof(float));

    // read data, do transpose, write data
    f0.read(fname0, (void*)a0);
    fftwf_plan tplan;
    tplan = fftwf_mpi_plan_transpose(
            n[0], n[1],
            a0, a1,
            MPI_COMM_WORLD,
            FFTW_ESTIMATE);
    fftwf_execute(tplan);
    f1.write(fname1, (void*)a1);

    // free arrays
    free(a0);
    free(a1);
    // free field descriptors
    f0.finalize();
    f1.finalize();
    return EXIT_SUCCESS;
}

int transpose_3D(
        int n[],
        const char *fname0,
        const char *fname1)
{
    // generate field descriptor objects
    int ntmp[3];
    ntmp[0] = n[0];
    ntmp[1] = n[1];
    ntmp[2] = n[2];
    field_descriptor f0, f1;
    f0.initialize(3, ntmp, MPI_FLOAT);
    ntmp[0] = n[2];
    ntmp[1] = n[1];
    ntmp[2] = n[0];
    f1.initialize(3, ntmp, MPI_FLOAT);

    // allocate arrays
    float *a0, *a1, *atmp;
    a0 = (float*)malloc(f0.local_size*sizeof(float));
    a1 = (float*)malloc(f1.local_size*sizeof(float));
    atmp = (float*)malloc(f0.sizes[1]*f0.sizes[2]*sizeof(float));

    // read data
    f0.read(fname0, (void*)a0);

    // transpose the two local dimensions 1 and 2:
    for (int k = 0; k < f0.subsizes[0]; k++)
    {
        // put transposed slice in atmp
        for (int j = 0; j<n[1]; j++)
            for (int i = 0; i<n[2]; i++)
            {
                atmp[i*n[1] + j] = a0[(k*n[1] + j)*n[2] + i];
            }
        // copy back transposed slice
        for (int j = 0; j<n[2]; j++)
            for (int i = 0; i<n[1]; i++)
            {
                a0[(k*n[2] + j)*n[1] + i] = atmp[j*n[1] + i];
            }
    }
    fftwf_plan tplan;
    tplan = fftwf_mpi_plan_transpose(
            n[0], n[1]*n[2],
            a0, a1,
            MPI_COMM_WORLD,
            FFTW_ESTIMATE);
    fftwf_execute(tplan);
    f1.write(fname1, (void*)a1);

    // free arrays
    free(a0);
    free(a1);
    free(atmp);
    // free field descriptors
    f0.finalize();
    f1.finalize();
    return EXIT_SUCCESS;
}

int main(int argc, char *argv[])
{
    /*************************/
    // init mpi environment
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    /*************************/

    int n[3];
    switch(argc)
    {
        case 3:
            if (myrank == 0)
                printf("transposing 2D array from \"data0\" into \"data1\" with %d processes.\n", nprocs);
            // dimensions
            n[0] = atoi(argv[1]);
            n[1] = atoi(argv[2]);
            transpose_2D(
                    n,
                    "data0",
                    "data1");
            break;
        case 4:
            if (myrank == 0)
                printf("transposing 3D array from \"data0\" into \"data1\" with %d processes.\n", nprocs);
            // dimensions
            n[0] = atoi(argv[1]);
            n[1] = atoi(argv[2]);
            n[2] = atoi(argv[3]);
            transpose_3D(
                    n,
                    "data0",
                    "data1");
            break;
        default:
            printf("you messed up the parameters, I'm not doing anything.\n");
    }

    /*************************/
    //finalize mpi environment
    MPI_Finalize();
    /*************************/
    return EXIT_SUCCESS;
}

