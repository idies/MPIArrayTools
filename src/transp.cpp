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
            if (myrank == 0)
                printf("transposing 2D array from \"data0\" into \"data1\" with %d processes.\n", nprocs);
            n[0] = atoi(argv[1]);
            n[1] = atoi(argv[2]);
            break;
        case 4:
            if (myrank == 0)
                printf("transposing 3D array from \"data0\" into \"data1\" with %d processes.\n", nprocs);
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
    f0 = new field_descriptor(argc - 1, n, MPI_FLOAT);
    f1 = f0->get_transpose();

    float *a0, *a1;
    a0 = (float*)malloc(f0->local_size*sizeof(float));
    a1 = (float*)malloc(f1->local_size*sizeof(float));
    f0->read("data0", (void*)a0);
    f0->transpose(a0, a1);
    f1->write("data1", (void*)a1);
    free(a0);
    free(a1);

    delete f0;
    delete f1;
    MPI_Finalize();
    return EXIT_SUCCESS;
}

