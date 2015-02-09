#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

int myrank, nprocs;

int main(int argc, char *argv[])
{
    /*************************/
    // init mpi environment
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    /*************************/

    printf("%d %d aloha\n", myrank, nprocs);

    MPI_Finalize();
    /*************************/
    return EXIT_SUCCESS;
}

