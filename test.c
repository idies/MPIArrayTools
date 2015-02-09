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
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    /*************************/

    int cols = 128;
    int rows = 256;
    int sizes[] = {cols, rows};
    int subsizes[] = {2, 4};
    int starts[] = {1, 1};
    float send[rows][cols];
    float recv[cols][rows];

    MPI_Info info;
    MPI_Info_create(&info);
    MPI_File file0, file1;


    MPI_Status status;
    MPI_Datatype col0, col1;
    MPI_Request req;

    MPI_Type_vector(rows, 1, cols, MPI_SINGLE_PRECISION, &col0);
    MPI_Type_commit(&col0);
    MPI_Type_hvector(cols, 1, sizeof(int), col, &col1);
    MPI_Type_commit(&col1);
    printf("%d aloha\n", myrank);
    MPI_File_open(MPI_COMM_WORLD, "data0", MPI_MODE_RDONLY, info, &file0);
    MPI_File_set_view(file0, 0_MPI_OFFSET_KIND, MPI_SINGLE_PRECISION, col0, "native", info);
    MPI_File_read_all(file0, (void*)send, cols*rows, MPI_SINGLE_PRECISION, MPI_STATUS_IGNORE);

    MPI_Isend((void*)send, rows*cols, MPI_SINGLE_PRECISION, 0, 1, MPI_COMM_WORLD,&req);
    MPI_Recv( (void*)recv, 1, col1, 0, 1, MPI_COMM_WORLD, &status);
    printf("%d aloha\n", myrank);

    MPI_Type_free(&col1);
    MPI_Type_free(&col0);

    /*************************/
    //finalize mpi environment
    MPI_Finalize();
    /*************************/
    return EXIT_SUCCESS;
}

