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

    int sizes[] = {256, 256};
    int subsizes[] = {sizes[1]/nprocs, sizes[0]};
    int starts[] = {myrank*sizes[1]/nprocs, 0};
    float a[subsizes[1]][subsizes[0]];

    MPI_Info info;
    MPI_Info_create(&info);
    MPI_File file0, file1;


    MPI_Status status;
    MPI_Datatype col0;
    MPI_Request req;

    MPI_Type_create_subarray(
            2,
            sizes,
            subsizes,
            starts,
            MPI_ORDER_C,
            MPI_FLOAT,
            &col0);
    MPI_Type_commit(&col0);

    MPI_File_open(
            MPI_COMM_WORLD,
            "data0",
            MPI_MODE_RDONLY,
            info,
            &file0);
    MPI_File_set_view(
            file0,
            0,
            MPI_FLOAT,
            col0,
            "native",
            info);
    MPI_File_read_all(
            file0,
            (void*)a,
            subsizes[0]*subsizes[1],
            MPI_FLOAT,
            MPI_STATUS_IGNORE);
    MPI_File_close(&file0);

    MPI_File_open(
            MPI_COMM_WORLD,
            "data1",
            MPI_MODE_CREATE | MPI_MODE_WRONLY,
            info,
            &file1);
    MPI_File_set_view(
            file1,
            0,
            MPI_FLOAT,
            col0,
            "native",
            info);
    MPI_File_write_all(
            file1,
            (void*)a,
            subsizes[0]*subsizes[1],
            MPI_FLOAT,
            MPI_STATUS_IGNORE);
    MPI_File_close(&file1);
    printf("%d aloha\n", myrank);

    MPI_Type_free(&col0);

    /*************************/
    //finalize mpi environment
    MPI_Finalize();
    /*************************/
    return EXIT_SUCCESS;
}

