#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <fftw3-mpi.h>

int myrank, nprocs;

int main(int argc, char *argv[])
{
    /*************************/
    // init mpi environment
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    /*************************/

    fftwf_plan tplan;
    int n0 = 256;
    int n1 = 256;
    float *b0, *b1;

    int sizes0[] = {n0, n1};
    int subsizes0[] = {sizes0[1]/nprocs, sizes0[0]};
    int starts0[] = {myrank*sizes0[1]/nprocs, 0};
    float a0[subsizes0[1]][subsizes0[0]];

    int sizes1[] = {n1, n0};
    int subsizes1[] = {sizes1[1]/nprocs, sizes1[0]};
    int starts1[] = {myrank*sizes1[1]/nprocs, 0};
    float a1[subsizes1[1]][subsizes1[0]];

    MPI_Info info;
    MPI_Info_create(&info);
    MPI_File file0, file1;

    MPI_Status status;
    MPI_Datatype coord0, coord1;

    MPI_Type_create_subarray(
            2,
            sizes0,
            subsizes0,
            starts0,
            MPI_ORDER_C,
            MPI_FLOAT,
            &coord0);
    MPI_Type_commit(&coord0);
    MPI_Type_create_subarray(
            2,
            sizes1,
            subsizes1,
            starts1,
            MPI_ORDER_C,
            MPI_FLOAT,
            &coord1);
    MPI_Type_commit(&coord1);

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
            coord0,
            "native",
            info);
    MPI_File_read_all(
            file0,
            (void*)a0,
            subsizes0[0]*subsizes0[1],
            MPI_FLOAT,
            MPI_STATUS_IGNORE);
    MPI_File_close(&file0);

    b0 = (float*)a0;
    b1 = (float*)a1;
    tplan = fftwf_mpi_plan_transpose(
            n0, n1,
            b0, b1,
            MPI_COMM_WORLD,
            FFTW_ESTIMATE);
    fftwf_execute(tplan);

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
            coord1,
            "native",
            info);
    MPI_File_write_all(
            file1,
            (void*)a1,
            subsizes1[0]*subsizes1[1],
            MPI_FLOAT,
            MPI_STATUS_IGNORE);
    MPI_File_close(&file1);

    MPI_Type_free(&coord0);
    MPI_Type_free(&coord1);

    /*************************/
    //finalize mpi environment
    MPI_Finalize();
    /*************************/
    return EXIT_SUCCESS;
}

