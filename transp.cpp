#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <fftw3-mpi.h>

int myrank, nprocs;

/**********************************************************************/
class field_descriptor
{
    public:
        int *sizes;
        int *subsizes;
        int *starts;
        int ndims;
        int local_size, full_size;
        MPI_Datatype mpi_array_dtype, mpi_dtype;
        /****/
        field_descriptor(){}
        ~field_descriptor(){}
        int initialize(
                int ndims,
                int *n,
                MPI_Datatype element_type);
        int finalize();
        int read(
                const char *fname,
                void *buffer);
        int write(
                const char *fname,
                void *buffer);
};

int field_descriptor::initialize(
        int ndims,
        int *n,
        MPI_Datatype element_type)
{
    this->ndims = ndims;
    this->sizes    = (int*)malloc(ndims*sizeof(int));
    this->subsizes = (int*)malloc(ndims*sizeof(int));
    this->starts   = (int*)malloc(ndims*sizeof(int));
    this->sizes[0] = n[0];
    this->subsizes[0] = n[0]/nprocs;
    this->starts[0] = myrank*(n[0]/nprocs);
    this->mpi_dtype = element_type;
    this->local_size = this->subsizes[0];
    this->full_size = this->sizes[0];
    for (int i = 1; i < this->ndims; i++)
    {
        this->sizes[i] = n[i];
        this->subsizes[i] = n[i];
        this->starts[i] = 0;
        this->local_size *= this->subsizes[i];
        this->full_size *= this->sizes[i];
    }
    MPI_Type_create_subarray(
            ndims,
            this->sizes,
            this->subsizes,
            this->starts,
            MPI_ORDER_C,
            this->mpi_dtype,
            &this->mpi_array_dtype);
    MPI_Type_commit(&this->mpi_array_dtype);
    return EXIT_SUCCESS;
}

int field_descriptor::finalize()
{
    free((void*)this->sizes);
    free((void*)this->subsizes);
    free((void*)this->starts);
    MPI_Type_free(&this->mpi_array_dtype);
    return EXIT_SUCCESS;
}

int field_descriptor::read(
        const char *fname,
        void *buffer)
{
    MPI_Info info;
    MPI_Info_create(&info);
    MPI_File f;

    MPI_File_open(
            MPI_COMM_WORLD,
            fname,
            MPI_MODE_RDONLY,
            info,
            &f);
    MPI_File_set_view(
            f,
            0,
            this->mpi_dtype,
            this->mpi_array_dtype,
            "native", //this needs to be made more general
            info);
    MPI_File_read_all(
            f,
            buffer,
            this->local_size,
            this->mpi_dtype,
            MPI_STATUS_IGNORE);
    MPI_File_close(&f);

    return EXIT_SUCCESS;
}

int field_descriptor::write(
        const char *fname,
        void *buffer)
{
    MPI_Info info;
    MPI_Info_create(&info);
    MPI_File f;

    MPI_File_open(
            MPI_COMM_WORLD,
            fname,
            MPI_MODE_CREATE | MPI_MODE_EXCL | MPI_MODE_WRONLY,
            info,
            &f);
    MPI_File_set_view(
            f,
            0,
            this->mpi_dtype,
            this->mpi_array_dtype,
            "native", //this needs to be made more general
            info);
    MPI_File_write_all(
            f,
            buffer,
            this->local_size,
            this->mpi_dtype,
            MPI_STATUS_IGNORE);
    MPI_File_close(&f);

    return EXIT_SUCCESS;
}
/**********************************************************************/

int main(int argc, char *argv[])
{
    /*************************/
    // init mpi environment
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    /*************************/

    // dimensions
    int n0 = 48;
    int n1 = 32;

    // generate field descriptor objects
    int sizes0[] = {n0, n1};
    int sizes1[] = {n1, n0};
    field_descriptor f0, f1;
    f0.initialize(2, sizes0, MPI_FLOAT);
    f1.initialize(2, sizes1, MPI_FLOAT);

    // allocate arrays
    float *a0, *a1;
    a0 = (float*)malloc(f0.local_size*sizeof(float));
    a1 = (float*)malloc(f1.local_size*sizeof(float));

    // read data, do transpose, write data
    f0.read("data0", (void*)a0);
    fftwf_plan tplan;
    tplan = fftwf_mpi_plan_transpose(
            n0, n1,
            a0, a1,
            MPI_COMM_WORLD,
            FFTW_ESTIMATE);
    fftwf_execute(tplan);
    f1.write("data1", (void*)a1);

    // free arrays
    free(a0);
    free(a1);
    // free field descriptors
    f0.finalize();
    f1.finalize();
    /*************************/
    //finalize mpi environment
    MPI_Finalize();
    /*************************/
    return EXIT_SUCCESS;
}

