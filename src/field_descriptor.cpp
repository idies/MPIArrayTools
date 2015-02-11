#include "field_descriptor.hpp"


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
            MPI_MODE_CREATE | MPI_MODE_WRONLY,
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

int field_descriptor::transpose(
        float *input,
        float *output)
{
    // IMPORTANT NOTE:
    // for 3D transposition, the input data is messed up
    fftwf_plan tplan;
    ptrdiff_t dim1;
    switch (this->ndims)
    {
        case 2:
            dim1 = this->sizes[1];
            break;
        case 3:
            // transpose the two local dimensions 1 and 2
            dim1 = this->sizes[1]*this->sizes[2];
            float *atmp;
            atmp = (float*)malloc(dim1*sizeof(float));
            for (int k = 0; k < this->subsizes[0]; k++)
            {
                // put transposed slice in atmp
                for (int j = 0; j < this->sizes[1]; j++)
                    for (int i = 0; i < this->sizes[2]; i++)
                        atmp[i*this->sizes[1] + j] =
                            input[(k*this->sizes[1] + j)*this->sizes[2] + i];
                // copy back transposed slice
                for (int j = 0; j < this->sizes[2]; j++)
                    for (int i = 0; i < this->sizes[1]; i++)
                        input[(k*this->sizes[2] + j)*this->sizes[1] + i] =
                            atmp[j*this->sizes[1] + i];
            }
            free(atmp);
            break;
        default:
            return -1;
            break;
    }
    tplan = fftwf_mpi_plan_transpose(
            this->sizes[0], dim1,
            input, output,
            MPI_COMM_WORLD,
            FFTW_ESTIMATE);
    fftwf_execute(tplan);
    fftwf_destroy_plan(tplan);
    return EXIT_SUCCESS;
}

