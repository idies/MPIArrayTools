#include <stdlib.h>
#include <algorithm>
#include <iostream>
#include "field_descriptor.hpp"

extern int myrank, nprocs;

field_descriptor::field_descriptor(
        int ndims,
        int *n,
        MPI_Datatype element_type)
{
    this->ndims = ndims;
    this->sizes    = new int[ndims];
    this->subsizes = new int[ndims];
    this->starts   = new int[ndims];
    this->sizes[0] = n[0];
    this->subsizes[0] = n[0]/nprocs;
    this->starts[0] = myrank*(n[0]/nprocs);
    this->mpi_dtype = element_type;
    this->slice_size = 1;
    this->local_size = this->subsizes[0];
    this->full_size = this->sizes[0];
    for (int i = 1; i < this->ndims; i++)
    {
        this->sizes[i] = n[i];
        this->subsizes[i] = n[i];
        this->starts[i] = 0;
        this->slice_size *= this->subsizes[i];
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
}

field_descriptor::~field_descriptor()
{
    delete[] this->sizes;
    delete[] this->subsizes;
    delete[] this->starts;
    MPI_Type_free(&this->mpi_array_dtype);
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
            return EXIT_FAILURE;
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

int field_descriptor::transpose(
        fftwf_complex *input,
        fftwf_complex *output)
{
    switch (this->ndims)
    {
        case 2:
            // do a global transpose over the 2 dimensions
            if (this->sizes[0] % nprocs != 0 || this->sizes[1] % nprocs != 0)
            {
                std::cerr << "you're trying to work with an array that cannot "
                             "be evenly distributed among processes.\n"
                          << std::endl;
                return EXIT_FAILURE;
            }
            if (output == NULL)
            {
                std::cerr << "bad arguments for transpose.\n" << std::endl;
                return EXIT_FAILURE;
            }
            fftwf_plan tplan;
            tplan = fftwf_mpi_plan_many_transpose(
                    this->sizes[0], this->sizes[1], 2,
                    FFTW_MPI_DEFAULT_BLOCK,
                    FFTW_MPI_DEFAULT_BLOCK,
                    (float*)input, (float*)output,
                    MPI_COMM_WORLD,
                    FFTW_ESTIMATE);
            fftwf_execute(tplan);
            fftwf_destroy_plan(tplan);
            break;
        case 3:
            // transpose the two local dimensions 1 and 2
            fftwf_complex *atmp;
            atmp = fftwf_alloc_complex(this->slice_size);
            for (int k = 0; k < this->subsizes[0]; k++)
            {
                // put transposed slice in atmp
                for (int j = 0; j < this->sizes[1]; j++)
                    for (int i = 0; i < this->sizes[2]; i++)
                    {
                        atmp[i*this->sizes[1] + j][0] =
                            input[(k*this->sizes[1] + j)*this->sizes[2] + i][0];
                        atmp[i*this->sizes[1] + j][1] =
                            input[(k*this->sizes[1] + j)*this->sizes[2] + i][1];
                    }
                // copy back transposed slice
                for (int j = 0; j < this->sizes[2]; j++)
                    for (int i = 0; i < this->sizes[1]; i++)
                    {
                        input[(k*this->sizes[2] + j)*this->sizes[1] + i][0] =
                            atmp[j*this->sizes[1] + i][0];
                        input[(k*this->sizes[2] + j)*this->sizes[1] + i][1] =
                            atmp[j*this->sizes[1] + i][1];
                    }
            }
            fftwf_free(atmp);
            break;
        default:
            return EXIT_FAILURE;
            break;
    }
    return EXIT_SUCCESS;
}

field_descriptor* field_descriptor::get_transpose()
{
    int n[this->ndims];
    for (int i=0; i<this->ndims; i++)
        n[i] = this->sizes[this->ndims - i - 1];
    return new field_descriptor(this->ndims, n, this->mpi_dtype);
}

