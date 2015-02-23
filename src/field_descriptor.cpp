#include <stdlib.h>
#include <algorithm>
#include <iostream>
#include "field_descriptor.hpp"

extern int myrank, nprocs;

field_descriptor::field_descriptor(
        int ndims,
        int *n,
        MPI_Datatype element_type,
        MPI_Comm COMM_TO_USE)
{
    this->comm = COMM_TO_USE;
    MPI_Comm_rank(this->comm, &this->myrank);
    MPI_Comm_size(this->comm, &this->nprocs);
    this->ndims = ndims;
    this->sizes    = new int[ndims];
    this->subsizes = new int[ndims];
    this->starts   = new int[ndims];
    ptrdiff_t *nfftw = new ptrdiff_t[ndims];
    ptrdiff_t local_n0, local_0_start;
    for (int i = 0; i < this->ndims; i++)
        nfftw[i] = n[i];
    this->local_size = fftwf_mpi_local_size_many(
            this->ndims,
            nfftw,
            1,
            FFTW_MPI_DEFAULT_BLOCK,
            this->comm,
            &local_n0,
            &local_0_start);
    this->sizes[0] = n[0];
    this->subsizes[0] = local_n0;
    this->starts[0] = local_0_start;
    this->mpi_dtype = element_type;
    this->slice_size = 1;
    this->full_size = this->sizes[0];
    for (int i = 1; i < this->ndims; i++)
    {
        this->sizes[i] = n[i];
        this->subsizes[i] = n[i];
        this->starts[i] = 0;
        this->slice_size *= this->subsizes[i];
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
    this->rank = new int[this->sizes[0]];
    int *local_rank = new int[this->sizes[0]];
    std::fill_n(local_rank, this->sizes[0], 0);
    for (int i = 0; i < this->sizes[0]; i++)
        if (i >= this->starts[0] && i < this->starts[0] + this->subsizes[0])
            local_rank[i] = this->myrank;
    MPI_Allreduce(
            local_rank,
            this->rank,
            this->sizes[0],
            MPI_INT,
            MPI_SUM,
            this->comm);
    delete[] local_rank;
}

field_descriptor::~field_descriptor()
{
    delete[] this->sizes;
    delete[] this->subsizes;
    delete[] this->starts;
    delete[] this->rank;
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
            this->comm,
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
            this->comm,
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
    if (this->ndims == 3)
    {
        // transpose the two local dimensions 1 and 2
        float *atmp;
        atmp = fftwf_alloc_real(this->slice_size);
        for (int k = 0; k < this->subsizes[0]; k++)
        {
            // put transposed slice in atmp
            for (int j = 0; j < this->sizes[1]; j++)
                for (int i = 0; i < this->sizes[2]; i++)
                    atmp[i*this->sizes[1] + j] =
                        input[(k*this->sizes[1] + j)*this->sizes[2] + i];
            // copy back transposed slice
            std::copy(
                    atmp,
                    atmp + this->slice_size,
                    input + k*this->slice_size);
        }
        fftwf_free(atmp);
    }
    tplan = fftwf_mpi_plan_transpose(
            this->sizes[0], this->slice_size,
            input, output,
            this->comm,
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
                    this->comm,
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
                std::copy(
                        (float*)(atmp),
                        (float*)(atmp + this->slice_size),
                        (float*)(input + k*this->slice_size));
            }
            fftwf_free(atmp);
            break;
        default:
            return EXIT_FAILURE;
            break;
    }
    return EXIT_SUCCESS;
}

int field_descriptor::interleave(
        float *input,
        float *output,
        int dim)
{
    // TODO: implement inplace interleaver
    for (int k = 0; k < this->local_size; k++)
        for (int j = 0; j < dim; j++)
            output[k*dim + j] = input[j*this->local_size + k];
    return EXIT_SUCCESS;
}

int field_descriptor::interleave(
        fftw_complex *input,
        fftw_complex *output,
        int dim)
{
    // TODO: implement inplace interleaver
    for (int k = 0; k < this->local_size; k++)
        for (int j = 0; j < dim; j++)
        {
            output[k*dim + j][0] = input[j*this->local_size + k][0];
            output[k*dim + j][1] = input[j*this->local_size + k][1];
        }
    return EXIT_SUCCESS;
}

field_descriptor* field_descriptor::get_transpose()
{
    int n[this->ndims];
    for (int i=0; i<this->ndims; i++)
        n[i] = this->sizes[this->ndims - i - 1];
    return new field_descriptor(this->ndims, n, this->mpi_dtype, this->comm);
}

void proc_print_err_message(const char *message)
{
    for (int i = 0; i < nprocs; i++)
    {
        if (myrank == i)
            std::cerr << i << " " << message << std::endl;
        MPI_Barrier(MPI_COMM_WORLD);
    }
}
