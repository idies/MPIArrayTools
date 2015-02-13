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

field_descriptor* field_descriptor::get_transpose()
{
    int n[this->ndims];
    for (int i=0; i<this->ndims; i++)
        n[i] = this->sizes[this->ndims - i - 1];
    return new field_descriptor(this->ndims, n, this->mpi_dtype);
}


// should I use a template here?
int fftwf_copy_complex_array(
        field_descriptor *fi,
        fftwf_complex *ai,
        field_descriptor *fo,
        fftwf_complex *ao)
{
    if ((fi->ndims != 3) || (fo->ndims != 3))
        return EXIT_FAILURE;
    fftwf_complex *buffer;
    buffer = fftwf_alloc_complex(fi->slice_size);

    int min_fast_dim;
    min_fast_dim =
        (fi->sizes[fi->ndims - 1] > fo->sizes[fi->ndims - 1]) ?
         fo->sizes[fi->ndims - 1] : fi->sizes[fi->ndims - 1];

    // clean up destination, in case we're padding with zeros
    // (even if only for one dimension)
    std::fill_n((float*)ao, fo->local_size, 0.0);

    int64_t ii0, ii1;
    int64_t oi0, oi1;
    int64_t delta1, delta0;
    delta0 = (fo->sizes[0] - fi->sizes[0]);
    delta1 = (fo->sizes[1] - fi->sizes[1]);
    for (ii0=0; ii0 < fi->sizes[0]; ii0++)
    {
        if (ii0 <= fi->sizes[0]/2)
        {
            oi0 = ii0;
            if (oi0 > fo->sizes[0]/2)
                continue;
        }
        else
        {
            oi0 = ii0 + delta0;
            if ((oi0 < 0) || ((fo->sizes[0] - oi0) >= fo->sizes[0]/2))
                continue;
        }
        if ((fi->rank(ii0) == fo->rank(oi0)) &&
            (myrank == fi->rank(ii0)))
        {
            std::copy(
                    ai + (ii0 - fi->starts[0]    )*fi->slice_size,
                    ai + (ii0 - fi->starts[0] + 1)*fi->slice_size,
                    buffer);
        }
        else
        {
            if (myrank == fi->rank(ii0))
            {
                MPI_Send(
                        (void*)(ai + (ii0-fi->starts[0])*fi->slice_size),
                        fi->slice_size,
                        MPI_COMPLEX8,
                        fo->rank(oi0),
                        ii0,
                        MPI_COMM_WORLD);
            }
            if (myrank == fo->rank(oi0))
            {
                MPI_Recv(
                        (void*)(buffer),
                        fi->slice_size,
                        MPI_COMPLEX8,
                        fi->rank(ii0),
                        ii0,
                        MPI_COMM_WORLD,
                        MPI_STATUS_IGNORE);
            }
        }
        if (myrank == fo->rank(oi0))
        {
            for (ii1 = 0; ii1 < fi->sizes[1]; ii1++)
            {
                if (ii1 <= fi->sizes[1]/2)
                {
                    oi1 = ii1;
                    if (oi1 > fo->sizes[1]/2)
                        continue;
                }
                else
                {
                    oi1 = ii1 + delta1;
                    if ((oi1 < 0) || ((fo->sizes[1] - oi1) >= fo->sizes[1]/2))
                        continue;
                }
                std::copy(
                        (buffer + ii1*fi->sizes[2]),
                        (buffer + ii1*fi->sizes[2] + min_fast_dim),
                        (ao + ((oi0 - fo->starts[0])*fo->sizes[1] + oi1)*fo->sizes[2]));
            }
        }
    }
    fftw_free(buffer);
    MPI_Barrier(MPI_COMM_WORLD);

    return EXIT_SUCCESS;
}

int fftwf_clip_zero_padding(
        field_descriptor *f,
        float *a)
{
    if (f->ndims != 3)
        return EXIT_FAILURE;
    float *b = a;
    for (int i0 = 0; i0 < f->subsizes[0]; i0++)
        for (int i1 = 0; i1 < f->sizes[1]; i1++)
        {
            std::copy(a, a + f->sizes[2], b);
            a += f->sizes[2] + 2;
            b += f->sizes[2];
        }
    return EXIT_SUCCESS;
}
