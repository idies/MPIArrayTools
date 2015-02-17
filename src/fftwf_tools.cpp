#include <stdlib.h>
#include <algorithm>
#include <iostream>
#include "field_descriptor.hpp"

extern int myrank, nprocs;

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

int fftwf_get_descriptors_3D(
        int n0, int n1, int n2,
        field_descriptor **fr,
        field_descriptor **fc)
{
    // I need to check whether there's already something there...
    if (*fr != NULL) delete *fr;
    if (*fc != NULL) delete *fc;
    int ntmp[3];
    ntmp[0] = n0;
    ntmp[1] = n1;
    ntmp[2] = n2;
    *fr = new field_descriptor(3, ntmp, MPI_REAL4, MPI_COMM_WORLD);
    ntmp[0] = n0;
    ntmp[1] = n1;
    ntmp[2] = n2/2+1;
    *fc = new field_descriptor(3, ntmp, MPI_COMPLEX8, MPI_COMM_WORLD);
    return EXIT_SUCCESS;
}

