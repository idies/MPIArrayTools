/***********************************************************************
*
*  Copyright 2015 Johns Hopkins University
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*
* Contact: turbulence@pha.jhu.edu
* Website: http://turbulence.pha.jhu.edu/
*
************************************************************************/

#include <mpi.h>
#include <fftw3-mpi.h>

#ifndef FIELD_DESCRIPTOR

#define FIELD_DESCRIPTOR

extern int myrank, nprocs;

class field_descriptor
{
    public:

        /* data */
        int *sizes;
        int *subsizes;
        int *starts;
        int ndims;
        int *rank;
        ptrdiff_t slice_size, local_size, full_size;
        MPI_Datatype mpi_array_dtype, mpi_dtype;
        int myrank, nprocs, io_myrank, io_nprocs;
        MPI_Comm comm, io_comm;


        /* methods */
        field_descriptor(
                int ndims,
                int *n,
                MPI_Datatype element_type,
                MPI_Comm COMM_TO_USE);
        ~field_descriptor();

        /* io is performed using MPI_File stuff, and our
         * own mpi_array_dtype that was defined in the constructor.
         * */
        int read(
                const char *fname,
                void *buffer,
                const char *datarep = "native");
        int write(
                const char *fname,
                void *buffer,
                const char *datarep = "native");

        /* a function that generates the transposed descriptor.
         * don't forget to delete the result once you're done with it.
         * the transposed descriptor is useful for io operations.
         * */
        field_descriptor *get_transpose();

        /* we don't actually need the transposed descriptor to perform
         * the transpose operation: we only need the in/out fields.
         * */
        int transpose(
                float *input,
                float *output);
        int transpose(
                fftwf_complex *input,
                fftwf_complex *output = NULL);

        int interleave(
                float *input,
                int dim);
        int interleave(
                fftwf_complex *input,
                int dim);
};


/* given two arrays of the same dimension, we do simple resizes in
 * Fourier space: either chop off high modes, or pad with zeros.
 * the arrays are assumed to use fftw layout.
 * */
int fftwf_copy_complex_array(
        field_descriptor *fi,
        fftwf_complex *ai,
        field_descriptor *fo,
        fftwf_complex *ao);

int fftwf_clip_zero_padding(
        field_descriptor *f,
        float *a);

#endif//FIELD_DESCRIPTOR

