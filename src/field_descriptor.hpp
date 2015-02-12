#include <mpi.h>
#include <fftw3-mpi.h>

#ifndef __FIELD_DESCRIPTOR__

#define __FIELD_DESCRIPTOR__

class field_descriptor
{
    public:

        /* data */
        int *sizes;
        int *subsizes;
        int *starts;
        int ndims;
        int slice_size, local_size, full_size;
        MPI_Datatype mpi_array_dtype, mpi_dtype;


        /* methods */
        field_descriptor(
                int ndims,
                int *n,
                MPI_Datatype element_type);
        ~field_descriptor();

        /* io is performed using MPI_File stuff, and our
         * own mpi_array_dtype that was defined in the constructor.
         * */
        int read(
                const char *fname,
                void *buffer);
        int write(
                const char *fname,
                void *buffer);

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

        inline int rank(int i0)
        {
            return i0 / this->subsizes[0];
        }
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

#endif//__FIELD_DESCRIPTOR__

