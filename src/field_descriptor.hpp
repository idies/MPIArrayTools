#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <fftw3-mpi.h>

#ifndef __FIELD_DESCRIPTOR__

#define __FIELD_DESCRIPTOR__

extern int myrank, nprocs;

class field_descriptor
{
    public:
        int *sizes;
        int *subsizes;
        int *starts;
        int ndims;
        int local_size, full_size;
        MPI_Datatype mpi_array_dtype, mpi_dtype;

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
        int transpose(
                float *input,
                float *output);
};

#endif//__FIELD_DESCRIPTOR__

