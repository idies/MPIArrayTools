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

#include "base.hpp"
#include "field_descriptor.hpp"
#include <ctime>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>

int myrank, nprocs;

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    int n[2];
    DEBUG_MSG_WAIT(
            MPI_COMM_WORLD,
            "nprocs = %d\n", nprocs);
    n[0] = nprocs;
    n[1] = 8;
    field_descriptor *f = new field_descriptor(
            2, n,
            MPI_FLOAT,
            MPI_COMM_WORLD);

    DEBUG_MSG_WAIT(
            MPI_COMM_WORLD,
            "local_size = %d\n", f->local_size);
    float *data = fftwf_alloc_real(f->local_size);
    srand48(myrank + time(NULL));

    for (int i=0; i<n[1]; i++)
    {
        //fread((char*)(data+i), 1, sizeof(float), randomData);
        data[i] = drand48();
    }
    DEBUG_MSG_WAIT(
            MPI_COMM_WORLD,
            "%g\n", data[0]);

    f->write("data_native", data, "native");
    f->write("data_internal", data, "internal");
    f->write("data_external32", data, "external32");

    fftwf_free(data);
    delete f;

    // clean up
    fftwf_mpi_cleanup();
    MPI_Finalize();
    return EXIT_SUCCESS;
}

