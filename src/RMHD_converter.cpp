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

#include "p3DFFT_to_iR.hpp"
#include "Morton_shuffler.hpp"
#include <iostream>

int myrank, nprocs;

const int iter0 = 138000;

int get_RMHD_names(
        int iteration,
        bool velocity,
        char **fname)
{
    if (velocity)
    {
        sprintf(fname[0], "K%.6dQNP002", iteration);
        sprintf(fname[1], "K%.6dQNP003", iteration);
    }
    else
    {
        sprintf(fname[0], "K%.6dQNP005", iteration);
        sprintf(fname[1], "K%.6dQNP006", iteration);
    }
    return EXIT_SUCCESS;
}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    const int n = 1364;
    const int N = 2048;
    int iteration;
    if (argc < 2)
    {
        std::cerr <<
            "not enough parameters.\naborting." <<
            std::endl;
        MPI_Finalize();
        return EXIT_SUCCESS;
    }
    p3DFFT_to_iR *r = new p3DFFT_to_iR(
            (n/2+1), n, n,
            N, N, N,
            2);
    Morton_shuffler *s = new Morton_shuffler(
            N, N, N, 2, 64);

    // initialize file names
    char *ifile[2];
    for (int i=0; i<2; i++)
        ifile[i] = (char*)malloc(sizeof(char)*100);

    for (int i=1; i<argc; i++)
    {
        iteration = atoi(argv[i]);
        // velocity
        get_RMHD_names(iteration, true, ifile);
        r->read(ifile);
        sprintf(ifile[0], "RMHD_u_t%.3x", iteration - iter0);
        s->shuffle(r->r3, ifile[0]);
        // magnetic
        get_RMHD_names(iteration, false, ifile);
        r->read(ifile);
        sprintf(ifile[0], "RMHD_b_t%.3x", iteration - iter0);
        s->shuffle(r->r3, ifile[0]);
    }

    //free file names
    for (int i=0; i<2; i++)
        free(ifile[i]);

    delete s;
    delete r;

    // clean up
    fftwf_mpi_cleanup();
    MPI_Finalize();
    return EXIT_SUCCESS;
}

