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
#include "field_descriptor.hpp"

#ifndef FFTWF_TOOLS

#define FFTWF_TOOLS

extern int myrank, nprocs;

/* given two arrays of the same dimension, we do a simple resize in
 * Fourier space: either chop off high modes, or pad with zeros.
 * the arrays are assumed to use 3D mpi fftw layout.
 * */
int fftwf_copy_complex_array(
        field_descriptor *fi,
        fftwf_complex *ai,
        field_descriptor *fo,
        fftwf_complex *ao);

int fftwf_clip_zero_padding(
        field_descriptor *f,
        float *a,
        int howmany=1);

/* function to get pair of descriptors for real and Fourier space
 * arrays used with fftw.
 * the n0, n1, n2 correspond to the real space data WITHOUT the zero
 * padding that FFTW needs.
 * IMPORTANT: the real space array must be allocated with
 * 2*fc->local_size, and then the zeros cleaned up before trying
 * to write data.
 * */
int fftwf_get_descriptors_3D(
        int n0, int n1, int n2,
        field_descriptor **fr,
        field_descriptor **fc);

#endif//FFTWF_TOOLS

