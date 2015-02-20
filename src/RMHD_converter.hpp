#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include "field_descriptor.hpp"
#include "fftwf_tools.hpp"
#include "Morton_shuffler.hpp"

#ifndef RMHD_CONVERTER

#define RMHD_CONVERTER

extern int myrank, nprocs;

class RMHD_converter
{
    public:
        /* members */
        field_descriptor *f0c; // descriptor for 2D input
        field_descriptor *f1c; // descriptor for 2D transposed input
        field_descriptor *f2c; // descriptor for 3D fully transposed input
        field_descriptor *f3c, *f3r; // descriptors for FFT

        // descriptor for N0*2 x N1 x N2 real space array
        field_descriptor *f4r;

        Morton_shuffler *s;

        fftwf_complex *c0 ; // array to store 2D input
        fftwf_complex *c12; // array to store transposed input
        fftwf_complex *c3 ; // array to store resized Fourier data
        float *r3         ; // array to store real space data

        fftwf_plan complex2real0;
        fftwf_plan complex2real1;

        /* methods */
        RMHD_converter(
                int n0, int n1, int n2,
                int N0, int N1, int N2,
                int nfiles);
        ~RMHD_converter();

        int convert(
                const char *ifile0,
                const char *ifile1,
                const char *ofile);
};

#endif//RMHD_CONVERTER

