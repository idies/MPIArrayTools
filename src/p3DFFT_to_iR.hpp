#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include "field_descriptor.hpp"
#include "fftwf_tools.hpp"

#ifndef P3DFFT_TO_IR

#define P3DFFT_TO_IR

extern int myrank, nprocs;


/* this class reads in separate Fourier space p3DFFT fields
 * and generates an interleaved realspace representation,
 * using FFTW and as little memory as possible.
 * */
class p3DFFT_to_iR
{
    public:
        /* members */
        int howmany;
        field_descriptor *f0c; // descriptor for 2D input
        field_descriptor *f1c; // descriptor for 2D transposed input
        field_descriptor *f2c; // descriptor for 3D fully transposed input
        field_descriptor *f3c, *f3r; // descriptors for FFT

        fftwf_complex *c0 ; // array to store 2D input
        fftwf_complex *c12; // array to store transposed input
        fftwf_complex *c3 ; // array to store resized Fourier data
        float *r3         ; // array to store real space data
        bool fields_allocated;

        fftwf_plan complex2real;

        /* methods */
        p3DFFT_to_iR(
                int n0, int n1, int n2,
                int N0, int N1, int N2,
                int howmany,
                bool allocate_fields = true);
        ~p3DFFT_to_iR();

        int read(
                char *ifile[]);
};

#endif//P3DFFT_TO_IR

