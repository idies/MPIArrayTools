#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include "field_descriptor.hpp"
#include "fftwf_tools.hpp"

#ifndef RMHD_CONVERTER

#define RMHD_CONVERTER


class RMHD_converter
{
    public:
        /* members */
        field_descriptor *f0c = NULL; // descriptor for 2D input
        field_descriptor *f1c = NULL; // descriptor for 2D transposed input
        field_descriptor *f2c = NULL; // descriptor for 3D fully transposed input
        field_descriptor *f3c = NULL, *f3r=NULL; // descriptors for FFT

        // descriptor for N0*2 x N1 x N2 real space array
        field_descriptor *f4r = NULL;

        // descriptor for N0/8 x N1/8 x N2/8 x 8 x 8 x 8 x 2 array
        field_descriptor *drcubbie = NULL;
        // descriptor for NZ x 8 x 8 x 8 x 2 array
        field_descriptor *dzcubbie = NULL;

        // descriptor for (NZ/nfiles) x 8 x 8 x 8 x 2 array
        field_descriptor *dout = NULL;

        // communicator to use for output
        MPI_Comm out_communicator;
        int out_group;

        fftwf_complex *c0  = NULL; // array to store 2D input
        fftwf_complex *c12 = NULL; // array to store transposed input
        fftwf_complex *c3  = NULL; // array to store resized Fourier data
        float *r3          = NULL; // array to store real space data

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

