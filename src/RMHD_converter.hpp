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
        field_descriptor *f4r = NULL; //descriptor for output

        fftwf_complex *c0  = NULL; // array to store 2D input
        fftwf_complex *c12 = NULL; // array to store transposed input
        fftwf_complex *c3  = NULL; // array to store resized Fourier data
        float *r3          = NULL; // array to store real space data

        fftwf_plan complex2real0;
        fftwf_plan complex2real1;

        /* methods */
        RMHD_converter(
                int n0, int n1, int n2,
                int N0, int N1, int N2);
        ~RMHD_converter()
        {
            if (this->f0c != NULL) delete this->f0c;
            if (this->f1c != NULL) delete this->f1c;
            if (this->f2c != NULL) delete this->f2c;
            if (this->f3c != NULL) delete this->f3c;
            if (this->f3r != NULL) delete this->f3r;
            if (this->f4r != NULL) delete this->f4r;

            if (this->c0  != NULL) fftwf_free(this->c0);
            if (this->c12 != NULL) fftwf_free(this->c12);
            if (this->c3  != NULL) fftwf_free(this->c3);
            if (this->r3  != NULL) fftwf_free(this->r3);

            fftwf_destroy_plan(this->complex2real0);
            fftwf_destroy_plan(this->complex2real1);
        }

        int convert(
                const char *ifile0,
                const char *ifile1,
                const char *ofile);
};

#endif//RMHD_CONVERTER

