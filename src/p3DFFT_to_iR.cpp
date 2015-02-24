#include "p3DFFT_to_iR.hpp"
#include <string>
#include <iostream>



p3DFFT_to_iR::p3DFFT_to_iR(
        int n0, int n1, int n2,
        int N0, int N1, int N2,
        int howmany)
{
    this->howmany = howmany;
    int n[7];
    proc_print_err_message("entering constructor of p3DFFT_to_iR");

    // first 3 arguments are dimensions for input array
    // i.e. actual dimensions for the Fourier representation.
    // NOT real space grid dimensions
    // the input array is read in as a 2D array,
    // since the first dimension must be a multiple of nprocs
    // (which is generally an even number)
    n[0] = n0*n1;
    n[1] = n2;
    this->f0c = new field_descriptor(2, n, MPI_COMPLEX8, MPI_COMM_WORLD);

    // f1c will be pointing at the input array after it has been
    // transposed in 2D, therefore we have this correspondence:
    // f0c->sizes[0] = f1c->sizes[1]*f1c->sizes[2]
    n[0] = n2;
    n[1] = n0;
    n[2] = n1;
    this->f1c = new field_descriptor(3, n, MPI_COMPLEX8, MPI_COMM_WORLD);

    // the description for the fully transposed field
    n[0] = n2;
    n[1] = n1;
    n[2] = n0;
    this->f2c = new field_descriptor(3, n, MPI_COMPLEX8, MPI_COMM_WORLD);

    // following 3 arguments are dimensions for real space grid dimensions
    // f3r and f3c will be allocated in this call
    fftwf_get_descriptors_3D(
            N0, N1, N2,
            &this->f3r, &this->f3c);

    //allocate fields
    this->c0  = fftwf_alloc_complex(this->f0c->local_size);
    this->c12 = fftwf_alloc_complex(this->f1c->local_size);
    // 4 instead of 2, because we have 2 fields to write
    this->r3  = fftwf_alloc_real( 2*this->f3c->local_size*this->howmany);
    // all FFTs are going to be inplace
    this->c3  = (fftwf_complex*)this->r3;

    // allocate plans
    ptrdiff_t blabla[] = {this->f3r->sizes[0],
                          this->f3r->sizes[1],
                          this->f3r->sizes[2]};
    this->complex2real = fftwf_mpi_plan_many_dft_c2r(
            3,
            blabla,
            this->howmany,
            FFTW_MPI_DEFAULT_BLOCK,
            FFTW_MPI_DEFAULT_BLOCK,
            this->c3, this->r3,
            MPI_COMM_WORLD,
            FFTW_ESTIMATE);

    proc_print_err_message("exiting constructor of p3DFFT_to_iR");
}

p3DFFT_to_iR::~p3DFFT_to_iR()
{
    delete this->f0c;
    delete this->f1c;
    delete this->f2c;
    delete this->f3c;
    delete this->f3r;

    fftwf_free(this->c0);
    fftwf_free(this->c12);
    fftwf_free(this->r3);

    fftwf_destroy_plan(this->complex2real);
}

int p3DFFT_to_iR::read(
        char *ifile[])
{
    //read fields
    for (int i = 0; i < this->howmany; i++)
    {
        proc_print_err_message("p3DFFT_to_iR::read "
                               "this->f0c->read(ifile0, (void*)this->c0);");
        this->f0c->read(ifile[i], (void*)this->c0);
        proc_print_err_message("p3DFFT_to_iR::read "
                               "this->f0c->transpose(this->c0, this->c12);");
        this->f0c->transpose(this->c0, this->c12);
        proc_print_err_message("p3DFFT_to_iR::read "
                               "this->f1c->transpose(this->c12);");
        this->f1c->transpose(this->c12);
        proc_print_err_message("p3DFFT_to_iR::read "
                               "fftwf_copy_complex_array(");
        fftwf_copy_complex_array(
                this->f2c, this->c12,
                this->f3c, this->c3 + i*this->f3c->local_size);
    }

    proc_print_err_message("p3DFFT_to_iR::read "
                           "this->f3c->interleave(this->c3, 2);");
    this->f3c->interleave(this->c3, this->howmany);

    proc_print_err_message("p3DFFT_to_iR::read "
                           "fftwf_execute(this->complex2real);");
    fftwf_execute(this->complex2real);

    proc_print_err_message("p3DFFT_to_iR::read "
                           "fftwf_clip_zero_padding(this->f3r, this->r3, 2);");
    fftwf_clip_zero_padding(this->f3r, this->r3, this->howmany);
    proc_print_err_message("p3DFFT_to_iR::read return");
    return EXIT_SUCCESS;
}

