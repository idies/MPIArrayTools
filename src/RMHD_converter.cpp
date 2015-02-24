#include "RMHD_converter.hpp"
#include <string>
#include <iostream>



RMHD_converter::RMHD_converter(
        int n0, int n1, int n2,
        int N0, int N1, int N2,
        int nfiles)
{
    int n[7];
    proc_print_err_message("entering constructor of RMHD_converter");

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
    this->r3  = fftwf_alloc_real( 4*this->f3c->local_size);
    // all FFTs are going to be inplace
    this->c3  = (fftwf_complex*)this->r3;

    // allocate plans
    ptrdiff_t blabla[] = {this->f3r->sizes[0],
                          this->f3r->sizes[1],
                          this->f3r->sizes[2]};
    this->complex2real = fftwf_mpi_plan_many_dft_c2r(
            3,
            blabla,
            2,
            FFTW_MPI_DEFAULT_BLOCK,
            FFTW_MPI_DEFAULT_BLOCK,
            this->c3, this->r3,
            MPI_COMM_WORLD,
            FFTW_ESTIMATE);

    this->s = new Morton_shuffler(N0, N1, N2, 2, nfiles);
    proc_print_err_message("exiting constructor of RMHD_converter");
}

RMHD_converter::~RMHD_converter()
{
    delete this->f0c;
    delete this->f1c;
    delete this->f2c;
    delete this->f3c;
    delete this->f3r;

    delete this->s;

    fftwf_free(this->c0);
    fftwf_free(this->c12);
    fftwf_free(this->r3);

    fftwf_destroy_plan(this->complex2real);
}

int RMHD_converter::convert(
        const char *ifile0,
        const char *ifile1,
        const char *ofile)
{
    //read first field
    proc_print_err_message("RMHD_converter::convert "
                           "this->f0c->read(ifile0, (void*)this->c0);");
    this->f0c->read(ifile0, (void*)this->c0);
    proc_print_err_message("RMHD_converter::convert "
                           "this->f0c->transpose(this->c0, this->c12);");
    this->f0c->transpose(this->c0, this->c12);
    proc_print_err_message("RMHD_converter::convert "
                           "this->f1c->transpose(this->c12);");
    this->f1c->transpose(this->c12);
    proc_print_err_message("RMHD_converter::convert "
                           "fftwf_copy_complex_array(");
    fftwf_copy_complex_array(
            this->f2c, this->c12,
            this->f3c, this->c3);

    //read second field
    this->f0c->read(ifile1, (void*)this->c0);
    this->f0c->transpose(this->c0, this->c12);
    this->f1c->transpose(this->c12);
    fftwf_copy_complex_array(
            this->f2c, this->c12,
            this->f3c, this->c3 + this->f3c->local_size);

    proc_print_err_message("RMHD_converter::convert "
                           "this->f3c->interleave(this->c3, 2);");
    this->f3c->interleave(this->c3, 2);

    proc_print_err_message("RMHD_converter::convert "
                           "fftwf_execute(this->complex2real);");
    fftwf_execute(this->complex2real);

    proc_print_err_message("RMHD_converter::convert "
                           "fftwf_clip_zero_padding(this->f3r, this->r3, 2);");
    fftwf_clip_zero_padding(this->f3r, this->r3, 2);
    proc_print_err_message("RMHD_converter::convert "
                           "this->s->shuffle(this->r3, ofile);");
    this->s->shuffle(this->r3, ofile);
    proc_print_err_message("RMHD_converter::convert return");
    return EXIT_SUCCESS;
}

