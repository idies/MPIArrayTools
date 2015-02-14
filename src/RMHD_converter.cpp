#include "RMHD_converter.hpp"

RMHD_converter::RMHD_converter(
        int n0, int n1, int n2,
        int N0, int N1, int N2)
{
    int n[3];

    // first 3 arguments are dimensions for input array
    // i.e. actual dimensions for the Fourier representation.
    // NOT real space grid dimensions
    // the input array is read in as a 2D array,
    // since the first dimension must be a multiple of nprocs
    // (which is generally an even number)
    n[0] = n0*n1;
    n[1] = n2;
    this->f0c = new field_descriptor(2, n, MPI_COMPLEX8);

    // f1c will be pointing at the input array after it has been
    // transposed in 2D, therefore we have this correspondence:
    // f0c->sizes[0] = f1c->sizes[1]*f1c->sizes[2]
    n[0] = n2;
    n[1] = n0;
    n[2] = n1;
    this->f1c = new field_descriptor(3, n, MPI_COMPLEX8);

    // the description for the fully transposed field
    n[0] = n2;
    n[1] = n1;
    n[2] = n0;
    this->f2c = new field_descriptor(3, n, MPI_COMPLEX8);

    // following 3 arguments are dimensions for output array
    // i.e. real space grid dimensions
    // f3r and f3c will be allocated in this call
    fftwf_get_descriptors_3D(
            N0, N1, N2,
            &this->f3r, &this->f3c);

    //allocate fields
    this->c0  = fftwf_alloc_complex(this->f0c->local_size);
    this->c12 = fftwf_alloc_complex(this->f1c->local_size);
    this->c3  = fftwf_alloc_complex(this->f3c->local_size);
    // 4 instead of 2, because we have 2 fields to write
    this->r3  = fftwf_alloc_real( 4*this->f3c->local_size);

    // allocate plans
    this->complex2real0 = fftwf_mpi_plan_dft_c2r_3d(
            f3r->sizes[0], f3r->sizes[1], f3r->sizes[2],
            this->c3, this->r3,
            MPI_COMM_WORLD,
            FFTW_ESTIMATE);
    this->complex2real1 = fftwf_mpi_plan_dft_c2r_3d(
            f3r->sizes[0], f3r->sizes[1], f3r->sizes[2],
            this->c3, this->r3 + 2*this->f3c->local_size,
            MPI_COMM_WORLD,
            FFTW_PATIENT);

    // the description for the output
    n[0] = N0*2;
    n[1] = N1;
    n[2] = N2;
    this->f4r = new field_descriptor(3, n, MPI_REAL4);
}

int RMHD_converter::convert(
        const char *ifile0,
        const char *ifile1,
        const char *ofile)
{
    //read first field
    this->f0c->read(ifile0, (void*)this->c0);
    this->f0c->transpose(this->c0, this->c12);
    this->f1c->transpose(this->c12);
    fftwf_copy_complex_array(
            this->f2c, this->c12,
            this->f3c, this->c3);
    fftwf_execute(this->complex2real0);

    //read second field
    this->f0c->read(ifile1, (void*)this->c0);
    this->f0c->transpose(this->c0, this->c12);
    this->f1c->transpose(this->c12);
    fftwf_copy_complex_array(
            this->f2c, this->c12,
            this->f3c, this->c3);
    fftwf_execute(this->complex2real1);

    fftwf_clip_zero_padding(this->f4r, this->r3);

    //now mix the two components
    float *atmp = fftwf_alloc_real(this->f4r->local_size);

    // put transposed slice in atmp
    for (int k = 0; k < this->f3r->local_size; k++)
        for (int j = 0; j < 2; j++)
                atmp[k*2 + j] = this->r3[j*this->f3r->local_size + k];
    // copy back transposed slice
    //for (int k = 0; k < this->f3r->local_size; k++)
    //    for (int j = 0; j < 2; j++)
    //            atmp[j*this->f3r->local_size + k] = this->r3[k*2 + j];
    fftwf_free(this->r3);
    this->r3 = atmp;

    f4r->write(ofile, (void*)this->r3);
    return EXIT_SUCCESS;
}

