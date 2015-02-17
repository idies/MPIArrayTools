#include "RMHD_converter.hpp"

extern int myrank, nprocs;

inline ptrdiff_t part1by2(ptrdiff_t x)
{
    ptrdiff_t n = x & 0x000003ff;
    n = (n ^ (n << 16)) & 0xff0000ff;
    n = (n ^ (n <<  8)) & 0x0300f00f;
    n = (n ^ (n <<  4)) & 0x030c30c3;
    n = (n ^ (n <<  2)) & 0x09249249;
    return n;
}

inline ptrdiff_t unpart1by2(ptrdiff_t z)
{
        ptrdiff_t n = z & 0x09249249;
        n = (n ^ (n >>  2)) & 0x030c30c3;
        n = (n ^ (n >>  4)) & 0x0300f00f;
        n = (n ^ (n >>  8)) & 0xff0000ff;
        n = (n ^ (n >> 16)) & 0x000003ff;
        return n;
}

inline ptrdiff_t regular_to_zindex(
        ptrdiff_t x0, ptrdiff_t x1, ptrdiff_t x2)
{
    return part1by2(x0) | (part1by2(x1) << 1) | (part1by2(x2) << 2);
}

inline void zindex_to_grid3D(
        ptrdiff_t z,
        ptrdiff_t &x0, ptrdiff_t &x1, ptrdiff_t &x2)
{
    x0 = unpart1by2(z     );
    x1 = unpart1by2(z >> 1);
    x2 = unpart1by2(z >> 2);
}

RMHD_converter::RMHD_converter(
        int n0, int n1, int n2,
        int N0, int N1, int N2)
{
    int n[7];

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

    // various descriptions for the real data
    n[0] = N0*2;
    n[1] = N1;
    n[2] = N2;
    this->f4r = new field_descriptor(3, n, MPI_REAL4, MPI_COMM_WORLD);
    n[0] = N0/8;
    n[1] = N1/8;
    n[2] = N2/8;
    n[3] = 8*8*8*2;
    this->drcubbie = new field_descriptor(4, n, MPI_REAL4, MPI_COMM_WORLD);
    n[0] = (N0/8) * (N1/8) * (N2/8);
    n[1] = 8*8*8*2;
    this->dzcubbie = new field_descriptor(2, n, MPI_REAL4, MPI_COMM_WORLD);

}

RMHD_converter::~RMHD_converter()
{
    if (this->f0c != NULL) delete this->f0c;
    if (this->f1c != NULL) delete this->f1c;
    if (this->f2c != NULL) delete this->f2c;
    if (this->f3c != NULL) delete this->f3c;
    if (this->f3r != NULL) delete this->f3r;
    if (this->f4r != NULL) delete this->f4r;
    if (this->drcubbie != NULL) delete this->drcubbie;
    if (this->dzcubbie != NULL) delete this->dzcubbie;

    if (this->c0  != NULL) fftwf_free(this->c0);
    if (this->c12 != NULL) fftwf_free(this->c12);
    if (this->c3  != NULL) fftwf_free(this->c3);
    if (this->r3  != NULL) fftwf_free(this->r3);

    fftwf_destroy_plan(this->complex2real0);
    fftwf_destroy_plan(this->complex2real1);
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

    // new array where mixed components will be placed
    float *rtmp = fftwf_alloc_real( 4*this->f3c->local_size);
    float *tpointer;

    // mix components
    for (int k = 0; k < this->f3r->local_size; k++)
        for (int j = 0; j < 2; j++)
                rtmp[k*2 + j] = this->r3[j*this->f3r->local_size + k];

    // point to mixed data
    tpointer = this->r3;
    this->r3 = rtmp;
    rtmp = tpointer;

    // shuffle into z order
    ptrdiff_t z, zz;
    int rid, zid;
    int kk;
    ptrdiff_t cubbie_size = 8*8*8*2;
    ptrdiff_t cc;
    float *rz = fftwf_alloc_real(cubbie_size);
    for (int k = 0; k < this->drcubbie->sizes[0]; k++)
    {
        rid = this->drcubbie->rank(k);
        kk = k - this->drcubbie->starts[0];
        for (int j = 0; j < this->drcubbie->sizes[1]; j++)
        for (int i = 0; i < this->drcubbie->sizes[2]; i++)
        {
            z = regular_to_zindex(k, j, i);
            zid = this->dzcubbie->rank(z);
            zz = z - this->dzcubbie->starts[0];
            if (myrank == rid || myrank == zid)
            {
                // first, copy data into cubbie
                if (myrank == rid)
                    for (int tk = 0; tk < 8; tk++)
                    for (int tj = 0; tj < 8; tj++)
                    {
                        cc = (((kk*8+tk)*this->f3r->sizes[1] + (j*8+tj)) *
                              this->f3r->sizes[2] + i*8)*2;
                        std::copy(
                                this->r3 + cc,
                                this->r3 + cc + 16,
                                rz + (tk*8 + tj)*16);
                    }
                // now copy or send/receive to zindexed array
                if (rid == zid) std::copy(
                        rz,
                        rz + cubbie_size,
                        rtmp + zz*cubbie_size);
                else
                {
                    if (myrank == rid) MPI_Send(
                            rz,
                            cubbie_size,
                            MPI_REAL4,
                            zid,
                            z,
                            MPI_COMM_WORLD);
                    else MPI_Recv(
                            rtmp + zz*cubbie_size,
                            cubbie_size,
                            MPI_REAL4,
                            rid,
                            z,
                            MPI_COMM_WORLD,
                            MPI_STATUS_IGNORE);
                }
            }
        }
    }
    fftwf_free(rz);

    //point to shuffled data
    tpointer = this->r3;
    this->r3 = rtmp;
    rtmp = tpointer;

    fftwf_free(rtmp);
    this->dzcubbie->write(ofile, (void*)this->r3);
    return EXIT_SUCCESS;
}

