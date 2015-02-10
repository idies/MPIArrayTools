transpose_2D:
	mpicc transpose_2D.c \
		-o t2D \
		-lfftw3_mpi \
		-lfftw3 \
		-lfftw3f_mpi \
		-lfftw3f
