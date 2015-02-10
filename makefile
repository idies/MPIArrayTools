transpose_2D:
	mpic++ transp.cpp \
		-o t2D \
		-lfftw3_mpi \
		-lfftw3 \
		-lfftw3f_mpi \
		-lfftw3f
