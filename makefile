MPICXX  = mpicxx
LINKER  = mpicxx
DEFINES =
CFLAGS  =

LIBS = -lfftw3_mpi \
	   -lfftw3 \
	   -lfftw3f_mpi \
	   -lfftw3f

vpath %.cpp ./src/

src := \
	transp.cpp \
	field_descriptor.cpp

obj := $(patsubst %.cpp, ./obj/%.cpp.o, ${src})

./obj/%.cpp.o: %.cpp
	${MPICXX} ${DEFINES} \
		${CFLAGS} \
		-c $^ -o $@

transpose_2D: ${obj}
	${LINKER} \
		${obj} \
		-o t2D \
		${LIBS} \
		${NULL}

