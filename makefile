MPICXX  = mpicxx
LINKER  = mpicxx
DEFINES =
CFLAGS  = -Wall \
		  -O2

LIBS = -lfftw3_mpi \
	   -lfftw3 \
	   -lfftw3f_mpi \
	   -lfftw3f

vpath %.cpp ./src/

src := \
	field_descriptor.cpp

obj := $(patsubst %.cpp, ./obj/%.o, ${src})

./obj/%.o: %.cpp
	${MPICXX} ${DEFINES} \
		${CFLAGS} \
		-c $^ -o $@

transpose: ${obj} ./obj/transpose.o
	${LINKER} \
		./obj/transpose.o \
		${obj} \
		-o transpose \
		${LIBS} \
		${NULL}

resize: ${obj} ./obj/resize.o
	${LINKER} \
		./obj/resize.o \
		${obj} \
		-o resize \
		${LIBS} \
		${NULL}

clean:
	rm ./obj/*.o
	rm t2D
