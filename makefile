MPICXX  = mpicxx
LINKER  = mpicxx
DEFINES =
CFLAGS  = -Wall \
		  -O2

LIBS = -lfftw3_mpi \
	   -lfftw3 \
	   -lfftw3f_mpi \
	   -lfftw3f

COMPILER_VERSION := $(shell ${MPICXX} --version)

ifneq (,$(findstring ICC,$(COMPILER_VERSION)))
	# using intel compiler
	# advice from
	# https://software.intel.com/en-us/forums/topic/298872
	# always link against both libimf and libm
    LIBS += -limf \
			-lm
else
    # not using intel compiler
endif

vpath %.cpp ./src/

src := \
	field_descriptor.cpp \
	fftwf_tools.cpp \
	RMHD_converter.cpp

obj := $(patsubst %.cpp, ./obj/%.o, ${src})

./obj/%.o: %.cpp
	${MPICXX} ${DEFINES} \
		${CFLAGS} \
		-c $^ -o $@

exec = \
	   transpose \
	   resize \
	   resize_and_transpose \
	   full

transpose: ${obj} ./obj/transpose.o
	${LINKER} \
		./obj/transpose.o \
		${obj} \
		-o $@ \
		${LIBS} \
		${NULL}

resize: ${obj} ./obj/resize.o
	${LINKER} \
		./obj/resize.o \
		${obj} \
		-o $@ \
		${LIBS} \
		${NULL}

resize_and_transpose: ${obj} ./obj/resize_and_transpose.o
	${LINKER} \
		./obj/resize_and_transpose.o \
		${obj} \
		-o $@ \
		${LIBS} \
		${NULL}

full: ${obj} ./obj/full.o
	${LINKER} \
		./obj/full.o \
		${obj} \
		-o $@ \
		${LIBS} \
		${NULL}

clean:
	rm ./obj/*.o
	rm -f ${exec}

