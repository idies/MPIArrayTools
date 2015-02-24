MPICXX  = mpicxx
LINKER  = mpicxx
DEFINES =
CFLAGS  = -Wall \
		  -O2 \
		  #-pg \
		  #-finstrument-functions

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

base_files := \
	field_descriptor \
	fftwf_tools \
	Morton_shuffler \
	RMHD_converter

#headers := $(patsubst %, ./src/%.hpp, ${base_files})
src := $(patsubst %, ./src/%.cpp, ${base_files})
obj := $(patsubst %, ./obj/%.o, ${base_files})

.PRECIOUS: ./obj/%.o

./obj/%.o: ./src/%.cpp
	${MPICXX} ${DEFINES} \
		${CFLAGS} \
		-c $^ -o $@

base: ${obj}

%.elf: ${obj} ./obj/%.o
	${LINKER} \
		$^ \
		-o $@ \
		${LIBS} \
		${NULL}

clean:
	rm -f ./obj/*.o
	rm -f *.elf

