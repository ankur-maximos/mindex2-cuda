include ../Makefile.in
LIBDIR:=../$(LIBDIR)
NRMCL=$(LIBDIR)/$(LIBNAME)
LDFLAGS=$(NRMCL) -openmp $(LDOPTIONS) 

CFLAGS = $(COPTIONS) $(OPTFLAGS) $(CINCLUDES)
CFLAGS += -I$(LIBDIR)
CUFLAGS = $(CUOPTFLAGS) $(CUINCLUDES) -I$(LIBDIR)
CUFLAGS = -arch sm_20 -O3 -I../nlibs
#CUFLAGS = -arch sm_20 -O0 -g -G -I../nlibs
#MKL_FLAGS=-m64  -w -I"/opt/intel/composer_xe_2013.4.183/mkl/include" -I/home/niuq/tools/gperftools-2.1/build/include
#MKL_LDFLAGS=-Wl,--start-group "/opt/intel/composer_xe_2013.4.183/mkl/lib/intel64/libmkl_intel_lp64.a" "/opt/intel/composer_xe_2013.4.183/mkl/lib/intel64/libmkl_intel_thread.a" "/opt/intel/composer_xe_2013.4.183/mkl/lib/intel64/libmkl_core.a" -Wl,--end-group -L"/opt/intel/composer_xe_2013.4.183/mkl/../compiler/lib/intel64" -liomp5 -lpthread -lm -ldl /home/niuq/tools/gperftools-2.1/build/lib/libprofiler.a

SOURCES=$(wildcard *.cc)
CUS=$(wildcard *.cu)
CUHS=$(wildcard *.cuh)
CUOBJS=$(shell echo $(CUS) | sed s/.cu/.o/g)
OBJS=$(shell echo $(SOURCES) | sed s/.cc/.o/g)
EXES=$(shell echo $(SOURCES) | sed s/.cc/.x/g)
all:$(EXES)

CC=icpc
%.x:%.o $(CUOBJS) $(NRMCL) 
	@echo $(LDFLAGS)
	$(CC) -o $@ $< $(CUOBJS) $(LDFLAGS) 

%.o: %.cu $(CUHS)
	$(NVCC) $(CUFLAGS) $< -c -o $@

$(NRMCL):$(shell find $(LIBDIR) -name '*.cc' -o -name '*.h' -o -name '*.cu')
	(cd $(LIBDIR); make)

%.o:%.cc 
	$(CC) $(CFLAGS) $^ -c 

clean:
	rm -rf *.o *.x

diskclean:
	rm -rf *.o *.x
	make -C $(LIBDIR) clean

test:
	for exe in $(shell ls *.x); \
	do \
		echo $${exe}; \
		./$${exe}; \
	done


#icpc -m64  -w -I"/opt/intel/composer_xe_2013.4.183/mkl/include"         ./source/nqp.cc -Wl,--start-group         "/opt/intel/composer_xe_2013.4.183/mkl/lib/intel64/libmkl_intel_lp64.a"         "/opt/intel/composer_xe_2013.4.183/mkl/lib/intel64/libmkl_intel_thread.a"         "/opt/intel/composer_xe_2013.4.183/mkl/lib/intel64/libmkl_core.a"         -Wl,--end-group -L"/opt/intel/composer_xe_2013.4.183/mkl/../compiler/lib/intel64" -liomp5 -lpthread -lm -ldl -o _results/intel_lp64_parallel_intel64_lib/nqp.out
