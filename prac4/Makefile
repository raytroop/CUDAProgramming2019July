
INC	:= -I$(CUDA_HOME)/include -I.
LIB	:= -L$(CUDA_HOME)/lib64 -lcudart

NVCCFLAGS	:= -lineinfo -arch=sm_35 --ptxas-options=-v --use_fast_math

all: reduction reductionV2

reduction:	reduction.cu Makefile
		nvcc reduction.cu -o reduction $(INC) $(NVCCFLAGS) $(LIB)

reductionV2:	reductionV2.cu Makefile
		nvcc reductionV2.cu -o reductionV2 $(INC) $(NVCCFLAGS) $(LIB)

clean:
		rm -f reduction reductionV2

