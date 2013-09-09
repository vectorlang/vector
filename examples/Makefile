NVCC_FLAGS=-arch=sm_11 -g
NVCC=nvcc
LDFLAGS=-g -L/opt/cuda/lib64 -lcudart

all: vecmult

%.o: %.cu
	$(NVCC) -c $(NVCC_FLAGS) $< -o $@

%: %.o
	$(CXX) $< -o $@ $(LDFLAGS)

clean:
	rm -f *.o vecmult
