NVCC_FLAGS=-arch=sm_20 -g
NVCC=nvcc
LDFLAGS=-g -locelot

all: vecmult

%.o: %.cu
	$(NVCC) -c $(NVCC_FLAGS) $< -o $@

%: %.o
	$(CXX) $< -o $@ $(LDFLAGS)

clean:
	rm -f *.o vecmult
