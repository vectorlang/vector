CFLAGS=-Wall -O2
NVCC_FLAGS=-arch=sm_20 -g -Xcompiler $(CFLAGS)
NVCC=nvcc
LDFLAGS=-g -locelot

EXAMPLES=vecmult mandelbrot

all: $(EXAMPLES)

%.o: %.cu
	$(NVCC) -c $(NVCC_FLAGS) $< -o $@

%: %.o
	$(CXX) $< -o $@ $(LDFLAGS)

clean:
	rm -f *.o $(EXAMPLES)
