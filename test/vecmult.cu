#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <libvector.hpp>

#define VEC_SIZE 5000

__global__
void vec_mult_kernel(long *a, long *b, long *c, size_t n)
{
    size_t i = threadIdx.x + blockDim.x * blockIdx.x;

    if (i < n)
        c[i] = a[i] * b[i];
}

VectorArray<long> vec_mult(VectorArray<long> &a, VectorArray<long> &b)
{
    VectorArray<long> c(1, a.size());
    size_t numblocks, numthreads;

    a.copyToDevice();
    b.copyToDevice();

    numblocks = ceil_div(c.size(), BLOCK_SIZE);
    numthreads = BLOCK_SIZE;
    vec_mult_kernel<<<numblocks, numthreads>>>(a.devPtr(), b.devPtr(),
            c.devPtr(), c.size());

    cudaDeviceSynchronize();
    checkError(cudaGetLastError());

    c.copyFromDevice();

    return c;
}

int main(void)
{
    VectorArray<long> a(1, VEC_SIZE), b(1, VEC_SIZE);
    int i;

    srandom(time(0));

    for (i = 0; i < VEC_SIZE; i++) {
        a.elem(i) = random();
        b.elem(i) = random();
    }

    a.copyToDevice();
    b.copyToDevice();

    VectorArray<long> c = vec_mult(a, b);

    for (i = 0; i < VEC_SIZE; i++) {
        if (c.elem(i) != a.elem(i) * b.elem(i)) {
            fprintf(stderr, "Mismatched number %ld\n", c.elem(i));
            return -1;
        }
    }
    printf("Computation was correct\n");

    return 0;
}
