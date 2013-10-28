#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <libvector.hpp>

#define VEC_SIZE 5000
#define BLOCK_SIZE 512

__global__
void vec_mult(long *a, long *b, long *c, size_t n)
{
    size_t i = threadIdx.x + blockDim.x * blockIdx.x;

    if (i < n)
        c[i] = a[i] * b[i];
}

int main(void)
{
    VectorArray<long> a(1, VEC_SIZE), b(1, VEC_SIZE), c(1, VEC_SIZE);
    int i;
    cudaError_t err;
    size_t numblocks, numthreads;

    srandom(time(0));

    for (i = 0; i < VEC_SIZE; i++) {
        a.elem(i) = random();
        b.elem(i) = random();
    }

    a.copyToDevice();
    b.copyToDevice();

    numblocks = ceil_div(VEC_SIZE, BLOCK_SIZE);
    numthreads = BLOCK_SIZE;
    vec_mult<<<numblocks, numthreads>>>(a.devPtr(), b.devPtr(), c.devPtr(), VEC_SIZE);

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
        goto cleanup;

    c.copyFromDevice();

    err = cudaSuccess;

    for (i = 0; i < VEC_SIZE; i++) {
        if (c.elem(i) != a.elem(i) * b.elem(i)) {
            fprintf(stderr, "Mismatched number %ld\n", c.elem(i));
            goto cleanup;
        }
    }
    printf("Computation was correct\n");

cleanup:
    if (err != cudaSuccess && err > 0)
        fprintf(stderr, "%s\n", cudaGetErrorString(err));
    return err;
}
