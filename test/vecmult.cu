#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "vector_utils.hpp"

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
    long a[VEC_SIZE], b[VEC_SIZE], c[VEC_SIZE];
    long *d_a, *d_b, *d_c;
    int i;
    cudaError_t err;
    size_t numblocks, numthreads;

    d_a = d_b = d_c = NULL;

    srandom(time(0));

    for (i = 0; i < VEC_SIZE; i++) {
        a[i] = random();
        b[i] = random();
    }

    err = cudaMalloc(&d_a, sizeof(a));
    if (err != cudaSuccess)
        goto cleanup;
    err = cudaMalloc(&d_b, sizeof(b));
    if (err != cudaSuccess)
        goto cleanup;
    err = cudaMalloc(&d_c, sizeof(c));
    if (err != cudaSuccess)
        goto cleanup;

    err = cudaMemcpy(d_a, a, sizeof(a), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
        goto cleanup;
    err = cudaMemcpy(d_b, b, sizeof(b), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
        goto cleanup;

    numblocks = ceil_div(VEC_SIZE, BLOCK_SIZE);
    numthreads = BLOCK_SIZE;
    vec_mult<<<numblocks, numthreads>>>(d_a, d_b, d_c, VEC_SIZE);

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
        goto cleanup;

    err = cudaMemcpy(c, d_c, sizeof(c), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
        goto cleanup;

    err = cudaSuccess;

    for (i = 0; i < VEC_SIZE; i++) {
        if (c[i] != a[i] * b[i]) {
            fprintf(stderr, "Mismatched number %ld\n", c[i]);
            goto cleanup;
        }
    }
    printf("Computation was correct\n");

cleanup:
    if (err != cudaSuccess && err > 0)
        fprintf(stderr, "%s\n", cudaGetErrorString(err));
    if (d_a != NULL)
        cudaFree(d_a);
    if (d_b != NULL)
        cudaFree(d_b);
    if (d_c != NULL)
        cudaFree(d_c);
    return err;
}
