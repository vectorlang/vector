#include <stdlib.h>
#include <stdio.h>
#include "utils.h"

__global__
void dotprod_kernel(int *dotprod, int *x, int *y, size_t n)
{
    size_t i = threadIdx.x + blockDim.x * blockIdx.x;
    size_t s;

    if (i < n)
        dotprod[i] = x[i] * y[i];
    
    for (s = 1; s < n; s *= 2) {
        if (i % (2 * s) == 0 && i + s < n)
            dotprod[i] += dotprod[i + s];
        __syncthreads();
    }
}

int main(void)
{
    int x[4] = {1, 2, 3, 4};
    int y[4] = {3, 5, 7, 9};
    int dp;

    int *d_x, *d_y, *d_dp;
    
    cudaError_t err;
    
    err = cudaMalloc(&d_x, sizeof(x));
    checkError(err);
    err = cudaMalloc(&d_y, sizeof(y));
    checkError(err);
    err = cudaMalloc(&d_dp, sizeof(x));
    checkError(err);

    err = cudaMemcpy(d_x, x, sizeof(x), cudaMemcpyHostToDevice);
    checkError(err);
    err = cudaMemcpy(d_y, y, sizeof(y), cudaMemcpyHostToDevice);
    checkError(err);

    dotprod_kernel<<<1, 4>>>(d_dp, d_x, d_y, 4);    
    cudaDeviceSynchronize();
    checkError(cudaGetLastError());

    cudaMemcpy(&dp, d_dp, sizeof(dp), cudaMemcpyDeviceToHost);
    printf("Dot product: %d\n", dp);

    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_dp);

    return 0;
}
