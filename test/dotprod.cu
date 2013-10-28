#include <stdlib.h>
#include <stdio.h>
#include "libvector.hpp"

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
    VectorArray<int> x = array_init<int>(4, 1, 2, 3, 4);
    VectorArray<int> y = array_init<int>(4, 3, 5, 7, 9);
    VectorArray<int> dp(1, 4);

    x.copyToDevice();
    y.copyToDevice();

    dotprod_kernel<<<1, 4>>>(dp.devPtr(), x.devPtr(), y.devPtr(), 4);
    cudaDeviceSynchronize();
    checkError(cudaGetLastError());

    dp.copyFromDevice();
    printf("Dot product: %d\n", dp.elem(0));

    return 0;
}
