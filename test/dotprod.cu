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
    VectorArray<int> x(1, 4);
    VectorArray<int> y(1, 4);
    VectorArray<int> dp(1, 4);

    for (int i = 0; i < 4; i++) {
        x.elem(true, i) = i + 1;
        y.elem(true, i) = 2 * i + 3;
    }

    x.copyToDevice();
    y.copyToDevice();

    dotprod_kernel<<<1, 4>>>(dp.devPtr(), x.devPtr(), y.devPtr(), 4);
    cudaDeviceSynchronize();
    checkError(cudaGetLastError());

    dp.copyFromDevice();
    printf("Dot product: %d\n", dp.elem(false, 0));

    return 0;
}
