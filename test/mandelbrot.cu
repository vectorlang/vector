#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <libvector.hpp>

#define MAX_ITER 1000
#define LEFT_MIN -2.5
#define RIGHT_MAX 1
#define BOTTOM_MIN -1
#define TOP_MAX 1

#define IMG_HEIGHT 256
#define IMG_WIDTH 384
#define BLOCK_WIDTH 32
#define BLOCK_HEIGHT 16

__global__
void mandelbrot(uchar3 *colors, uchar3 *colorMap,
        float left, float right, float top, float bottom)
{
    size_t xi, yi, xn, yn, i;
    float x0, y0, x, y, xtemp;
    int iter = 0;

    xi = threadIdx.x + blockDim.x * blockIdx.x;
    yi = threadIdx.y + blockDim.y * blockIdx.y;
    xn = blockDim.x * gridDim.x;
    yn = blockDim.y * gridDim.y;
    i = xn * yi + xi;

    x0 = left + (right - left) / xn * xi;
    y0 = bottom + (top - bottom) / yn * yi;

    x = y = 0;

    while (iter < MAX_ITER && (x * x + y * y) < 4) {
        xtemp = x * x - y * y + x0;
        y = 2 * x * y + y0;
        x = xtemp;
        iter++;
    }

    colors[i] = colorMap[iter];
}

int write_ppm(FILE *f, VectorArray<uchar3> &colors,
        unsigned int width, unsigned int height)
{
    size_t x, y, i;
    uchar3 pixel;

    fprintf(f, "P3\n");
    fprintf(f, "%u %u\n", width, height);
    fprintf(f, "255\n");

    for (y = 0; y < height; y++) {
        for (x = 0; x < width; x++) {
            i = y * width + x;
            pixel = colors.elem(i);
            fprintf(f, "%d %d %d ", pixel.x, pixel.y, pixel.z);
        }
        fprintf(f, "\n");
    }
    return 0;
}

int main(int argc, char *argv[])
{
    float left, right, top, bottom;
    VectorArray<uchar3> colors(1, IMG_WIDTH * IMG_HEIGHT);
    VectorArray<uchar3> colorMap(1, MAX_ITER + 1);
    dim3 grid_dim(IMG_WIDTH / BLOCK_WIDTH, IMG_HEIGHT / BLOCK_HEIGHT, 1);
    dim3 block_dim(BLOCK_WIDTH, BLOCK_HEIGHT, 1);
    int i;

    left = (argc > 1) ? atof(argv[1]) : -2.0;
    right = (argc > 2) ? atof(argv[2]) : 1.0;
    top = (argc > 3) ? atof(argv[3]) : 1.0;
    bottom = (argc > 4) ? atof(argv[4]) : -1.0;

    if (left < LEFT_MIN || right > RIGHT_MAX
            || top > TOP_MAX || bottom < BOTTOM_MIN) {
        fprintf(stderr, "Window out of bounds\n");
        exit(EXIT_FAILURE);
    }

    for (i = 0; i < MAX_ITER; i++) {
        colorMap.elem(i).x = i & 0xff;
        colorMap.elem(i).y = i & 0xff;
        colorMap.elem(i).z = i & 0xff;
    }

    colorMap.elem(MAX_ITER).x = 0;
    colorMap.elem(MAX_ITER).y = 0;
    colorMap.elem(MAX_ITER).z = 0;

    colorMap.copyToDevice();

    mandelbrot<<<grid_dim, block_dim>>>(colors.devPtr(), colorMap.devPtr(),
            left, right, top, bottom);
    cudaDeviceSynchronize();
    checkError(cudaGetLastError());

    colors.copyFromDevice();

    if (write_ppm(stdout, colors, IMG_WIDTH, IMG_HEIGHT))
        exit(EXIT_FAILURE);

    return 0;
}
