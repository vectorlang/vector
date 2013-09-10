#include <stdio.h>
#include <stdlib.h>
#include <time.h>

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

static void _check(cudaError_t err, const char *file, int line)
{
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s:%d\n", file, line);
        fprintf(stderr, "%s\n", cudaGetErrorString(err));
        exit(err);
    }
}

#define checkError(err) _check((err), __FILE__, __LINE__)

int write_ppm(const char *fname, uchar3 *colors, 
        unsigned int width, unsigned int height)
{
    FILE *f = fopen(fname, "w");
    size_t x, y, i;
    if (f == NULL) {
        perror("fopen");
        return -1;
    }

    fprintf(f, "P3\n");
    fprintf(f, "%u %u\n", width, height);
    fprintf(f, "255\n");

    for (y = 0; y < height; y++) {
        for (x = 0; x < width; x++) {
            i = y * width + x;
            fprintf(f, "%d %d %d ", colors[i].x, colors[i].y, colors[i].z);
        }
        fprintf(f, "\n");
    }
    return 0;
}

int main(int argc, char *argv[])
{
    float left, right, top, bottom;
    uchar3 colors[IMG_WIDTH * IMG_HEIGHT], colorMap[MAX_ITER + 1];
    uchar3 *d_colors = NULL, *d_colorMap;
    cudaError_t err;
    dim3 grid_dim(IMG_WIDTH / BLOCK_WIDTH, IMG_HEIGHT / BLOCK_HEIGHT, 1);
    dim3 block_dim(BLOCK_WIDTH, BLOCK_HEIGHT, 1);
    int i;
    long ran;

    if (argc < 5) {
        fprintf(stderr, "Usage: %s left right top bottom\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    left = atof(argv[1]);
    right = atof(argv[2]);
    top = atof(argv[3]);
    bottom = atof(argv[4]);

    if (left < LEFT_MIN || right > RIGHT_MAX 
            || top > TOP_MAX || bottom < BOTTOM_MIN) {
        fprintf(stderr, "Window out of bounds\n");
        exit(EXIT_FAILURE);
    }

    srandom(time(0));

    for (i = 0; i < MAX_ITER; i++) {
        ran = random();
        colorMap[i].x = (ran >> 16) & 0xff;
        colorMap[i].y = (ran >> 8) & 0xff;
        colorMap[i].z = ran & 0xff;
    }

    colorMap[MAX_ITER].x = 0;
    colorMap[MAX_ITER].y = 0;
    colorMap[MAX_ITER].z = 0;

    err = cudaMalloc(&d_colors, sizeof(colors));
    checkError(err);

    err = cudaMemset(d_colors, 0, sizeof(colors));

    err = cudaMalloc(&d_colorMap, sizeof(colorMap));
    checkError(err);

    err = cudaMemcpy(d_colorMap, colorMap, sizeof(colorMap), cudaMemcpyHostToDevice);
    checkError(err);

    mandelbrot<<<grid_dim, block_dim>>>(d_colors, d_colorMap, 
            left, right, top, bottom);
    cudaDeviceSynchronize();
    checkError(cudaGetLastError());

    err = cudaMemcpy(colors, d_colors, sizeof(colors), cudaMemcpyDeviceToHost);
    checkError(err);

    if (write_ppm("mandelbrot.ppm", colors, IMG_WIDTH, IMG_HEIGHT))
        exit(EXIT_FAILURE);

    cudaFree(d_colors);
    cudaFree(d_colorMap);

    return 0;
}
