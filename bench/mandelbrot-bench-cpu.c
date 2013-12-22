#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

float left = -2.0f;
float right = 1.0f;
float top = 1.0f;
float bottom = -1.0f;

static inline double get_time(void)
{
	struct timeval tv;

	gettimeofday(&tv, NULL);

	return (double) tv.tv_sec + ((double) tv.tv_usec) / 1000000.0;
}

uint8_t mandelbrot(int xi, int yi, int xn, int yn)
{
	uint8_t iter;

	float x0 = left + (right - left) / xn * xi;
	float y0 = bottom + (top - bottom) / yn * yi;
	float x = 0.0f;
	float y = 0.0f;
	float xtemp;

	while (x * x + y * y < 4 && iter < 255) {
		xtemp = x * x - y * y + x0;
		y = 2 * x * y + y0;
		x = xtemp;
		iter++;
	}

	return iter;
}

void compute_shades(uint8_t *shades, int width, int height)
{
	int i, xi, yi;

	for (i = 0; i < width * height; i++) {
		xi = i % width;
		yi = i / width;
		shades[i] = mandelbrot(xi, yi, width, height);
	}
}

void benchmark(int width, int height, int ntrials)
{
	uint8_t shades[height * width];
	double runtimes[ntrials];
	double start, finish;
	int trial;

	for (trial = 0; trial < ntrials; trial++)
	{
		start = get_time();
		compute_shades(shades, width, height);
		finish = get_time();
		runtimes[trial] = finish - start;
	}

	printf("CPU %dx%d: ", width, height);

	for (trial = 0; trial < ntrials; trial++)
		printf("%.4f ", runtimes[trial]);

	printf("\n");
}

int main(void)
{
	int img_heights[5] = {480, 600, 768, 864, 960};
	int img_widths[5] = {640, 800, 1024, 1152, 1280};
	int i;

	for (i = 0; i < 5; i++)
		benchmark(img_widths[i], img_heights[i], 3);

	return 0;

}
