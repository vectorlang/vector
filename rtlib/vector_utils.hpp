#ifndef __VECTOR_UTILS_H__
#define __VECTOR_UTILS_H__

#include <sys/time.h>

static inline void _check(cudaError_t err, const char *file, int line)
{
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s:%d\n", file, line);
        fprintf(stderr, "%s\n", cudaGetErrorString(err));
        exit(err);
    }
}

static inline double get_time(void)
{
	struct timeval tv;

	gettimeofday(&tv, NULL);

	return (double) tv.tv_sec + ((double) tv.tv_usec) / 1000000.0;
}

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 1024
#endif
#define checkError(err) _check((err), __FILE__, __LINE__)
#define ceil_div(n, d) (((n) - 1) / (d) + 1)
#define min(a, b) (((a) < (b)) ? (a) : (b))
#define max(a, b) (((a) > (b)) ? (a) : (b))

#endif
