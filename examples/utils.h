#ifndef __VECTOR_UTILS_H__
#define __VECTOR_UTILS_H__

static inline void _check(cudaError_t err, const char *file, int line)
{
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s:%d\n", file, line);
        fprintf(stderr, "%s\n", cudaGetErrorString(err));
        exit(err);
    }
}

#define checkError(err) _check((err), __FILE__, __LINE__)

#endif
