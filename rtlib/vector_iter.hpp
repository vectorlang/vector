#ifndef __VECTOR_ITER_H__
#define __VECTOR_ITER_H__

#include "vector_utils.hpp"

struct range_iter {
	size_t start;
	size_t stop;
	size_t inc;
	size_t len;
	size_t mod;
	size_t div;
};

void fillin_iters(struct range_iter *iters, size_t n)
{
	int i;
	size_t last_mod = 1;

	for (i = n - 1; i >= 0; i--) {
		iters[i].len = ceil_div(iters[i].stop - iters[i].start,
					iters[i].inc);
		iters[i].div = last_mod;
		iters[i].mod = last_mod * iters[i].len;
		last_mod = iters[i].mod;
	}
}

size_t get_index(struct range_iter *iter, size_t oned_ind)
{
	return iter->start + (oned_ind % iter->mod) / iter->div * iter->inc;
}

size_t total_iterations(struct range_iter *iter, size_t n)
{
	int total = 1;
	int i;

	for (i = 0; i < n; i++)
		total *= iter[i].len;

	return total;
}

struct range_iter *device_iter(struct range_iter *iters, size_t n)
{
	cudaError_t err;
	struct range_iter *d_iters;

	err = cudaMalloc(&d_iters, n * sizeof(struct range_iter));
	checkError(err);
	err = cudaMemcpy(d_iters, iters, n * sizeof(struct range_iter),
			cudaMemcpyHostToDevice);
	checkError(err);

	return err;
}

#endif
