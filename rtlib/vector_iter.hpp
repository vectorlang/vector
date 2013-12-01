#ifndef __VECTOR_ITER_H__
#define __VECTOR_ITER_H__

struct range_iter {
	size_t len;
	size_t mod;
	size_t div;
	size_t start;
	size_t inc;
};

void fillin_moddiv(struct range_iter *iters, size_t n)
{
	int i;
	size_t last_mod = 1;

	for (i = n - 1; i >= 0; i--) {
		iters[i].div = last_mod;
		iters[i].mod = last_mod * iters[i].len;
		last_mod = iters[i].mod;
	}
}

size_t get_index(struct range_iter *iter, size_t oned_ind)
{
	return iter->start + (oned_ind % iter->mod) / iter->div * iter->inc;
}

#endif
