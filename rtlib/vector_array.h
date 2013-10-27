#ifndef __VECTOR_ARRAY_H__
#define __VECTOR_ARRAY_H__

#include <stdlib.h>

using namespace std;

template <class T>
class VectorArray {
	private:
		T *values;
		size_t ndims;
		size_t *dims;
	public:
		VectorArray();
		VectorArray(size_t ndims, ...);
		T &elem(size_t first_ind, ...);
		~VectorArray();
};

#endif