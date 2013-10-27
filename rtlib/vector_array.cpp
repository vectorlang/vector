#include "vector_array.h"
#include <stdarg.h>

template <class T>
VectorArray<T>::VectorArray()
{
	this->ndims = 0;
	this->dims = NULL;
	this->values = NULL;
}

template <class T>
VectorArray<T>::VectorArray(size_t ndims, ...)
{
	size_t i, total_size = 1;
	va_list dim_list;

	va_start(dim_list, ndims);

	this->ndims = ndims;
	this->dims = (size_t *) calloc(ndims, sizeof(size_t));

	for (i = 0; i < ndims; i++) {
		this->dims[i] = va_arg(dim_list, size_t);
		total_size *= this->dims[i];
	}

	va_end(dim_list);

	this->values = (T *) calloc(ndims, sizeof(T));
}

template <class T>
T &VectorArray<T>::elem(size_t first_ind, ...)
{
	size_t ind = first_ind, onedind = first_ind;
	int i;
	va_list indices;

	va_start(indices, first_ind);

	for (i = 1; i < this->ndims; i++) {
		ind = va_arg(indices, size_t);
		onedind = onedind * this->dims[i] + ind;
	}

	va_end(indices);

	return this->values[onedind];
}

template <class T>
VectorArray<T>::~VectorArray()
{
	if (this->dims != NULL)
		free(this->dims);
	if (this->values != NULL)
		free(this->values);
}
