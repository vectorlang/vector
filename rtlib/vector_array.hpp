#ifndef __VECTOR_ARRAY_H__
#define __VECTOR_ARRAY_H__

#include <stdlib.h>
#include <stdarg.h>

using namespace std;

template <class T>
class VectorArray {
	private:
		T *values;
		size_t ndims;
		size_t *dims;
		size_t size;
	public:
		VectorArray();
		VectorArray(size_t ndims, ...);
		T &elem(size_t first_ind, ...);
		~VectorArray();
};

template <class T>
VectorArray<T>::VectorArray()
{
	this->ndims = 0;
	this->dims = NULL;
	this->values = NULL;
	this->size = NULL;
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

	this->size = total_size * sizeof(T);
	this->values = (T *) malloc(this->size);
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

template <class T>
VectorArray<T> array_init(size_t length, ...)
{
    VectorArray<T> array(1, length);
    va_list elem_list;
    int i;

    va_start(elem_list, length);

    for (i = 0; i < length; i++)
        array.elem(i) = va_arg(elem_list, T);

    va_end(elem_list);

    return array;
}

#endif
