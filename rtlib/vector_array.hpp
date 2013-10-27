#ifndef __VECTOR_ARRAY_H__
#define __VECTOR_ARRAY_H__

#include <stdlib.h>
#include <stdarg.h>
#include "vector_utils.hpp"

using namespace std;

template <class T>
class VectorArray {
	private:
		T *values;
		T *d_values;
		size_t ndims;
		size_t *dims;
		size_t nelems;
	public:
		VectorArray();
		VectorArray(size_t ndims, ...);
		T &elem(size_t first_ind, ...);
		~VectorArray();
		size_t bsize();
		size_t size();
		size_t length(size_t dim = 0);
		void deviceAllocate();
		void copyToDevice();
		void copyFromDevice();
		T *devPtr();
};

template <class T>
VectorArray<T>::VectorArray()
{
	this->ndims = 0;
	this->dims = NULL;
	this->values = NULL;
	this->nelems = 0;
}

template <class T>
VectorArray<T>::VectorArray(size_t ndims, ...)
{
	size_t i;
	va_list dim_list;

	va_start(dim_list, ndims);

	this->ndims = ndims;
	this->dims = (size_t *) calloc(ndims, sizeof(size_t));
	this->nelems = 1;

	for (i = 0; i < ndims; i++) {
		this->dims[i] = va_arg(dim_list, size_t);
		this->nelems *= this->dims[i];
	}

	va_end(dim_list);

	this->values = (T *) calloc(this->nelems, sizeof(T));
	this->d_values = NULL;
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
	if (this->d_values != NULL)
		cudaFree(this->d_values);
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

template <class T>
size_t VectorArray<T>::size()
{
	return this->nelems;
}

template <class T>
size_t VectorArray<T>::bsize()
{
	return sizeof(T) * this->nelems;
}

template <class T>
size_t VectorArray<T>::length(size_t dim)
{
	return this->dims[dim];
}

template <class T>
void VectorArray<T>::deviceAllocate()
{
	cudaError_t err;
	if (this->d_values == NULL) {
		err = cudaMalloc(&this->d_values, bsize());
		checkError(err);
	}
}

template <class T>
void VectorArray<T>::copyToDevice()
{
	cudaError_t err;
	deviceAllocate();
	err = cudaMemcpy(this->d_values, this->values,
			bsize(), cudaMemcpyHostToDevice);
	checkError(err);
}

template <class T>
void VectorArray<T>::copyFromDevice()
{
	cudaError_t err;
	if (this->d_values == NULL) {
		fprintf(stderr, "The device data has not yet been allocated\n");
		exit(EXIT_FAILURE);
	}
	err = cudaMemcpy(this->values, this->d_values,
			bsize(), cudaMemcpyDeviceToHost);
	checkError(err);
}

template <class T>
T *VectorArray<T>::devPtr()
{
	deviceAllocate();
	return this->d_values;
}

#endif
