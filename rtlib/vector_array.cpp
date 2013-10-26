#include "vector_array.h"

template <class T>
VectorArray<T>::VectorArray()
{
	this->ndims = 0;
	this->dims = NULL;
	this->values = NULL;
}

template <class T>
VectorArray<T>::VectorArray(size_t ndims, size_t *dims)
{
	this->ndims = ndims;
	this->dims = dims;
	this->values = NULL;
}

template <class T>
VectorArray<T>::VectorArray(size_t ndims, size_t *dims, T *values)
{
	this->ndims = ndims;
	this->dims = dims;
	this->values = values;
}

template <class T>
void VectorArray<T>::set_dims(size_t ndims, size_t *dims)
{
	if (this->dims != NULL)
		delete[] this->dims;
	this->dims = dims;
	this->ndims = ndims;
}

template <class T>
void VectorArray<T>::set_values(T *values)
{
	if (this->values != NULL)
		delete[] this->values;
	this->values = values;
}

template <class T>
T &VectorArray<T>::elem(size_t *indices)
{
	size_t onedind = 0;
	int i;

	for (i = 0; i < this->ndims - 1; i++)
		onedind = (onedind + indices[i]) * this->dims[i+1];
	onedind += indices[this->ndims - 1];

	return this->values[onedind];
}

template <class T>
VectorArray<T>::~VectorArray()
{
	if (this->dims != NULL)
		delete[] this->dims;
	if (this->values != NULL)
		delete[] this->values;
}
