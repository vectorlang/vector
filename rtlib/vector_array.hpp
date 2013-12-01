#ifndef __VECTOR_ARRAY_H__
#define __VECTOR_ARRAY_H__

#include <stdlib.h>
#include <stdarg.h>
#include "vector_utils.hpp"

using namespace std;

struct array_ctrl {
	int refcount;
	char h_dirty;
	char d_dirty;
};

template <class T>
struct device_info {
	T *values;
	size_t *dims;
	size_t ndims;
};

template <class T>
class VectorArray {
	private:
		T *values;
		T *d_values;
		size_t ndims;
		size_t *dims;
		size_t *d_dims;
		size_t nelems;
		struct array_ctrl *ctrl;
		struct device_info<T> *dev_info;
		size_t bsize();
		void incRef();
		void decRef();
	public:
		VectorArray();
		VectorArray(size_t ndims, ...);
		VectorArray(const VectorArray<T> &orig);
		VectorArray<T> dim_copy(void);
                T &oned_elem(size_t ind);
		T &elem(bool modify, size_t first_ind, ...);
		VectorArray<T> &chain_set(size_t ind, T val);
		VectorArray<T>& operator= (const VectorArray<T> &orig);
		~VectorArray();
		size_t size();
		size_t length(size_t dim = 0);
		void copyToDevice(size_t n = 0);
		void copyFromDevice(size_t n = 0);
		T *devPtr();
		struct device_info<T> *devInfo();
		void markDeviceDirty(void);
};

template <class T>
VectorArray<T>::VectorArray()
{
	this->ndims = 0;
	this->dims = NULL;
	this->d_dims = NULL;
	this->values = NULL;
	this->d_values = NULL;
	this->dev_info = NULL;
	this->nelems = 0;
	this->ctrl = (struct array_ctrl *) malloc(sizeof(struct array_ctrl));
	this->ctrl->refcount = 1;
	this->ctrl->h_dirty = 0;
	this->ctrl->d_dirty = 0;
}

template <class T>
VectorArray<T>::VectorArray(size_t ndims, ...)
{
	size_t i;
	va_list dim_list;
	cudaError_t err;
	struct device_info<T> h_dev_info;

	va_start(dim_list, ndims);

	this->ndims = ndims;
	this->dims = (size_t *) calloc(ndims, sizeof(size_t));
	this->nelems = 1;
	this->ctrl = (struct array_ctrl *) malloc(sizeof(struct array_ctrl));
	this->ctrl->refcount = 1;
	this->ctrl->h_dirty = 0;
	this->ctrl->d_dirty = 0;

	for (i = 0; i < ndims; i++) {
		this->dims[i] = va_arg(dim_list, size_t);
		this->nelems *= this->dims[i];
	}

	va_end(dim_list);

	this->values = (T *) calloc(this->nelems, sizeof(T));
	err = cudaMalloc(&this->d_values, bsize());
	checkError(err);

	err = cudaMalloc(&this->d_dims, sizeof(size_t) * ndims);
	checkError(err);
	err = cudaMemcpy(this->d_dims, this->dims, sizeof(size_t) * ndims,
			cudaMemcpyHostToDevice);
	checkError(err);

	h_dev_info.ndims = ndims;
	h_dev_info.dims = this->d_dims;
	h_dev_info.values = this->d_values;
	err = cudaMalloc(&this->dev_info, sizeof(h_dev_info));
	checkError(err);
	err = cudaMemcpy(this->dev_info, &h_dev_info, sizeof(h_dev_info),
			cudaMemcpyHostToDevice);
	checkError(err);
}

template <class T>
VectorArray<T>::VectorArray(const VectorArray<T> &orig)
{
	this->ndims = orig.ndims;
	this->dims = orig.dims;
	this->d_dims = orig.d_dims;
	this->nelems = orig.nelems;
	this->ctrl = orig.ctrl;
	this->values = orig.values;
	this->d_values = orig.d_values;
	this->dev_info = orig.dev_info;

	incRef();
}

template <class T>
VectorArray<T> VectorArray<T>::dim_copy(void)
{
	VectorArray<T> copy;
	cudaError_t err;

	copy.ndims = this->ndims;
	copy.dims = (size_t *) calloc(copy.ndims, sizeof(size_t));

	for (int i = 0; i < this->ndims; i++)
		copy.dims[i] = this->dims[i];

	copy.nelems = this->nelems;
	copy.values = (T *) calloc(this->nelems, sizeof(T));
	err = cudaMalloc(&copy.d_values, copy.bsize());
	checkError(err);

	return copy;
}

template <class T>
VectorArray<T>& VectorArray<T>::operator= (const VectorArray<T>& orig)
{
	// avoid self-assignment
	if (this == &orig)
		return *this;

	decRef();

	this->ndims = orig.ndims;
	this->dims = orig.dims;
	this->d_dims = orig.d_dims;
	this->nelems = orig.nelems;
	this->ctrl = orig.ctrl;
	this->values = orig.values;
	this->d_values = orig.d_values;
	this->dev_info = orig.dev_info;

	incRef();

	return *this;
}

template <class T>
void VectorArray<T>::incRef(void)
{
	this->ctrl->refcount++;
}

template <class T>
void VectorArray<T>::decRef(void)
{
	if (--(this->ctrl->refcount) > 0)
		return;
	free(this->ctrl);
	if (this->dims != NULL)
		free(this->dims);
	if (this->d_dims != NULL)
		cudaFree(this->d_dims);
	if (this->values != NULL)
		free(this->values);
	if (this->d_values != NULL)
		cudaFree(this->d_values);
	if (this->dev_info != NULL)
		cudaFree(this->dev_info);
}

template <class T>
T &VectorArray<T>::oned_elem(size_t ind)
{
	if (this->ctrl->d_dirty)
		copyFromDevice();

	this->ctrl->h_dirty = 1;
	return this->values[ind];
}

template <class T>
T &VectorArray<T>::elem(bool modify, size_t first_ind, ...)
{
	size_t ind = first_ind, onedind = first_ind;
	int i;
	va_list indices;

	if (this->ctrl->d_dirty)
		copyFromDevice();

	if (modify)
		this->ctrl->h_dirty = 1;

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
	decRef();
}

template <class T>
VectorArray<T>& VectorArray<T>::chain_set(size_t ind, T value)
{
	oned_elem(ind) = value;
	return *this;
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
void VectorArray<T>::copyToDevice(size_t n)
{
        if (n == 0)
            n = size();
	cudaError_t err;
	err = cudaMemcpy(this->d_values, this->values,
			sizeof(T) * n, cudaMemcpyHostToDevice);
	checkError(err);
	this->ctrl->h_dirty = 0;
}

template <class T>
void VectorArray<T>::copyFromDevice(size_t n)
{
	cudaError_t err;
        if (n == 0)
            n = size();
        err = cudaMemcpy(this->values, this->d_values,
                        sizeof(T) * n, cudaMemcpyDeviceToHost);
	checkError(err);
	this->ctrl->d_dirty = 0;
}

template <class T>
T *VectorArray<T>::devPtr()
{
	if (this->ctrl->h_dirty)
		copyToDevice();
	return this->d_values;
}

template <class T>
struct device_info<T> *VectorArray<T>::devInfo()
{
	devPtr();
	return this->dev_info;
}

template <class T>
void VectorArray<T>::markDeviceDirty(void)
{
	this->ctrl->d_dirty = 1;
}

#endif
