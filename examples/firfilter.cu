#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <limits.h>
#include "utils.h"

static inline __device__ float atomicAddFloat(float *addr, float f)
{
    int *iaddr = (int *) addr;
    int assumed, oldval, newval;

    oldval = *iaddr;
    
    do {
        assumed = oldval;
        newval = __float_as_int(f + __int_as_float(oldval));
        oldval = atomicCAS(iaddr, assumed, newval);
    } while (assumed != oldval);

    return __int_as_float(oldval);
}

__global__
void firfilter(float *output, float *signal, float *response,
        unsigned int slen, unsigned int rlen)
{
    extern __shared__ float sharedy[];
    unsigned int n = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int i = threadIdx.y + blockDim.y * blockIdx.y;
    unsigned int ti = threadIdx.y;
    unsigned int tlen = blockDim.y;
    unsigned int s;

    if (i < rlen) {
        if (n >= i && n < slen)
            sharedy[ti] = response[i] * signal[n - i];
        else
            sharedy[ti] = 0;
    }
    __syncthreads();

    for (s = 1; s <= tlen / 2; s *= 2) {
        if (ti % (2 * s) == 0)
            sharedy[ti] += sharedy[ti + s];
        __syncthreads();
    }

    if (ti == 0 && n < slen)
        atomicAddFloat(&output[n], sharedy[0]);
}

#define BLOCK_SIZE 512
#define SIGNAL_LEN 4096
#define RESPONSE_LEN 1024
#define TIME_STEP (5.0 / SIGNAL_LEN)

void init_signal(float *signal, size_t n)
{
    unsigned int i, j;
    float freq = 130.81;
    float theta;
    float harmonics[4] = {0.5, 0.25, 0.125, 0.125};
    
    memset(signal, 0, sizeof(float) * n);

    for (i = 0; i < 4; i++) {
        for (j = 0; j < n; j++) {
            theta = j * TIME_STEP * 2 * M_PI * freq;
            signal[j] += harmonics[i] * sin(theta);
        }
        freq *= 2;
    }
}

void init_response(float *response, size_t n)
{
    float cutoff = 200.0;
    float starttime = -TIME_STEP * n / 2;
    float t;
    unsigned int i;
    
    for (i = 0; i < n; i++) {
        t = starttime + i * TIME_STEP;
        response[i] = cutoff * sin(cutoff * t) / (cutoff * t);
    }
}

void validate_result(float *output, float *signal, float *response)
{
    float reference;
    float diff, maxerr = 0;
    int n, i;
    
    for (n = 0; n < SIGNAL_LEN; n++) {
        reference = 0;
        for (i = 0; i < RESPONSE_LEN; i++) {
            if (n >= i)
                reference += (response[i] * signal[n - i]);
        }
        diff = output[n] - reference;
        diff = (diff < 0) ? -diff : diff;

        if (diff > maxerr)
            maxerr = diff;
    }

    printf("Maximum error: %f\n", maxerr);
}

int main(void)
{
    dim3 grid_dim(SIGNAL_LEN, RESPONSE_LEN / BLOCK_SIZE, 1);
    dim3 block_dim(1, BLOCK_SIZE, 1);

    float output[SIGNAL_LEN], signal[SIGNAL_LEN], response[RESPONSE_LEN];
    float *d_output, *d_signal, *d_response;

    cudaError_t err;
    size_t shm_size = sizeof(float) * BLOCK_SIZE;

    init_signal(signal, SIGNAL_LEN);
    init_response(response, RESPONSE_LEN);

    err = cudaMalloc(&d_output, sizeof(output));
    checkError(err);
    err = cudaMalloc(&d_signal, sizeof(signal));
    checkError(err);
    err = cudaMalloc(&d_response, sizeof(response));
    checkError(err);
    
    err = cudaMemset(d_output, 0, sizeof(output));
    checkError(err);
    err = cudaMemcpy(d_signal, signal, sizeof(signal), cudaMemcpyHostToDevice);
    checkError(err);
    err = cudaMemcpy(d_response, response, sizeof(response), cudaMemcpyHostToDevice);
    checkError(err);

    firfilter<<<grid_dim, block_dim, shm_size>>>(
            d_output, d_signal, d_response, SIGNAL_LEN, RESPONSE_LEN);
    cudaDeviceSynchronize();
    checkError(cudaGetLastError());

    err = cudaMemcpy(output, d_output, sizeof(output), cudaMemcpyDeviceToHost);
    checkError(err);

    validate_result(output, signal, response);

    cudaFree(d_output);
    cudaFree(d_response);
    cudaFree(d_signal);

    return 0;
}
