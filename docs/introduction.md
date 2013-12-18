# Vector: A High-Level Programming Language for the GPU
*Harry Lee (hhl2114), Howard Mao (zm2169), Zachary Newman (zjn2101), Sidharth Shanker (sps2133), Jonathan Yu (jy2432)*

##1. Introduction

As the single-core performance of CPUs plateaus and becomes constrained by
power consumption, more and more high-performance computing (HPC) must be done
in parallel in order to see any performance increases. The current state of the
art in HPC systems is using GPUs for compute-heavy tasks, due to the intrinsic
parallelism of GPU architectures. However, programming GPUs remains difficult
due to a lack of language tools. Currently, the only mature general-purpose
GPU computing languages are low level languages, such as OpenCL and CUDA,
which expose a lot of the incidental complexity of GPU hardware to the GPU programmer.

To address these issues, we have implemented *Vector*, a high-level programming
language for the GPU. In Vector, GPU computation becomes almost as simple as
CPU computation. The compiler abstracts away details such as allocating memory
on the GPU, copying memory between CPU and GPU, and choosing the proper work
sizes (i.e. number of threads per block and number of blocks per grid).
In addition, certain parallel programming idioms (such as `map` and `reduce`)
are supported natively in Vector as higher-order primitives.

Our strategy for implementing this language is to generate CUDA code, which can
then be compiled and run on Nvidia GPUs. We chose to generate CUDA rather than
the more platform-independent OpenCL due to our greater familiarity with CUDA.
Also, most HPC systems use CUDA-compatible Nvidia Tesla GPUs. During development,
we tested using [GPU Ocelot][], an emulator for Nvidia GPUs that runs PTX
(the intermediate representation for CUDA) on the CPU.

###1.1 Language Features

* basic type inference for assignments
* parallel for (`pfor`) loops: generate a CUDA kernel which is compiled into
  PTX and run on the GPU
* map/reduce: syntactic sugar built on top of `pfor`
  (because these generate kernels, we need to implement them as primitives)
* first-class functions (sort of): we can only implement these at compile-time
* handles memory allocation and communication between CPU and GPU
* abstracts grid and block sizing for kernels
* handles higher-dimensional kernel indices (CUDA handles at most 3)

###1.2 Sample Program

    __device__ int add(int x, int y) { return x + y; }

    /*
     * Compute the dot product of two vectors
     */
    int dot_product(int x[], int y[]) {
        int z[len(x)];

        pfor (i in 0:len(x)) {
            // each iteration of the for loop runs on a different thread on the GPU
            z[i] = x[i] * y[i];
        }

        return @reduce(add, z);
    }

The equivalent CUDA program would look like this.

    #include <stdio.h>
    #include <stdlib.h>
    #include <stdint.h>
    #include <libvector.hpp>

    __device__ int32_t add (int32_t, int32_t);
    __global__ void
    __sym5_ (int32_t * output, int32_t * input, size_t n)
    {
      extern __shared__ int32_t temp[];

      int ti = threadIdx.x;
      int bi = blockIdx.x;
      int starti = blockIdx.x * blockDim.x;
      int gi = starti + ti;
      int bn = min (n - starti, blockDim.x);
      int s;

      if (ti < bn)
        temp[ti] = input[gi];
      __syncthreads ();

      for (s = 1; s < blockDim.x; s *= 2)
        {
          if (ti % (2 * s) == 0 && ti + s < bn)
        temp[ti] = add (temp[ti], temp[ti + s]);
          __syncthreads ();
        }

      if (ti == 0)
        output[bi] = temp[0];
    }

    int32_t
    __sym4_ (VectorArray < int32_t > arr)
    {
      int n = arr.size ();
      int num_blocks = ceil_div (n, BLOCK_SIZE);
      int atob = 1;
      int shared_size = BLOCK_SIZE * sizeof (int32_t);
      VectorArray < int32_t > tempa (1, num_blocks);
      VectorArray < int32_t > tempb (1, num_blocks);

      __sym5_ <<< num_blocks, BLOCK_SIZE, shared_size >>> (tempa.devPtr (),
                                   arr.devPtr (), n);
      cudaDeviceSynchronize ();
      checkError (cudaGetLastError ());
      tempa.markDeviceDirty ();
      n = num_blocks;

      while (n > 1)
        {
          num_blocks = ceil_div (n, BLOCK_SIZE);
          if (atob)
        {
          __sym5_ <<< num_blocks, BLOCK_SIZE,
            shared_size >>> (tempb.devPtr (), tempa.devPtr (), n);
          tempb.markDeviceDirty ();
        }
          else
        {
          __sym5_ <<< num_blocks, BLOCK_SIZE,
            shared_size >>> (tempa.devPtr (), tempb.devPtr (), n);
          tempa.markDeviceDirty ();
        }
          cudaDeviceSynchronize ();
          checkError (cudaGetLastError ());
          atob = !atob;
          n = num_blocks;
        }

      if (atob)
        {
          tempa.copyFromDevice (1);
          return tempa.elem (false, 0);
        }
      tempb.copyFromDevice (1);
      return tempb.elem (false, 0);
    }

    __global__ void
    __sym3_ (struct range_iter *__sym6_, size_t __sym7_, size_t __sym8_,
         device_info < int32_t > *x, device_info < int32_t > *y,
         device_info < int32_t > *z)
    {
      size_t __sym9_ = threadIdx.x + blockIdx.x * blockDim.x;
      if (__sym9_ < __sym8_)
        {
          size_t i = get_index_gpu (&__sym6_[0], __sym9_);
          {
        z->values[get_mid_index < int32_t > (z, i, 0)] =
          (x->values[get_mid_index < int32_t > (x, i, 0)]) *
          (y->values[get_mid_index < int32_t > (y, i, 0)]);
          }
        }
    }

    __device__ int32_t
    add (int32_t x, int32_t y)
    {
      return (x) + (y);
    }

    int32_t
    dot_product (VectorArray < int32_t > x, VectorArray < int32_t > y)
    {
      VectorArray < int32_t > z (1, (x).size ());
      {
        struct range_iter __sym0_[1];
        __sym0_[0].start = 0;
        __sym0_[0].stop = (x).size ();
        __sym0_[0].inc = 1;
        fillin_iters (__sym0_, 1);
        struct range_iter *__sym1_ = device_iter (__sym0_, 1);
        size_t __sym2_ = total_iterations (__sym0_, 1);
        __sym3_ <<< ceil_div (__sym2_, BLOCK_SIZE), BLOCK_SIZE >>> (__sym1_, 1,
                                    __sym2_,
                                    x.devInfo (),
                                    y.devInfo (),
                                    z.devInfo ());
        cudaDeviceSynchronize ();
        checkError (cudaGetLastError ());
        z.markDeviceDirty ();
        cudaFree (__sym1_);
      }

      return __sym4_ (z);
    }

    int
    main (void)
    {
      return vec_main ();
    }

