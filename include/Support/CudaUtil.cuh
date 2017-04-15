/*------------------------------------------------------------------------------
Copyright © 2016 by Nicola Bombieri

XLib is provided under the terms of The MIT License (MIT):

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
------------------------------------------------------------------------------*/
/**
 * @author Federico Busato
 * Univerity of Verona, Dept. of Computer Science
 * federico.busato@univr.it
 */
#pragma once

#include "Support/VectorUtil.cuh"
#include <limits>
#include <string>
#include <cuda_runtime.h>

#if !defined(NO_CHECK_CUDA_ERROR)
    #define CHECK_CUDA_ERROR                                                   \
        {                                                                      \
            cudaDeviceSynchronize();                                           \
            xlib::__getLastCudaError("", __FILE__, __LINE__, __func__);        \
        }
    #define CUDA_ERROR(msg)                                                    \
        {                                                                      \
            cudaDeviceSynchronize();                                           \
            xlib::__getLastCudaError(msg, __FILE__, __LINE__, __func__);       \
        }
#else
    #define CUDA_ERROR()
    #define CUDA_ERROR(msg)
#endif

#define SAFE_CALL(function)                                                    \
    {                                                                          \
        xlib::__safe_call(function, __FILE__, __LINE__, __func__);             \
    }

namespace xlib {

using  cusize_t = int;
using cusize2_t = typename make2_str<int>::type;

enum THREAD_GROUP { VOID = 0, WARP, BLOCK };

template<typename T>
struct numeric_limits {         // available in CUDA kernels
    static const T    min = std::numeric_limits<T>::min();
    static const T    max = std::numeric_limits<T>::max();
    static const T lowest = std::numeric_limits<T>::lowest();
};

//------------------------------------------------------------------------------

//iteratorB_t = device
template<bool FAULT = true, class iteratorA_t, class iteratorB_t>
bool cuEqual(iteratorA_t start_A, iteratorA_t end_A, iteratorB_t start_B);

template<bool FAULT = true, class iteratorA_t, class iteratorB_t>
bool cuEqual(iteratorA_t start_A, iteratorA_t end_A, iteratorB_t start_B,
             bool (*equalFunction)(
                typename std::iterator_traits<iteratorA_t>::value_type,
                typename std::iterator_traits<iteratorB_t>::value_type));

template<bool FAULT = true, class iteratorA_t, class iteratorB_t>
bool cuEqualSorted(iteratorA_t start_A, iteratorA_t end_A,
                   iteratorB_t start_B);


void __getLastCudaError(const char* error_message, const char* file, int line,
                        const char* func_name);

void __cudaErrorHandler(cudaError_t err, const char* error_message,
                        const char* file, int line, const char* func_name);

void __safe_call(cudaError_t err, const char* file, int line,
                 const char* func_name);

class deviceProperty {
    public:
        static int getNum_of_SMs();
    private:
        static int NUM_OF_STREAMING_MULTIPROCESSOR;
};

template<class T>
inline unsigned gridConfig(T FUN, unsigned block_dim,
                           unsigned dyn_shared_mem = 0,
                           int problem_size = std::numeric_limits<int>::max());

bool memInfoCUDA(std::size_t requested);
void memCheckCUDA(std::size_t requested);
void deviceInfo();
/*
template<typename T>
__global__ void scatter(const int* __restrict__ toScatter,
                        int scatter_size, T* __restrict__ Dest, T value);

template<typename T>
__global__ void fill(T* devArray, int fill_size, T value);

template <typename T>
__global__ void fill(T* devMatrix, int n_of_rows, int n_of_columns,
                     T value, int integer_pitch = 0);

template <unsigned UNROLLING = 4, typename T>
__global__ void copy_unroll(T* __restrict__ Input, const int size,
                            T* __restrict__ Output);

template <typename T>
__global__ void copy(T* __restrict__ Input,
                     const int size,
                     T* __restrict__ Output);*/
} // namespace xlib

namespace NVTX {

const int GREEN = 0x0000ff00, BLUE = 0x000000ff, YELLOW = 0x00ffff00,
          PURPLE = 0x00ff00ff, CYAN = 0x0000ffff, RED = 0x00ff0000,
          WHITE = 0x00ffffff;

void PushRange(const std::string& s, const int color);
void PopRange();

} // namespace NVTX

#include "impl/CudaUtil.i.cuh"
