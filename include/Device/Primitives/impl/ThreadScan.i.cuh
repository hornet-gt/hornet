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
#include "Base/Device/Primitives/CudaFunctional.cuh"
#include "Base/Device/Primitives/ThreadReduce.cuh"

namespace xlib {
namespace ThreadInclusiveScanILP {
namespace detail {

template<int SIZE, typename T, cuda_functional::binary_op<T> BinaryOP,
         int STRIDE = SIZE / 4>
struct ThreadInclusiveScanSupport {
    static_assert(IsPower2<SIZE>::value,
                  "ThreadReduce : SIZE must be a power of 2");

    __device__ __forceinline__ static void DownSweepRight(T (&Array)[SIZE]) {
        #pragma unroll
        for (int INDEX = STRIDE * 2; INDEX < SIZE; INDEX += STRIDE * 2)
            Array[INDEX - 1 + STRIDE] = BinaryOP(Array[INDEX - 1],
                                                 Array[INDEX - 1 + STRIDE]);
        ThreadInclusiveScanSupport<SIZE, T, BinaryOP, STRIDE / 2>
            ::DownSweepRight(Array);
    }
};

template<int SIZE, typename T, cuda_functional::binary_op<T> BinaryOP>
struct ThreadInclusiveScanSupport<SIZE, T, BinaryOP, 0> {
    __device__ __forceinline__ static void DownSweepRight(T (&Array)[SIZE]) {}
};

} // namespace detail

//==========================================================================

template<typename T, int SIZE>
__device__ __forceinline__ static void Add(T (&Array)[SIZE]) {
    /*using namespace cuda_functional;
    using namespace detail;
    ThreadReduce::detail::
        ThreadReduceSupport<SIZE, T, plus<T>>::UpSweepRight(Array);
    ThreadInclusiveScanSupport<SIZE, T, plus<T>>::DownSweepRight(Array);*/
}

} // namespace ThreadInclusiveScanILP

namespace ThreadInclusiveScan {

template<typename T, int SIZE>
__device__ __forceinline__ static void Add(T (&Array)[SIZE]) {
    #pragma unroll
    for (int i = 1; i < SIZE; i++)
        Array[i] += Array[i - 1];
}

template<typename T>
__device__ __forceinline__ static void Add(T* Array, const int size) {
    for (int i = 1; i < size; i++)
        Array[i] += Array[i - 1];
}

} // namespace ThreadInclusiveScan

namespace ThreadExclusiveScan {
    template<typename T, int SIZE>
    __device__ __forceinline__ static void Add(T (&Array)[SIZE]) {
        T tmp = Array[0], tmp2;
        Array[0] = 0;
        #pragma unroll
        for (int i = 1; i < SIZE; i++) {
            tmp2 = Array[i];
            Array[i] = tmp;
            tmp += tmp2;
        }
    }

    template<typename T>
    __device__ __forceinline__ static void Add(T* Array, const int size) {
        T tmp = Array[0], tmp2;
        Array[0] = 0;
        for (int i = 1; i < size; i++) {
            tmp2 = Array[i];
            Array[i] = tmp;
            tmp += tmp2;
        }
    }

} // namespace ThreadExclusiveScan
} // namespace xlib
