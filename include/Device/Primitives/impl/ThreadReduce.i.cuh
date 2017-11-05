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
#include "Base/Host/Basic.hpp"
#include "Base/Host/Numeric.hpp"

namespace xlib {
namespace thread_reduce {
namespace detail {

template<int SIZE, int STRIDE = 1>
struct ThreadReduceSupport {
    static_assert(IsPower2<SIZE>::value,
                  "ThreadReduce : SIZE must be a power of 2");

    template<typename T, typename Lambda>
    __device__ __forceinline__
    static void upSweepLeft(T (&array)[SIZE], const Lambda& lambda) {
        #pragma unroll
        for (int INDEX = 0; INDEX < SIZE; INDEX += STRIDE * 2)
            array[INDEX] = lambda(array[INDEX], array[INDEX + STRIDE]);
        ThreadReduceSupport<SIZE, STRIDE * 2>::upSweepLeft(array, lambda);
    }

    /*__device__ __forceinline__
    static void UpSweepRight(T (&array)[SIZE]) {
        #pragma unroll
        for (int INDEX = STRIDE - 1; INDEX < SIZE; INDEX += STRIDE * 2) {
            array[INDEX + STRIDE] = BinaryOP(array[INDEX],
                                             array[INDEX + STRIDE]);
        }
        ThreadReduceSupport<SIZE, T, BinaryOP, STRIDE * 2>::UpSweepRight(array);
    }*/
};

template<int SIZE>
struct ThreadReduceSupport<SIZE, SIZE> {
    template<typename T, typename Lambda>
    __device__ __forceinline__
    static void upSweepLeft(T (&)[SIZE], const Lambda&) {}
    //__device__ __forceinline__ static void UpSweepRight(T (&array)[SIZE]) {}
};

} // namespace detail

//==============================================================================

template<typename T, int SIZE>
__device__ __forceinline__ static void add(T (&array)[SIZE]) {
    const auto lambda = [](T a, T b){ return a + b; };
    detail::ThreadReduceSupport<SIZE>::upSweepLeft(array, lambda);
}

template<typename T, int SIZE>
__device__ __forceinline__ static void min(T (&array)[SIZE]) {
    const auto lambda = [](T a, T b){ return min(a, b); };
    detail::ThreadReduceSupport<SIZE>::upSweepLeft(array, lambda);
}

template<typename T, int SIZE>
__device__ __forceinline__ static void max(T (&array)[SIZE]) {
    const auto lambda = [](T a, T b){ return max(a, b); };
    detail::ThreadReduceSupport<SIZE>::upSweepLeft(array, lambda);
}

template<typename T, int SIZE>
__device__ __forceinline__ static void logicAnd(T (&array)[SIZE]) {
    const auto lambda = [](T a, T b){ return a && b; };
    detail::ThreadReduceSupport<SIZE>::upSweepLeft(array, lambda);
}

template<typename T, int SIZE, typename Lambda>
__device__ __forceinline__ static void custom(T (&array)[SIZE], Lambda lambda) {
    detail::ThreadReduceSupport<SIZE>::upSweepLeft(array, lambda);
}

} // namespace thread_reduce
} // namespace xlib
