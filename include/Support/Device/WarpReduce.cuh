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

#include "Base/Host/Basic.hpp"

namespace xlib {

template<int WARP_SZ = 32>//!!!!!!!!!!!!! if WARP_SZ == 1
struct WarpReduce {
    static_assert(IsPower2<WARP_SZ>::value &&
                  WARP_SZ >= 1 && WARP_SZ <= 32,
                  "WarpReduce : WARP_SZ must be a power of 2 and\
                                2 <= WARP_SZ <= 32");

    template<typename T>
    static __device__ __forceinline__ void add(T& value);

    template<typename T>
    static __device__ __forceinline__ void min(T& value);

    template<typename T>
    static __device__ __forceinline__ void max(T& value);

    //--------------------------------------------------------------------------

    template<typename T>
    static __device__ __forceinline__ void addAll(T& value);

    template<typename T>
    static __device__ __forceinline__ void minAll(T& value);

    template<typename T>
    static __device__ __forceinline__ void maxAll(T& value);

    //--------------------------------------------------------------------------

    template<typename T>
    static __device__ __forceinline__ void add(T& value, T* pointer);

    template<typename T>
    static __device__ __forceinline__ void min(T& value, T* pointer);

    template<typename T>
    static __device__ __forceinline__ void max(T& value, T* pointer);

    //--------------------------------------------------------------------------

    //template<typename T>
    //static __device__ __forceinline__ T atomicAdd(const T& value, T* pointer);

    template<typename T>
    static __device__ __forceinline__ T atomicAdd(const T& value, T* pointer);

    //template<typename T>
    //static __device__ __forceinline__ T rAtomicAdd(T& value, T* pointer);

    template<typename T>
    static __device__ __forceinline__
    void atomicMin(const T& value, T* pointer);

    template<typename T>
    static __device__ __forceinline__
    void atomicMax(const T& value, T* pointer);

    //--------------------------------------------------------------------------

    /*template<typename T>
    __device__ __forceinline__
    static void atomicadd(T& value1, T* pointer1, T& value2, T* pointer2);*/
};

} // namespace xlib

#include "impl/WarpReduce.i.cuh"
