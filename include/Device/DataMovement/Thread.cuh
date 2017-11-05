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

#include "Base/Device/Util/Basic.cuh"
#include <type_traits>

namespace xlib {

template<typename T, int SIZE>
__device__ __forceinline__
void vector_copy(T (&queue)[SIZE], T* ptr) {
    const int CAST_SIZE = (SIZE * sizeof(T)) % 16 == 0 ? 16 :
                          (SIZE * sizeof(T)) % 8 == 0 ? 8 :
                          (SIZE * sizeof(T)) % 4 == 0 ? 4 :
                          (SIZE * sizeof(T)) % 2 == 0 ? 2 : 1;
    using R = xlib::Pad<CAST_SIZE>;
    const int LOOPS = (SIZE * sizeof(T)) / CAST_SIZE;

    #pragma unroll
    for (int i = 0; i < LOOPS; i++)
        reinterpret_cast<R*>(ptr)[i] = reinterpret_cast<R*>(queue)[i];
}

template<typename T, typename Lambda>
__device__ __forceinline__
void vector_op(const T* ptr, int size, const Lambda& lambda) {
    using R1 = typename std::conditional<
                sizeof(T) <= 16 && xlib::IsPower2<sizeof(T)>::value, int4, T>
              ::type;
    using R2 = int2;
    using R3 = int;
    using R4 = short;

    const unsigned SIZE = sizeof(R1) / sizeof(T);
    T queue[SIZE];
    int size_loop = size / SIZE;

    for (int i = 0; i < size_loop; i++) {
        reinterpret_cast<R1*>(queue)[0] = reinterpret_cast<const R1*>(ptr)[i];
        #pragma unroll
        for (int j = 0; j < SIZE; j++)
            lambda(queue[j]);
    }
    if (sizeof(R1) == sizeof(T))
        return;
    int remain = size - size_loop * SIZE;
    ptr       += size_loop * SIZE;
    //--------------------------------------------------------------------------
    if (remain * sizeof(T) >= sizeof(R2)) {
        reinterpret_cast<R2*>(queue)[0] = reinterpret_cast<const R2*>(ptr)[0];
        const int L_SIZE = sizeof(R2) / sizeof(T);
        #pragma unroll
        for (int j = 0; j < L_SIZE; j++)
            lambda(queue[j]);
        if (sizeof(R2) == sizeof(T))
            return;
        remain -= L_SIZE;
        ptr    += L_SIZE;
    }
    //--------------------------------------------------------------------------
    if (remain * sizeof(T) >= sizeof(R3)) {
        reinterpret_cast<R3*>(queue)[0] = reinterpret_cast<const R3*>(ptr)[0];
        const int L_SIZE = sizeof(R3) / sizeof(T);
        #pragma unroll
        for (int j = 0; j < L_SIZE; j++)
            lambda(queue[j]);
        if (sizeof(R3) == sizeof(T))
            return;
        remain -= L_SIZE;
        ptr    += L_SIZE;
    }
    //--------------------------------------------------------------------------
    if (remain * sizeof(T) >= sizeof(R4)) {
        reinterpret_cast<R4*>(queue)[0] = reinterpret_cast<const R4*>(ptr)[0];
        const int L_SIZE = sizeof(R4) / sizeof(T);
        #pragma unroll
        for (int j = 0; j < L_SIZE; j++)
            lambda(queue[j]);
        if (sizeof(R4) == sizeof(T))
            return;
        remain -= L_SIZE;
        ptr    += L_SIZE;
    }
    //--------------------------------------------------------------------------
    if (remain != 0)
        lambda(*ptr);
}



template<typename T, typename Lambda>
__device__ __forceinline__
bool vector_first_of(const T* ptr, int size, const Lambda& lambda) {
    using R1 = typename std::conditional<
                sizeof(T) <= 16 && xlib::IsPower2<sizeof(T)>::value, int4, T>
              ::type;
    using R2 = int2;
    using R3 = int;
    using R4 = short;

    const unsigned SIZE = sizeof(R1) / sizeof(T);
    T queue[SIZE];
    int size_loop = size / SIZE;

    for (int i = 0; i < size_loop; i++) {
        reinterpret_cast<R1*>(queue)[0] = reinterpret_cast<const R1*>(ptr)[i];
        #pragma unroll
        for (int j = 0; j < SIZE; j++) {
            if (lambda(queue[j]))
                return true;
        }
    }
    if (sizeof(R1) == sizeof(T))
        return false;
    int remain = size - size_loop * SIZE;
    ptr       += size_loop * SIZE;
    //--------------------------------------------------------------------------
    if (remain * sizeof(T) >= sizeof(R2)) {
        reinterpret_cast<R2*>(queue)[0] = reinterpret_cast<const R2*>(ptr)[0];
        const int L_SIZE = sizeof(R2) / sizeof(T);
        #pragma unroll
        for (int j = 0; j < L_SIZE; j++) {
            if (lambda(queue[j]))
                return true;
        }
        if (sizeof(R2) == sizeof(T))
            return false;
        remain -= L_SIZE;
        ptr    += L_SIZE;
    }
    //--------------------------------------------------------------------------
    if (remain * sizeof(T) >= sizeof(R3)) {
        reinterpret_cast<R3*>(queue)[0] = reinterpret_cast<const R3*>(ptr)[0];
        const int L_SIZE = sizeof(R3) / sizeof(T);
        #pragma unroll
        for (int j = 0; j < L_SIZE; j++) {
            if (lambda(queue[j]))
                return true;
        }
        if (sizeof(R3) == sizeof(T))
            return false;
        remain -= L_SIZE;
        ptr    += L_SIZE;
    }
    //--------------------------------------------------------------------------
    if (remain * sizeof(T) >= sizeof(R4)) {
        reinterpret_cast<R4*>(queue)[0] = reinterpret_cast<const R4*>(ptr)[0];
        const int L_SIZE = sizeof(R4) / sizeof(T);
        #pragma unroll
        for (int j = 0; j < L_SIZE; j++) {
            if (lambda(queue[j]))
                return true;
        }
        if (sizeof(R4) == sizeof(T))
            return false;
        remain -= L_SIZE;
        ptr    += L_SIZE;
    }
    //--------------------------------------------------------------------------
    if (remain != 0)
        return lambda(*ptr);
    return false;
}

} // namespace xlib
