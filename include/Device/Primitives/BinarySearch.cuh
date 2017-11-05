/*------------------------------------------------------------------------------
Copyright © 2016 by Nicola Bombieri

XLib is provided under the terms of The MIT License (MIT):

Permission is hereby granted, free of charge, to any person obtaining a copy of
this oftware and associated documentation files (the "Software"), to deal in
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
/**
 * @file
 * @version v1.3
 */
#pragma once

#include "Base/Host/Numeric.hpp"

namespace xlib {

template<unsigned SIZE, typename T>
__device__ __forceinline__
int binary_search_pow2(const T* shared_mem, T searched) {
    static_assert(IsPower2<SIZE>::value, "SIZE must be a power of 2");
    int low = 0;
    #pragma unroll
    for (int i = 1; i <= Log2<SIZE>::value; i++) {
        int pos = low + ((SIZE) >> i);
        if (searched >= shared_mem[pos])
            low = pos;
    }
    return low;
}

template<typename T>
__device__ __forceinline__
int binary_search_warp(T reg_value, T searched) {
    int low = 0;
    #pragma unroll
    for (int i = 1; i <= Log2<WARP_SIZE>::value; i++) {
        int pos = low + ((WARP_SIZE) >> i);
        if (searched >= __shfl(reg_value, pos))
            low = pos;
    }
    return low;
}


#include <cassert>

//@@@ deprecated
// the searched value must be in the intervall
template<typename T>
__device__ __forceinline__
void binarySearch(T* mem, const T searched,
                  int& pos, int size) {
    int start = 0, end = size - 1;
    pos = end / 2u;

    while (start < end) {
        assert(pos + 1 < size);
        if (searched >= mem[pos + 1])
            start = pos + 1;
        else if (searched < mem[pos])
            end = pos - 1;
        else
            break;
        pos = (start + end) / 2u;
    }
}

} // namespace xlib
