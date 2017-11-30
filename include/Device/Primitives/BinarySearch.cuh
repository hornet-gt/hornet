/**
 * @author Federico Busato                                                  <br>
 *         Univerity of Verona, Dept. of Computer Science                   <br>
 *         federico.busato@univr.it
 * @date November, 2017
 * @version v1.4
 *
 * @copyright Copyright © 2017 XLib. All rights reserved.
 *
 * @license{<blockquote>
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright notice, this
 *   list of conditions and the following disclaimer.
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 * * Neither the name of the copyright holder nor the names of its
 *   contributors may be used to endorse or promote products derived from
 *   this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 * </blockquote>}
 *
 * @file
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
