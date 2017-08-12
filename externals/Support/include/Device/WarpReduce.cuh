/**
 * @internal
 * @author Federico Busato                                                  <br>
 *         Univerity of Verona, Dept. of Computer Science                   <br>
 *         federico.busato@univr.it
 * @date August, 2017
 * @version v2
 *
 * @copyright Copyright © 2017 cuStinger. All rights reserved.
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
