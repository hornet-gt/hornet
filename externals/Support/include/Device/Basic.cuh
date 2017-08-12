/**
 * @internal
 * @author Federico Busato                                                  <br>
 *         Univerity of Verona, Dept. of Computer Science                   <br>
 *         federico.busato@univr.it
 * @date April, 2017
 * @version v1.3
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

#include "Device/CudaUtil.cuh"

/** @namespace basic
 *  provide basic cuda functions
 */
namespace xlib {

extern __shared__ char dyn_smem[];

template<int SIZE>
using Pad = typename std::conditional<SIZE == 16, int4,
            typename std::conditional<SIZE == 8, int2,
            typename std::conditional<SIZE == 4, int,
            typename std::conditional<SIZE == 2, short,
            char>::type>::type>::type>::type;

template<int SIZE, typename T>
 __device__ __forceinline__ void reg_fill(T (&reg)[SIZE], T value);

template<int SIZE, typename T>
 __device__ __forceinline__
 void reg_copy(const T (&reg1)[SIZE], T (&reg2)[SIZE]);

/**
 *  @brief return the warp ID within the block
 *
 *  Provide the warp ID within the current block.
 *  @return warp ID in the range 0 &le; ID &le; 32
 */
__device__ __forceinline__ unsigned warp_id();

template<unsigned BLOCK_SIZE, unsigned VIRTUAL_WARP = 1>
__device__ __forceinline__ unsigned global_id();

template<unsigned BLOCK_SIZE, unsigned VIRTUAL_WARP = 1>
__device__ __forceinline__ unsigned global_stride();

/**
 *  @brief return the warp ID within the block
 *
 *  Provide the warp ID within the current block.
 *  @return warp ID in the range 0 &le; ID &le; (blockDim.x / WARP_SIZE)
 */
template<unsigned WARP_SZ = WARP_SIZE>
__device__ __forceinline__ unsigned warp_base();

/** @fn T WarpBroadcast(T value, int predicate)
 *  @brief broadcast 'value' of the last lane that evaluates 'predicate' to true
 *
 *  @return 'value' of the last lane that evaluates 'predicate' to true
 */
template<typename T>
__device__ __forceinline__ T warp_broadcast(T value, int predicate);

template<unsigned VW_SIZE>
struct VWarp {
    __device__ __forceinline__
    VWarp();

    __device__ __forceinline__
    bool any(bool pred) const;

    __device__ __forceinline__
    bool all(bool pred) const;

    __device__ __forceinline__
    unsigned ballot(bool pred) const;

    __device__ __forceinline__
    static unsigned mask();
private:
    const unsigned _mask;
};

namespace detail {

template<typename T, typename Lambda>
__device__ __forceinline__
T generic_shfl(const T& var, int src_lane, int width, const Lambda& lambda) {
    const int NUM = (sizeof(T) + sizeof(int) - 1) / sizeof(int);

    int tmp[NUM];
    reinterpret_cast<T&>(tmp) = var;
    #pragma unroll
    for (int i = 0; i < NUM; i++)
        tmp[i] = lambda(tmp[i], src_lane, width);
    return reinterpret_cast<T&>(tmp);
}

} // namespace detail

template<typename T>
__device__ __forceinline__
T shfl(const T& var, int src_lane, int width = 32) {
    const auto& lambda = [](const int& value, int src_lane, int width) {
                            return __shfl(value, src_lane, width);
                        };
    return detail::generic_shfl(var, src_lane, width, lambda);
}

template<typename T>
__device__ __forceinline__
T shfl_xor(const T& var, int src_lane, int width = 32) {
    const auto& lambda = [](const int& value, int src_lane, int width) {
                            return __shfl_xor(value, src_lane, width);
                        };
    return detail::generic_shfl(var, src_lane, width, lambda);
}

template<typename T>
__device__ __forceinline__
T shfl_up(const T& var, int src_lane, int width = 32) {
    const auto& lambda = [](const int& value, int src_lane, int width) {
                            return __shfl_up(value, src_lane, width);
                        };
    return detail::generic_shfl(var, src_lane, width, lambda);
}

template<typename T>
__device__ __forceinline__
T shfl_down(const T& var, int src_lane, int width = 32) {
    const auto& lambda = [](const int& value, int src_lane, int width) {
                            return __shfl_down(value, src_lane, width);
                        };
    return detail::generic_shfl(var, src_lane, width, lambda);
}


/** @fn void swap(T& A, T& B)
 *  @brief swap A and B
 */
template<typename T>
__device__ __forceinline__ void swap(T& A, T& B);

template<int BlockSize, THREAD_GROUP GRP>
__device__ __forceinline__ void syncthreads();

template<bool CONDITION>
__device__ __forceinline__ void syncthreads();

} // namespace xlib

#include "impl/Basic.i.cuh"
