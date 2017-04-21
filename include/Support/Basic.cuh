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

//#include "Base/Device/Util/CacheModifier.cuh"
#include "Support/CudaUtil.cuh"

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

template<typename T>
__device__ __forceinline__
T shfl(const T& var, int src_lane, int width = 32) {
    const int NUM = sizeof(T) / sizeof(int);
    static_assert(sizeof(T) % sizeof(int) == 0, "T must be multiple of 4");

    int tmp[NUM];
    reinterpret_cast<T&>(tmp) = var;
    #pragma unroll
    for (int i = 0; i < NUM; i++)
        tmp[i] = __shfl(tmp[i], src_lane, width);
    return reinterpret_cast<T&>(tmp);
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
