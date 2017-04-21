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
#include "Support/PTX.cuh"

namespace xlib {

template<int SIZE, typename T>
 __device__ __forceinline__ void reg_fill(T (&reg)[SIZE], T value) {
    #pragma unroll
    for (int i = 0; i < SIZE; i++)
        reg[i] = value;
}

template<int SIZE, typename T>
 __device__ __forceinline__
 void reg_copy(const T (&reg1)[SIZE], T (&reg2)[SIZE]) {
    #pragma unroll
    for (int i = 0; i < SIZE; i++)
        reg2[i] = reg1[i];
}

__device__ __forceinline__ unsigned warp_id() {
    return threadIdx.x / xlib::WARP_SIZE;
}


template<unsigned BLOCK_SIZE, unsigned VIRTUAL_WARP>
__device__ __forceinline__ unsigned global_id() {
    return (blockIdx.x * BLOCK_SIZE + threadIdx.x) / VIRTUAL_WARP;
}

template<unsigned BLOCK_SIZE, unsigned VIRTUAL_WARP>
__device__ __forceinline__ unsigned global_stride() {
    return (gridDim.x * BLOCK_SIZE) / VIRTUAL_WARP;
}

template<unsigned WARP_SZ>
__device__ __forceinline__ unsigned warp_base() {
    return threadIdx.x & ~(WARP_SZ - 1u);
}

template<typename T>
__device__ __forceinline__ T warp_broadcast(T value, int predicate) {
    const unsigned electedLane = xlib::__msb(__ballot(predicate));
    return __shfl(value, electedLane);
}

template<typename T>
__device__ __forceinline__ void swap(T*& A, T*& B) {
    T* tmp = A;
    A = B;
    B = tmp;
}

template<int BlockSize, THREAD_GROUP GRP>
__device__ __forceinline__ void syncthreads() {
    if (BlockSize != 32 && GRP != WARP)
        __syncthreads();
}

template<bool CONDITION>
__device__ __forceinline__ void syncthreads() {
    if (CONDITION)
        __syncthreads();
}

template<unsigned VW_SIZE>
__device__ __forceinline__
VWarp<VW_SIZE>::VWarp() : _mask(VWarp<VW_SIZE>::mask()) {}

template<unsigned VW_SIZE>
__device__ __forceinline__
bool VWarp<VW_SIZE>::any(bool pred) const {
    return VW_SIZE == 1 ? pred :
          (VW_SIZE == 32 ? __any(pred) :  __ballot(pred) & _mask);
}

template<unsigned VW_SIZE>
__device__ __forceinline__
bool VWarp<VW_SIZE>::all(bool pred) const {
    return VW_SIZE == 1 ? pred :
           (VW_SIZE == 32 ? __all(pred) : (__ballot(pred) & _mask) == _mask);
}

template<unsigned VW_SIZE>
__device__ __forceinline__
unsigned VWarp<VW_SIZE>::ballot(bool pred) const {
    return VW_SIZE == 32 ? __ballot(pred) : __ballot(pred) & _mask;
}

template<unsigned VW_SIZE>
__device__ __forceinline__
unsigned VWarp<VW_SIZE>::mask() {
    return (0xFFFFFFFF >> (32 - VW_SIZE)) << (lane_id() & ~(VW_SIZE - 1));
}

} // namespace xlib
