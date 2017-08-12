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
 */
#include "Device/PTX.cuh"

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
