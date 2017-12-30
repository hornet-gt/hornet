/**
 * @internal
 * @author Federico Busato                                                  <br>
 *         Univerity of Verona, Dept. of Computer Science                   <br>
 *         federico.busato@univr.it
 * @date April, 2017
 * @version v1.3
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
#pragma once

#include "Host/Algorithm.hpp"                    //xlib::upper_bound_left
#include "Device/DataMovement/RegReordering.cuh" //xlib::shuffle_reordering
#include "Device/Util/Basic.cuh"                 //xlib::sync
#include "Device/Util/DeviceProperties.cuh"      //xlib::WARP_SIZE

namespace xlib {

template<typename T, typename R>
__device__ __forceinline__
R lower_bound2(const T* mem, R size, T searched) {
    R start = 0, end = size;
    while (start < end) {
        int mid = (start + end) / 2u;
        T tmp0  = mem[mid];
        T tmp1  = mem[mid + 1];
        if (searched >= tmp0 && searched < tmp1)
            return mid;
        if (searched < tmp0)
            end = mid;
        else
            start = mid + 1;
    }
}

template<unsigned PARTITION_SIZE, typename T>
__global__
void blockPartition(const T* __restrict__ d_prefixsum,
                    int                   prefixsum_size,
                    int*     __restrict__ d_partitions,
                    int                   num_partitions) {

    int id     = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = id; i < num_partitions; i += stride) {
    	T searched      = static_cast<T>(i) * PARTITION_SIZE;
		d_partitions[i] = xlib::upper_bound_left(d_prefixsum, prefixsum_size,
                                                 searched);
    }
    if (id == 0)
        d_partitions[num_partitions] = prefixsum_size - 2;
}

//==============================================================================
//==============================================================================

template<unsigned BLOCK_SIZE, bool LAST_BLOCK,
         unsigned ITEMS_PER_THREAD, typename T>
__device__ __forceinline__
void threadPartition(const T* __restrict__ d_prefixsum,
                     int                   prefixsum_size,
                     int                   block_start_pos,
                     int                   block_end_pos,
                     int                   block_search_low,
                     T*       __restrict__ smem_prefix,
                     int                 (&reg_pos)[ITEMS_PER_THREAD],
                     T                   (&reg_offset)[ITEMS_PER_THREAD]) {

    T   searched  = block_search_low +
                    static_cast<T>(threadIdx.x) * ITEMS_PER_THREAD;
    int smem_size = block_end_pos - block_start_pos + 2;

    // seems faster in CUDA 9
    d_prefixsum += block_start_pos;
    for (int i = threadIdx.x; i < smem_size; i += BLOCK_SIZE)
        smem_prefix[i] = d_prefixsum[i];
    xlib::sync<BLOCK_SIZE>();

    int smem_pos = xlib::upper_bound_left(smem_prefix, smem_size, searched);
    T   next     = smem_prefix[smem_pos + 1];
    T   offset   = searched - smem_prefix[smem_pos];
    T   limit    = smem_prefix[smem_size - 1];

    const int LOWEST = xlib::numeric_limits<int>::lowest;

    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        reg_pos[i]    = (!LAST_BLOCK || searched < limit) ? smem_pos : LOWEST;
        reg_offset[i] = offset;
        searched++;
        bool pred = (searched == next);
        offset    = (pred) ? 0 : offset + 1;
        smem_pos  = (pred) ? smem_pos + 1 : smem_pos;
        next      = smem_prefix[smem_pos + 1];
    }
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        reg_pos[i] += block_start_pos;
        assert(reg_pos[i] < prefixsum_size);
    }
    xlib::sync<BLOCK_SIZE>();
}

//==============================================================================
//==============================================================================

template<unsigned BLOCK_SIZE, unsigned ITEMS_PER_THREAD, bool LAST_BLOCK = true,
         typename T, typename Lambda>
__device__ __forceinline__
void binarySearchLB2(const int* __restrict__ d_partitions,
                     int                     num_partitions,
                     const T*   __restrict__ d_prefixsum,
                     int                     prefixsum_size,
                     void*      __restrict__ smem,
                     const Lambda&           lambda) {

    const unsigned ITEMS_PER_WARP  = xlib::WARP_SIZE * ITEMS_PER_THREAD;
    const unsigned ITEMS_PER_BLOCK = BLOCK_SIZE * ITEMS_PER_THREAD;

    int reg_pos   [ITEMS_PER_THREAD];
    T   reg_offset[ITEMS_PER_THREAD];

    int  block_start_pos  = d_partitions[ blockIdx.x ];
    int  block_end_pos    = d_partitions[ blockIdx.x + 1 ];
    int  block_search_low = blockIdx.x * ITEMS_PER_BLOCK;
    auto smem_prefix      = static_cast<T*>(smem);

    threadPartition<BLOCK_SIZE, LAST_BLOCK>
        (d_prefixsum, prefixsum_size, block_start_pos, block_end_pos,
         block_search_low, smem_prefix, reg_pos, reg_offset);

    xlib::smem_reordering<>(reg_pos, smem_prefix);
    //xlib::shuffle_reordering<>(reg_pos, smem_prefix);

    int id    = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int index = (id / xlib::WARP_SIZE) * ITEMS_PER_WARP + xlib::lane_id();

    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        if (!LAST_BLOCK || reg_pos[i] >= 0) {
            assert(reg_pos[i] < prefixsum_size);
            lambda(reg_pos[i], reg_offset[i], index + i * xlib::WARP_SIZE);
        }
    }
}

} // namespace xlib
