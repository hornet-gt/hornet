/**
 * @author Federico Busato                                                  <br>
 *         Univerity of Verona, Dept. of Computer Science                   <br>
 *         federico.busato@univr.it
 * @date April, 2017
 * @version v1.3
 *
 * @copyright Copyright © 2017 by Nicola Bombieri
 *
 * @license{<blockquote>
 * XLib is provided under the terms of The MIT License (MIT)                <br>
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 *  all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 * </blockquote>}
 *
 * @file
 */
#pragma once

#include "Support/Definition.cuh"
#include "Support/Indexing.cuh"
#include "Support/Basic.cuh"


namespace xlib {
namespace detail {

template<unsigned BLOCK_SIZE, typename T, unsigned ITEMS_PER_THREAD>
__device__ __forceinline__
void threadPartitionAuxLoop(const T* __restrict__ ptr,
                            int block_start_pos, int chunk_size, T searched,
                            T* __restrict__ smem_prefix,
                            int (&reg_pos)[ITEMS_PER_THREAD],
                            T   (&reg_offset)[ITEMS_PER_THREAD]) {

    const unsigned ITEMS_PER_BLOCK = BLOCK_SIZE * ITEMS_PER_THREAD;
    T low_limit = 0;
    while (chunk_size > 0) {
        int smem_size = ::min(chunk_size, ITEMS_PER_BLOCK);

        #pragma unroll
        for (int i = 0; i < ITEMS_PER_THREAD; i++) {
            int index = i * BLOCK_SIZE + threadIdx.x;
            if (index < smem_size)
                smem_prefix[index] = ptr[index];
        }
        __syncthreads();

        int   ubound = xlib::upper_bound_left(smem_prefix, smem_size, searched);
        int smem_pos = ::min(::max(0, ubound), ITEMS_PER_BLOCK - 2);
        assert(smem_pos >= 0 && smem_pos + 1 < ITEMS_PER_BLOCK);
        T     offset = ::max(searched - smem_prefix[smem_pos], 0);
        T       next = smem_prefix[smem_pos + 1];
        T high_limit = smem_prefix[smem_size - 1];

        #pragma unroll
        for (int i = 0; i < ITEMS_PER_THREAD; i++) {
            T loc_search = searched + i;
            if (loc_search < low_limit || loc_search >= high_limit)
                continue;
            if (loc_search == next) {
                do {
                    smem_pos++;
                    assert(smem_pos >= 0 && smem_pos + 1 < smem_size);
                    next = smem_prefix[smem_pos + 1];
                } while (loc_search == next);
                offset = 0;
            }
            reg_pos[i]    = block_start_pos + smem_pos;
            reg_offset[i] = offset;
            offset++;
        }
        __syncthreads();
        low_limit        = high_limit;
        chunk_size      -= ITEMS_PER_BLOCK - 1;
        ptr             += ITEMS_PER_BLOCK - 1;
        block_start_pos += ITEMS_PER_BLOCK - 1;
    }
}

/**
 * @brief
 * @details
 * @verbatim
 *    d_prefixsum input: 0, 3, 7, 10, 13
 *     ITEMS_PER_THREAD: 5
 *    reg_pos  output: t1(0, 0, 0, 1, 1) t2(1, 1, 2, 2, 2) t3(3, 3, 3, *, *)
 * reg_offset  output: t1(0, 1, 2, 0, 1) t2(2, 3, 0, 1, 2) t3(0, 1, 2, *, *)
 *                    *: undefined
 * @endverbatim
 *
 * @tparam BLOCK_SIZE
 * @tparam T
 * @tparam ITEMS_PER_THREAD
 * @param[in] d_partition
 * @param[in] d_prefixsum
 * @param[in] reg_pos
 * @param[in] reg_offset
 * @param[in] smem
 *
 * @attention |smem| must be equal to BLOCK_SIZE * ITEMS_PER_THREAD
 * @attention The best way to detect unused registers in the last thread block
 *            is to fill the `reg_offset` array with a special value
 * @attention requires `__syncthreads()` at the end if the shared memory is used
 * @remark the function uses static indexing for `reg_pos` and `reg_offset`
 */
template<unsigned BLOCK_SIZE, unsigned ITEMS_PER_THREAD, typename T>
__device__ __forceinline__
void threadPartition(const T* __restrict__ d_prefixsum, int psize,
                     void*    __restrict__ smem,
                     int (&reg_pos)[ITEMS_PER_THREAD],
                     T   (&reg_offset)[ITEMS_PER_THREAD]) {

    const unsigned ITEMS_PER_BLOCK = BLOCK_SIZE * ITEMS_PER_THREAD;
    auto smem_prefix = static_cast<T*>(smem);
    //--------------------------------------------------------------------------
    //BLOCK PARTITION
    const unsigned IDX1 = BLOCK_SIZE >= 64 ? 32 : 1;
    if (threadIdx.x == 0) {
        T   blk_search = static_cast<T>(blockIdx.x) * ITEMS_PER_BLOCK;
        smem_prefix[0] = xlib::upper_bound_left(d_prefixsum, psize, blk_search);
    }
    else if (threadIdx.x == IDX1) {
        T   blk_search = static_cast<T>(blockIdx.x + 1) * ITEMS_PER_BLOCK;
        smem_prefix[1] = blockIdx.x == gridDim.x - 1 ? psize - 2 :
                         xlib::upper_bound_left(d_prefixsum, psize, blk_search);
    }
    __syncthreads();
    int block_start_pos = smem_prefix[0];
    int   block_end_pos = smem_prefix[1];
    __syncthreads();
    //--------------------------------------------------------------------------
    //THREAD PARTITION
	T  block_search_low = static_cast<T>(blockIdx.x) * ITEMS_PER_BLOCK;
	T          searched = block_search_low +
                          static_cast<T>(threadIdx.x) * ITEMS_PER_THREAD;

	int      chunk_size = block_end_pos - block_start_pos + 2;
    const T*        ptr = d_prefixsum + block_start_pos;

    if (blockIdx.x == gridDim.x - 1)
        xlib::reg_fill(reg_pos, -1);

    detail::threadPartitionAuxLoop<BLOCK_SIZE>
        (ptr, block_start_pos, chunk_size, searched,
         smem_prefix, reg_pos, reg_offset);
}

} // namespace detail


template<unsigned BLOCK_SIZE, unsigned ITEMS_PER_THREAD = 0,
         typename T, typename Lambda>
__device__ __forceinline__
void binarySearchLB(const T* __restrict__ d_prefixsum, int psize,
                    void*    __restrict__ smem,
                    const Lambda& lambda) {
    const unsigned _ITEMS_PER_THREAD = ITEMS_PER_THREAD == 0 ?
                                           SMemPerThread<T, BLOCK_SIZE>::value :
                                           ITEMS_PER_THREAD;
    int reg_pos[_ITEMS_PER_THREAD];
    T   reg_offset[_ITEMS_PER_THREAD];

    detail::threadPartition<BLOCK_SIZE>(d_prefixsum, psize, smem,
                                        reg_pos, reg_offset);

    threadToWarpIndexing<_ITEMS_PER_THREAD>(reg_pos, reg_offset, smem);

    if (blockIdx.x == gridDim.x - 1) {
        #pragma unroll
        for (int i = 0; i < _ITEMS_PER_THREAD; i++) {
            if (reg_pos[i] != -1)
                lambda(reg_pos[i], reg_offset[i]);
        }
    }
    else {
        #pragma unroll
        for (int i = 0; i < _ITEMS_PER_THREAD; i++)
            lambda(reg_pos[i], reg_offset[i]);
    }
}

} // namespace xlib
