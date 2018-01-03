/**
 * @internal
 * @author Federico Busato                                                  <br>
 *         Univerity of Verona, Dept. of Computer Science                   <br>
 *         federico.busato@univr.it
 * @date January, 2018
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

namespace xlib {

template<unsigned PARTITION_SIZE, typename T>
__global__
void blockPartition2(const T* __restrict__ d_prefixsum,
                     int                   prefixsum_size,
                     T                     max_value,
                     int                   num_merge,
                     int*     __restrict__ d_partitions,
                     int                   num_partitions) {

    int id     = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    NaturalIterator natural_iterator;

    for (int i = id; i <= num_partitions; i += stride) {
    	T diagonal      = ::min(i * PARTITION_SIZE, num_merge);
        auto value      = xlib::merge_path_search(d_prefixsum, prefixsum_size,
                                                  natural_iterator, max_value,
                                                  diagonal);
        d_partitions[i] = value.y;
        if (i < 10)
            printf("%d\t%d\t%d\t%d\n", value.x, value.y, value.x + value.y, d_prefixsum[value.x]);
    }
    //if (id == 0)
    //    d_partitions[num_partitions] = max_value;
}


template<unsigned BLOCK_SIZE, bool LAST_BLOCK,
         unsigned ITEMS_PER_THREAD, typename T>
__device__ __forceinline__
void threadPartition2(const T* __restrict__ d_prefixsum,
                     int                   prefixsum_size,
                     int2                  block_coord_start,
                     int2                  block_coord_end,
                     int                   block_search_low,
                     T*       __restrict__ smem_prefix,
                     int                 (&reg_pos)[ITEMS_PER_THREAD],
                     T                   (&reg_offset)[ITEMS_PER_THREAD]) {

    NaturalIterator natural_iterator;
    T   diagonal  = block_search_low +
                    static_cast<T>(threadIdx.x) * ITEMS_PER_THREAD;
    int smem_size = block_coord_end.x - block_coord_start.x + 2;

    auto smem_tmp = smem_prefix + threadIdx.x;
    auto d_tmp    = d_prefixsum + block_coord_start.x + threadIdx.x;

    for (int i = threadIdx.x; i < smem_size; i += BLOCK_SIZE) {
        *smem_tmp = *d_tmp;
        smem_tmp += BLOCK_SIZE;
        d_tmp    += BLOCK_SIZE;
    }

    xlib::sync<BLOCK_SIZE>();

    int max_value = block_coord_end.y - block_coord_start.y;

    auto thread_coord = xlib::merge_path_search(smem_prefix, smem_size,
                                                natural_iterator, max_value,
                                                diagonal);

    const int LOWEST = xlib::numeric_limits<int>::lowest;
    int y_value = thread_coord.y;

    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        if (y_value < smem_prefix[thread_coord.x]) {
            reg_pos[i]    = thread_coord.x;
            reg_offset[i] = smem_prefix[thread_coord.x] - y_value;
            y_value++;
        }
        else {
            reg_pos[i] = LOWEST;
            thread_coord.x++;
        }
    }
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        reg_pos[i] += block_coord_start.x;
        assert(reg_pos[i] < prefixsum_size);
    }
    xlib::sync<BLOCK_SIZE>();
}

template<unsigned BLOCK_SIZE, unsigned ITEMS_PER_THREAD, bool LAST_BLOCK = true,
         typename T, typename Lambda>
__device__ __forceinline__
void mergePathLB(const int* __restrict__ d_partitions,
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
    auto smem_prefix      = static_cast<T*>(smem);

    int a_count = 0, b_count = 0;
    int diag0 = blockIdx.x * ITEMS_PER_BLOCK;
    int diag1 = ::min(a_count + b_count, diag0 + ITEMS_PER_BLOCK);

    int2 block_coord_start { block_start_pos, diag0 - block_start_pos };
    int2 block_coord_end   { block_end_pos,   diag1 - block_end_pos   };

    /*threadPartition2<BLOCK_SIZE, LAST_BLOCK>
        (d_prefixsum, prefixsum_size, block_start_pos, block_end_pos,
         block_search_low, smem_prefix, reg_pos, reg_offset);*/

    xlib::smem_reordering(reg_pos, smem_prefix);

    int id    = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int index = (id / xlib::WARP_SIZE) * ITEMS_PER_WARP + xlib::lane_id();

    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        if (reg_pos[i] >= 0) {
            assert(reg_pos[i] < prefixsum_size);
            lambda(reg_pos[i], reg_offset[i], index + i * xlib::WARP_SIZE);
        }
    }
}

} // namespace xlib
