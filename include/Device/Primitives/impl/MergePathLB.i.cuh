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
 */
namespace xlib {

template<unsigned ITEMS_PER_BLOCK, typename T>
__global__
void mergePathLBPartition(const T* __restrict__ d_prefixsum,
                          int                   prefixsum_size,
                          T                     max_value,
                          int                   num_merge,
                          int*     __restrict__ d_partitions,
                          int                   num_partitions) {

    int id     = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = id; i <= num_partitions; i += stride) {
    	T diagonal      = ::min(i * ITEMS_PER_BLOCK, num_merge);
        auto value      = xlib::merge_path_search(d_prefixsum, prefixsum_size,
                                                  NaturalIterator(), max_value,
                                                  diagonal);
        d_partitions[i] = value.y;
        //if (i < 10)
        //    printf("%d\t%d\t\t%d\n", value.x, value.y, d_prefixsum[value.x]);
    }
}

//==============================================================================

template<unsigned BLOCK_SIZE, bool INDICES,
         unsigned ITEMS_PER_THREAD, typename T>
__device__ __forceinline__
void blockMergePathLB(const T* __restrict__ d_prefixsum,
                      int2                  block_coord_start,
                      int                   y_size,
                      T*       __restrict__ smem_prefix,
                      int                   smem_size,
                      T*       __restrict__ smem_buffer) {


    auto smem_tmp = smem_prefix + threadIdx.x;
    auto d_tmp    = d_prefixsum + block_coord_start.x + threadIdx.x;

    for (int i = threadIdx.x; i < smem_size; i += BLOCK_SIZE) {
        *smem_tmp = *d_tmp;
        smem_tmp += BLOCK_SIZE;
        d_tmp    += BLOCK_SIZE;
    }

    xlib::sync<BLOCK_SIZE>();

    T diagonal = threadIdx.x * ITEMS_PER_THREAD;
    NaturalIterator natural_iterator(block_coord_start.y);
    auto thread_coord = xlib::merge_path_search(smem_prefix, smem_size,
                                                natural_iterator, y_size,
                                                diagonal);

    const auto MAX = xlib::numeric_limits<int>::max;
    int first      = (threadIdx.x == 0);
    thread_coord.x = max(thread_coord.x - first, 0);
    int next       = (thread_coord.x < smem_size) ? smem_prefix[thread_coord.x]
                                                  : MAX;
    int y_value    = block_coord_start.y + thread_coord.y;

    /*#pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        assert(y_value <= next);
        //if (blockIdx.x == 2606 && threadIdx.x == 63)
        //    printf("(%d, %d, %d)\n", thread_coord.x, y_value, next);
        if (y_value < next) {
            assert(thread_coord.y < y_size || blockIdx.x == gridDim.x - 1);
            assert(thread_coord.x >= 0);
            assert(thread_coord.x - 1 < smem_size);
            smem_buffer[thread_coord.y] = thread_coord.x - 1;
            thread_coord.y++;
            y_value++;
        }
        else {
            thread_coord.x++;
            next = (thread_coord.x < smem_size) ?
                    smem_prefix[thread_coord.x] : MAX;
        }
    }*/
    /*if (blockIdx.x == 0 && threadIdx.x == 0) {
        //printf("%d\t%d\n", smem_size, y_size);
        xlib::printfArray(smem_prefix, smem_size);
        printf("\n");
        xlib::printfArray(smem_buffer, y_size);
    }*/

    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        bool pred = (y_value < next);
        if (pred)
            smem_buffer[thread_coord.y] = thread_coord.x - 1;

        thread_coord.x = (pred) ? thread_coord.x     : thread_coord.x + 1;
        y_value        = (pred) ? y_value + 1        : y_value;
        thread_coord.y = (pred) ? thread_coord.y + 1 : thread_coord.y;
        next           = (thread_coord.x < smem_size) ?
                         smem_prefix[thread_coord.x] : MAX;
    }
    xlib::sync<BLOCK_SIZE>();
}

//==============================================================================

template<unsigned BLOCK_SIZE, unsigned ITEMS_PER_THREAD, bool INDICES = true,
         typename T, typename Lambda>
__device__ __forceinline__
void mergePathLB(const int* __restrict__ d_partitions,
                 int                     num_partitions,
                 const T*   __restrict__ d_prefixsum,
                 int                     prefixsum_size,
                 void*      __restrict__ smem,
                 const Lambda&           lambda) {

    const unsigned ITEMS_PER_BLOCK = BLOCK_SIZE * ITEMS_PER_THREAD;

    int block_diag0 = blockIdx.x * ITEMS_PER_BLOCK;
    int block_diag1 = block_diag0 + ITEMS_PER_BLOCK;

    int  block_start_pos  = d_partitions[ blockIdx.x ];
    int  block_end_pos    = d_partitions[ blockIdx.x + 1 ];
    int  min_value        = ::min(block_diag1 - block_end_pos,
                                  prefixsum_size);
    int2 block_coord_start { block_diag0 - block_start_pos, block_start_pos };
    int2 block_coord_end   { min_value,                     block_end_pos   };

    int  smem_size   = block_coord_end.x - block_coord_start.x;
    int  y_size      = block_coord_end.y - block_coord_start.y;
    auto smem_prefix = static_cast<T*>(smem);
    auto smem_buffer = static_cast<T*>(smem) + smem_size;

    assert(blockIdx.x == gridDim.x - 1 ||
           (block_coord_end.x - block_coord_start.x) +
           (block_coord_end.y - block_coord_start.y) == ITEMS_PER_BLOCK);
    assert(block_coord_start.x + smem_size <= prefixsum_size);

    /*if (blockIdx.x < 10 && threadIdx.x == 0)
        printf("%d\t%d\t%d\t%d\n", block_coord_start.x, block_coord_end.x,
                                   block_coord_start.y, block_coord_end.y);*/

    blockMergePathLB<BLOCK_SIZE, INDICES, ITEMS_PER_THREAD>
        (d_prefixsum, block_coord_start, y_size,
         smem_prefix, smem_size, smem_buffer);

    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        int index = threadIdx.x + i * BLOCK_SIZE;
        if (index < y_size) {
            int reg_pos    = smem_buffer[index];
            int reg_index  = block_coord_start.y + index;
            int reg_offset = reg_index - smem_prefix[reg_pos];
            lambda(reg_pos + block_coord_start.x, reg_offset, reg_index);
        }
    }
}

} // namespace xlib
