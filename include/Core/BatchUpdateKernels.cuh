/**
 * @author Federico Busato                                                  <br>
 *         Univerity of Verona, Dept. of Computer Science                   <br>
 *         federico.busato@univr.it
 * @date April, 2017
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
 */
namespace custinger {

struct __align__(16) UpdateStr {
    vid_t     src;
    degree_t old_degree;
    degree_t new_degree;

    __device__ __forceinline__
    UpdateStr(id_t src, degree_t old_degree, degree_t new_degree) :
        src(src), old_degree(old_degree), new_degree(new_degree) {}
};

std::ostream& operator<<(std::ostream& os, const UpdateStr& obj) {
    os << "(" << obj.src << ": " << obj.old_degree << "," << obj.new_degree
       << ")";
    return os;
}

//==============================================================================

__global__
void findSpaceRequest(const VertexBasicData* __restrict__ d_nodes,
                      const id_t*            __restrict__ d_unique_src,
                      const int*             __restrict__ d_counts,
                      int                                 batch_size,
                      UpdateStr*             __restrict__ d_queue,
                      int*                   __restrict__ d_queue_size) {

    int     id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    const int SIZE = 16;
    UpdateStr array[SIZE];
    xlib::WarpQueueSimple<UpdateStr, SIZE> queue(array, d_queue, d_queue_size);
    int size = xlib::upper_approx<xlib::WARP_SIZE>(batch_size);

    for (int i = id; i < size; i += stride) {
        if (i < batch_size) {
            vid_t       src = d_unique_src[i];
            auto    request = d_counts[i];
            auto       node = d_nodes[src];
            auto new_degree = request + node.degree;
            //printf("src: %lld\td: %d\treq %d\tp:%d \n",
            //    src, node.degree, request,new_degree > node.limit);
            if (new_degree > node.limit)
                queue.insert(UpdateStr(src, node.degree, new_degree));
        }
        queue.store();
    }
}

__global__
void CollectInfoForHost(const VertexBasicData* __restrict__ d_nodes,
                        const UpdateStr*       __restrict__ d_queue,
                        int                                 queue_size,
                        degree_t*              __restrict__ d_old_degrees,
                        edge_t**               __restrict__ d_old_ptrs) {

    int     id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = id; i < queue_size; i += stride) {
        UpdateStr    str = d_queue[i];
        auto        node = d_nodes[str.src];
        d_old_degrees[i] = node.degree;
        d_old_ptrs[i]    = node.edge_ptr;
    }
}

__global__
void updateVertexBasicData(const VertexBasicData* __restrict__ d_nodes,
                           const UpdateStr*       __restrict__ d_queue,
                           int                                 queue_size,
                           const edge_t* const*   __restrict__ d_new_ptrs) {

    int     id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = id; i < queue_size; i += stride) {
        UpdateStr    str = d_queue[i];
        d_nodes[str.src] = VertexBasicData(d_new_ptrs[i], str.new_degree);
    }
}

template<unsigned BLOCK_SIZE, unsigned ITEMS_PER_BLOCK>
__global__
void bulkCopyAdjLists(const int*     __restrict__ d_prefixsum,
                      edge_t* const* __restrict__ d_old_ptr,
                      edge_t* const* __restrict__ d_new_ptrs) {
    __shared__ degree_t smem[ITEMS_PER_BLOCK];
    auto lambda = [] (int pos, degree_t offset) {
                        auto    old_ptr = d_old_ptr[ pos ];
                        auto    new_ptr = d_new_ptrs[ pos ];
                        new_ptr[offset] = old_ptr[ offset ];
                    };
    xlib::binarySearchLBs<BLOCK_SIZE>(d_work, work_size, smem, lambda);
}

    /*const int SIZE = xlib::SMemPerThread<int>::value;
    __shared__ int smem[SIZE * BLOCK_SIZE];
    int queue_pos[SIZE];
    int queue_offset[SIZE];
    xlib::reg_fill(queue_offset, -1);

    xlib::threadPartition<BLOCK_SIZE>(d_partition, d_prefixsum,
                                      queue_pos, queue_offset, smem);
    __syncthreads();

    xlib::threadToWarpIndexing(queue_pos, queue_offset, smem);

    #pragma unroll
    for (int i = 0; i < SIZE; i++) {
        if (queue_offset[i] != -1) {
            auto      old_ptr = d_old_ptr[ queue_pos[i] ];
            auto      new_ptr = d_new_ptrs[ queue_pos[i] ];
            int        offset = queue_offset[i];
            new_ptr[ offset ] = old_ptr[ offset ];
            //printf("%d\t%d\t%d\t%lld\n", threadIdx.x, queue_pos[i], queue_offset[i]);
        }
    }*/

} // namespace custinger
