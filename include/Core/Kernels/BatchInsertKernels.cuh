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
#include "Support/Device/DeviceQueue.cuh"
#include "Support/Device/Definition.cuh"
#include "Support/Host/Algorithm.hpp"
#include "Support/Device/BinarySearchLB.cuh"

namespace custinger {

template<typename EqualOp>
__global__
void findDuplicateKernel(cuStingerDevice    custinger,
                         BatchUpdate        batch_update,
                         EqualOp            equal_op,
                         bool* __restrict__ d_flags) {
    int     id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = id; i < batch_update.size(); i += stride) {
        auto        src = batch_update.src(i);
        auto batch_edge = batch_update.dst(i);

        bool flag;
        if (i + 1 != batch_update.size() //&& src == batch_update.src(i)
                 && equal_op(batch_edge, batch_update.dst(i + 1))) {
            flag = false;
        }
        else {
            auto vertex_src = custinger.vertex(src);
            auto     degree = vertex_src.degree();
            flag = true;
            for (degree_t j = 0; j < degree; j++) {
                /*printf("i %d\t b (%d, %d)\t edge %d -> %d\n",
                        i, src, batch_edge.dst(), vertex_src.edge(j).dst(),
                       equal_op(batch_edge, vertex_src.edge(j)));*/
                if (equal_op(batch_edge, vertex_src.neighbor_id(j))) {
                    flag = false;
                    break;
                }
            }
        }
        d_flags[i] = flag;
        /*if (flag) {
            auto  degree_ptr = vertex_src.degree_ptr();
            degree_t old_pos = atomicAdd(degree_ptr, 1);
            vertex_src.store(batch_edge, old_pos);
        }*/
    }
}

//==============================================================================

struct __align__(16) UpdateStr {
    vid_t    src;
    degree_t old_degree;
    degree_t new_degree;

    __device__ __forceinline__
    UpdateStr() {}

    __device__ __forceinline__
    UpdateStr(vid_t src, degree_t old_degree, degree_t new_degree) :
        src(src), old_degree(old_degree), new_degree(new_degree) {}
};

std::ostream& operator<<(std::ostream& os, const UpdateStr& obj) {
    os << "(" << obj.src << ": " << obj.old_degree << "," << obj.new_degree
       << ")";
    return os;
}

//==============================================================================

__global__
void findSpaceRequest(VertexBasicData* __restrict__ d_vertex,
                      const vid_t*     __restrict__ d_unique_src,
                      const int*       __restrict__ d_counts,
                      int                           unique_size,
                      UpdateStr*       __restrict__ d_queue,
                      int*             __restrict__ d_queue_size,
                      degree_t*        __restrict__ d_degrees_changed) {

    int     id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    const int SIZE = 16;
    UpdateStr array[SIZE];
    xlib::DeviceQueue<UpdateStr, SIZE> queue(array, d_queue, d_queue_size);
    int size = xlib::upper_approx<xlib::WARP_SIZE>(unique_size);

    for (int i = id; i < size; i += stride) {
        if (i < unique_size) {
            vid_t       src = d_unique_src[i];
            auto    request = d_counts[i];
            auto      vnode = d_vertex[src];
            auto new_degree = request + vnode.degree;
            //printf("src: %d\td: %d\treq: %d\tp:%d \n",
            //       src, vnode.degree, request, new_degree > vnode.limit());
            if (new_degree >= vnode.limit())
                queue.insert(UpdateStr(src, vnode.degree, new_degree));
            else
                d_vertex[src] = VertexBasicData(new_degree, vnode.neighbor_ptr);
            //if (...)
                d_degrees_changed[i] = vnode.degree;
        }
        queue.store();
    }
}

__global__
void collectInfoForHost(const VertexBasicData* __restrict__ d_vertex,
                        const UpdateStr*       __restrict__ d_queue,
                        int                                 queue_size,
                        degree_t*              __restrict__ d_old_degrees,
                        edge_t**               __restrict__ d_old_ptrs) {

    int     id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = id; i < queue_size; i += stride) {
        UpdateStr    str = d_queue[i];
        auto       vnode = d_vertex[str.src];
        d_old_degrees[i] = vnode.degree;
        d_old_ptrs[i]    = reinterpret_cast<edge_t*>(vnode.neighbor_ptr);
        //if (vnode.neighbor_ptr != nullptr)
        //printf("%d  ptr %llX  %d\n", str.src, d_old_ptrs[i],
        //        reinterpret_cast<vid_t*>(vnode.neighbor_ptr)[0]);
    }
}

__global__
void updateVertexBasicData(VertexBasicData* __restrict__ d_vertex,
                           const UpdateStr* __restrict__ d_queue,
                           int                           queue_size,
                           edge_t* const*   __restrict__ d_new_ptrs) {

    int     id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = id; i < queue_size; i += stride) {
        UpdateStr     str = d_queue[i];
        d_vertex[str.src] = VertexBasicData(str.new_degree,
                                      reinterpret_cast<byte_t*>(d_new_ptrs[i]));
    }
}

template<unsigned BLOCK_SIZE, unsigned ITEMS_PER_BLOCK>
__global__
void bulkCopyAdjLists(const degree_t* __restrict__ d_prefixsum,
                      int                          work_size,
                      edge_t* const*  __restrict__ d_old_ptrs,
                      edge_t* const*  __restrict__ d_new_ptrs) {
    __shared__ degree_t smem[ITEMS_PER_BLOCK];
    auto lambda = [&] (int pos, degree_t offset) {
                    auto    old_ptr = reinterpret_cast<vid_t*>(d_old_ptrs[pos]);
                    auto    new_ptr = reinterpret_cast<vid_t*>(d_new_ptrs[pos]);
                    new_ptr[offset] = old_ptr[offset];
                    //printf("p: %d    %d \t\t%llX\n", pos, old_ptr[offset], old_ptr);
                };
    xlib::binarySearchLB<BLOCK_SIZE>(d_prefixsum, work_size, smem, lambda);
}

__global__
void mergeAdjListKernel(cuStingerDevice              custinger,
                        const degree_t* __restrict__ d_degrees_changed,
                        BatchUpdate                  batch_update,
                        const vid_t* __restrict__    d_unique_src,
                        const int*   __restrict__    d_counts_ps,
                        int                          num_uniques) {
    int     id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = id; i < num_uniques; i += stride) {
        auto    vertex = custinger.vertex(d_unique_src[i]);
        auto  left_ptr = vertex.neighbor_ptr();
        auto left_size = d_degrees_changed[i];

        int      start = d_counts_ps[i];
        int        end = d_counts_ps[i + 1];
        int right_size = end - start;
        auto right_ptr = batch_update.dst_ptr() + start;

        xlib::inplace_merge(left_ptr, left_size, right_ptr, right_size);
    }
}

} // namespace custinger
