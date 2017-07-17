/**
 * @author Federico Busato                                                  <br>
 *         Univerity of Verona, Dept. of Computer Science                   <br>
 *         federico.busato@univr.it
 * @date July, 2017
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
#include "Support/Host/Algorithm.hpp"          //xlib::binary_search
#include "Support/Device/BinarySearchLB.cuh"   //xlib::binarySearchLB

namespace custinger {

__global__
void markUniqueKernel(const vid_t* __restrict__ d_batch_src,
                      const vid_t* __restrict__ d_batch_dst,
                      int                       batch_size,
                      bool*        __restrict__ d_flags) {
    int     id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = id; i < batch_size - 1; i += stride) {
        d_flags[i] = d_batch_src[i] != d_batch_src[i + 1] ||
                     d_batch_dst[i] != d_batch_dst[i + 1];
    }
    if (id == 0)
        d_flags[batch_size - 1] = true;
}

__global__
void collectOldDegreeKernel(cuStingerDevice           custinger,
                            const vid_t* __restrict__ d_unique,
                            int                       num_uniques,
                            degree_t*    __restrict__ d_degree_old,
                            eoff_t*      __restrict__ d_inverse_pos) {

    int     id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = id; i < num_uniques; i += stride) {
        auto        src = d_unique[i];
        d_degree_old[i] = Vertex(custinger, src).degree();
        d_inverse_pos[src] = i;
    }
}

__global__
void deleteEdgesKernel(cuStingerDevice              custinger,
                       BatchUpdate                  batch_update,
                       const degree_t* __restrict__ d_degree_old_prefix,
                       const eoff_t*   __restrict__ d_inverse_pos,
                       bool*           __restrict__ d_flags) {
    int     id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = id; i < batch_update.size(); i += stride) {
        vid_t src = batch_update.src(i);
        vid_t dst = batch_update.dst(i);

        Vertex src_vertex(custinger, src);
        auto adj_ptr = src_vertex.neighbor_ptr();

        auto pos = xlib::binary_search(adj_ptr, src_vertex.degree(), dst);
        int inverse_pos = d_inverse_pos[src];
        d_flags[d_degree_old_prefix[inverse_pos] + pos] = false;
        //printf("del %d %d \t%d\t%d\t%d\n",
        //    src, dst, d_degree_old_prefix[inverse_pos], pos,
        //    d_degree_old_prefix[inverse_pos] + pos);
    }
}

//collect d_ptrs_array, d_degree_new and update custinger degree
__global__
void collectDataKernel(cuStingerDevice           custinger,
                       const vid_t* __restrict__ d_unique,
                       degree_t*    __restrict__ d_count,
                       int                       num_uniques,
                       degree_t*    __restrict__ d_degree_new,
                       byte_t**     __restrict__ d_ptrs_array) {
    int     id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = id; i < num_uniques; i += stride) {
        auto d_vertex_ptrs = reinterpret_cast<VertexBasicData*>
                                (custinger.d_vertex_ptrs[0]);
        auto         src = d_unique[i];
        auto vertex_data = d_vertex_ptrs[src];
        auto  new_degree = vertex_data.degree - d_count[i];

        d_ptrs_array[i] = vertex_data.neighbor_ptr;
        d_degree_new[i] = new_degree;

        d_vertex_ptrs[src] = VertexBasicData(new_degree,
                                             vertex_data.neighbor_ptr);
    }
}

template<unsigned BLOCK_SIZE, unsigned ITEMS_PER_BLOCK>
__global__
void moveDataKernel1(const degree_t* __restrict__ d_degree_old_prefix,
                     int                          num_items,
                     byte_t**        __restrict__ d_ptrs_array,
                     edge_t*         __restrict__ d_tmp) {
    __shared__ degree_t smem[ITEMS_PER_BLOCK];

        auto lambda = [&] (int pos, degree_t offset) {
            int  tmp_offset = d_degree_old_prefix[pos] + offset;
            auto ptr_vertex = reinterpret_cast<vid_t*>(d_ptrs_array[pos]);
            auto adj_vertex = reinterpret_cast<vid_t*>(ptr_vertex)[offset];

            if (NUM_EXTRA_ETYPES == 1) {
                auto ptr_weight = ptr_vertex + EDGES_PER_BLOCKARRAY;
                auto adj_weight = reinterpret_cast<int*>(ptr_weight)[offset];
                reinterpret_cast<int2*>(d_tmp)[tmp_offset] =
                                             make_int2(adj_vertex, adj_weight);
            } else
                reinterpret_cast<vid_t*>(d_tmp)[tmp_offset] = adj_vertex;
        };
    xlib::binarySearchLB<BLOCK_SIZE>(d_degree_old_prefix, num_items, smem,
                                     lambda);
}

template<unsigned BLOCK_SIZE, unsigned ITEMS_PER_BLOCK>
__global__
void moveDataKernel2(const degree_t* __restrict__ d_degree_new_prefix,
                      int                         num_items,
                      edge_t*        __restrict__ d_tmp,
                      byte_t**       __restrict__ d_ptrs_array) {
    __shared__ degree_t smem[ITEMS_PER_BLOCK];

    auto lambda = [&] (int pos, degree_t offset) {
            int  tmp_offset = d_degree_new_prefix[pos] + offset;
            auto ptr_vertex = reinterpret_cast<vid_t*>(d_ptrs_array[pos]);

            if (NUM_EXTRA_ETYPES == 1) {
                auto   tmp_data = reinterpret_cast<int2*>(d_tmp)[tmp_offset];
                auto ptr_weight = ptr_vertex + EDGES_PER_BLOCKARRAY;
                ptr_vertex[offset] = tmp_data.x;
                reinterpret_cast<int*>(ptr_weight)[offset]   = tmp_data.y;
            }
            else {
                auto tmp_data = reinterpret_cast<vid_t*>(d_tmp)[tmp_offset];
                ptr_vertex[offset] = tmp_data;
            }
        };
    xlib::binarySearchLB<BLOCK_SIZE>(d_degree_new_prefix, num_items, smem,
                                     lambda);
}

__global__
void scatterDegreeKernel(cuStingerDevice custinger,
                         const vid_t* __restrict__ d_unique,
                         const int*   __restrict__ d_counts,
                         int                       num_uniques,
                         int*         __restrict__ d_batch_offsets) {
    int     id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = id; i < num_uniques; i += stride)
        d_batch_offsets[d_unique[i]] = d_counts[i];
}

} // namespace custinger
