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
#include "Support/Device/Definition.cuh"
#include "Support/Host/Algorithm.hpp"
#include "Support/Device/BinarySearchLB.cuh"

namespace custinger {

__global__
void collectOldDegreeKernel(cuStingerDevData          data,
                            const vid_t* __restrict__ d_unique,
                            int                       num_uniques,
                            degree_t*    __restrict__ d_degree_old,
                            int*         __restrict__ d_inverse_pos) {

    int     id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = id; i < num_uniques; i += stride) {
        auto        src = d_unique[i];
        d_degree_old[i] = Vertex(data, src).degree();
        d_inverse_pos[src] = i;
    }
    if (id == 0)
        d_degree_old[num_uniques] = 0;
}

__global__
void deleteEdgesKernel(cuStingerDevData              data,
                       BatchUpdate                   batch_update,
                       const degree_t*  __restrict__ d_degree_old_prefix,
                       bool*            __restrict__ d_flags,
                       const int*       __restrict__ d_inverse_pos) {
    int     id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = id; i < batch_update.size(); i += stride) {
        vid_t src = batch_update.src(i);
        vid_t dst = batch_update.dst(i);

        Vertex src_vertex(data, src);
        auto adj_ptr = src_vertex.edge_ptr();

        auto pos = xlib::binary_search(adj_ptr, src_vertex.degree(), dst);
        int inverse_pos = d_inverse_pos[src];
        d_flags[d_degree_old_prefix[inverse_pos] + pos] = false;
        //printf("del %d %d \t%d\t%d\t%d\n",
        //    src, dst, d_degree_old_prefix[inverse_pos], pos,
        //    d_degree_old_prefix[inverse_pos] + pos);
    }
}

__global__
void collectDataKernel(cuStingerDevData          data,
                       const vid_t* __restrict__ d_unique,
                       degree_t*    __restrict__ d_count,
                       int                       num_uniques,
                       degree_t*    __restrict__ d_degree_new,
                       byte_t**     __restrict__ d_ptrs_array) {
    int     id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = id; i < num_uniques; i += stride) {
        auto d_vertex_ptrs = reinterpret_cast<VertexBasicData*>
                                (data.d_vertex_ptrs[0]);
        auto         src = d_unique[i];
        auto vertex_data = d_vertex_ptrs[src];
        auto  new_degree = vertex_data.degree - d_count[i];

        d_ptrs_array[i] = vertex_data.edge_ptr;
        d_degree_new[i] = new_degree;

        d_vertex_ptrs[src] = VertexBasicData(new_degree, vertex_data.edge_ptr);
    }
    if (id == 1)
        d_degree_new[num_uniques] = 0;
    if (id == 2)
        d_count[num_uniques] = 0;
}

template<unsigned BLOCK_SIZE, unsigned ITEMS_PER_BLOCK>
__global__
void moveDataKernel1(const degree_t* __restrict__ d_degree_old_prefix,
                     int                          num_items,
                     byte_t**        __restrict__ d_ptrs_array,
                     int2*           __restrict__ d_tmp) {
    __shared__ degree_t smem[ITEMS_PER_BLOCK];

    auto lambda = [&] (int pos, degree_t offset) {
                    xlib::SeqDev<ETypeSizePS> ETYPE_SIZE_PS_D;

                    int  tmp_offset = d_degree_old_prefix[pos] + offset;
                    auto ptr_vertex = (vid_t*) d_ptrs_array[pos];
                    auto ptr_weight = ptr_vertex +
                                      EDGES_PER_BLOCKARRAY;// * ETYPE_SIZE_PS_D[1];
                    auto adj_vertex = reinterpret_cast<vid_t*>
                                        (ptr_vertex)[offset];
                    auto adj_weight = reinterpret_cast<int*>
                                        (ptr_weight)[offset];
                    d_tmp[tmp_offset] = make_int2(adj_vertex, adj_weight);
                };
    xlib::binarySearchLB<BLOCK_SIZE>(d_degree_old_prefix, num_items, smem,
                                     lambda);
}

template<unsigned BLOCK_SIZE, unsigned ITEMS_PER_BLOCK>
__global__
void moveDataKernel2(const degree_t* __restrict__ d_degree_new_prefix,
                      int                         num_items,
                      int2*          __restrict__ d_tmp,
                      byte_t**       __restrict__ d_ptrs_array) {
    __shared__ degree_t smem[ITEMS_PER_BLOCK];

    auto lambda = [&] (int pos, degree_t offset) {
                    xlib::SeqDev<ETypeSizePS> ETYPE_SIZE_PS_D;

                    int  tmp_offset = d_degree_new_prefix[pos] + offset;
                    auto ptr_vertex = (vid_t*) d_ptrs_array[pos];
                    auto ptr_weight = ptr_vertex +
                                      EDGES_PER_BLOCKARRAY;// * ETYPE_SIZE_PS_D[1];
                    auto   tmp_data = d_tmp[tmp_offset];
                    reinterpret_cast<vid_t*>(ptr_vertex)[offset] = tmp_data.x;
                    reinterpret_cast<int*>
                        (ptr_weight)[offset] = tmp_data.y;
                };
    xlib::binarySearchLB<BLOCK_SIZE>(d_degree_new_prefix, num_items, smem,
                                     lambda);
}

} // namespace custinger
