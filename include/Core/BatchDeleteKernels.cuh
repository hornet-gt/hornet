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

struct TmpData {
    using WeightT = typename std::tuple_element<(NUM_ETYPES > 1 ? 1 : 0),
                                                 edge_t>::type;
    TmpData(vid_t id, WeightT weight) : id(id), weight(weight){}
    vid_t   id;
    WeightT weight;
};

__global__
void deleteEdgesKernel(cuStingerDevData data,
                       BatchUpdate      batch_update) {
    int     id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = id; i < batch_update.size(); i += stride) {
        vid_t src = batch_update.src(i);
        vid_t dst = batch_update.dst(i);

        Vertex vertex(data, src);
        auto    adj_ptr = vertex.edge_ptr();
        auto weight_ptr = vertex.edge_weight_ptr();

        auto pos = xlib::binary_search(adj_ptr, vertex.degree(), dst);
        adj_ptr[pos]    = -1;
        weight_ptr[pos] = -1;
    }
}

__global__
void collectDataKernel(cuStingerDevData data,
                       const vid_t*    __restrict__ d_unique,
                       const degree_t* __restrict__ d_count,
                       int num_uniques,
                       degree_t* __restrict__ d_degree_old,
                       degree_t* __restrict__ d_degree_new,
                       byte_t**  __restrict__ d_ptrs_array) {
    int     id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = id; i < num_uniques; i += stride) {
        auto d_vertex_ptrs = reinterpret_cast<VertexBasicData*>
                                (data.d_vertex_ptrs[0]);
        auto         src = d_unique[i];
        auto vertex_data = d_vertex_ptrs[src];
        auto  new_degree = vertex_data.degree - d_count[i];
        printf("src %d  degree %d, d_count %d\n", src, vertex_data.degree, d_count[i]);

        d_ptrs_array[i] = vertex_data.edge_ptr;
        d_degree_old[i] = vertex_data.degree;
        d_degree_new[i] = new_degree;

        d_vertex_ptrs[src] = VertexBasicData(new_degree, vertex_data.edge_ptr);
    }
    if (id == 0)
        d_degree_old[num_uniques] = 0;
    if (id == 1)
        d_degree_new[num_uniques] = 0;
}

template<unsigned BLOCK_SIZE, unsigned ITEMS_PER_BLOCK>
__global__
void moveDataKernel1(const degree_t* __restrict__ d_degree_old_prefix,
                     int                          num_items,
                     byte_t**        __restrict__ d_ptrs_array,
                     TmpData*        __restrict__ d_tmp) {
    __shared__ degree_t smem[ITEMS_PER_BLOCK];
    auto lambda = [&] (int pos, degree_t offset) {
                    int    tmp_offset = d_degree_old_prefix[pos] + offset;
                    auto          ptr = d_ptrs_array[pos];
                    TmpData tmp(ptr, );
                    d_tmp[tmp_offset] = d_ptrs_array[pos][offset];
                    d_tmp[tmp_offset] = TmpData(ptr, );
                };
    xlib::binarySearchLB<BLOCK_SIZE>(d_degree_old_prefix, num_items, smem,
                                     lambda);
}

template<unsigned BLOCK_SIZE, unsigned ITEMS_PER_BLOCK>
__global__
void moveDataKernel2(const degree_t* __restrict__ d_degree_new_prefix,
                      int                         num_items,
                      TmpData*       __restrict__ d_tmp,
                      TmpData**      __restrict__ d_ptrs_array) {
    __shared__ degree_t smem[ITEMS_PER_BLOCK];
    auto lambda = [&] (int pos, degree_t offset) {
                    int tmp_offset = d_degree_new_prefix[pos] + offset;
                    d_ptrs_array[pos][offset] = d_tmp[tmp_offset];
                };
    xlib::binarySearchLB<BLOCK_SIZE>(d_degree_new_prefix, num_items, smem,
                                     lambda);
}

} // namespace custinger
