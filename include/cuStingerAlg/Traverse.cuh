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
 *
 * @file
 */
#pragma once

#include "cuStingerAlg/cuStingerAlgConfig.cuh"
#include "Core/RawTypes.hpp"
#include "Core/cuStingerTypes.cuh"
#include "Support/Device/Definition.cuh"
#include "Support/Device/PTX.cuh"
#include "Support/Device/WarpScan.cuh"
#include "Support/Device/MergePath.cuh"
#include "Support/Host/Numeric.hpp"

/**
 * @brief
 */
namespace cu_stinger_alg {

__device__   int2  d_queue_counter;
__constant__ int*    d_work = nullptr;
__constant__ id_t* d_queue1 = nullptr;
__constant__ int2* d_queue2 = nullptr;

/**
 * @brief
 */
template<unsigned ITEMS_PER_BLOCK>
__global__ void loadBalancingExpand(int work_size) {
    __shared__ degree_t smem[ITEMS_PER_BLOCK];
    int index2 = (blockIdx.x * BLOCK_SIZE + xlib::warp_id() * xlib::WARP_SIZE) *
                 ITEMS_PER_BLOCK + xlib::lane_id();

    //if (threadIdx.x < work_size)
    //    printf("work: %d\n", d_work[threadIdx.x]);

    const auto lambda = [](int pos, degree_t offset) {
                            int index = d_work[pos] + offset;
                            id_t  src = d_queue1[pos];
                            //printf("%d\tp: %d\to:%d\ts: %d\n", threadIdx.x, pos, offset, src);
                            d_queue2[index] = xlib::make2(src, offset);
                        };
    xlib::binarySearchLB<BLOCK_SIZE>(d_work, work_size, smem, lambda);
}

/**
 * @brief
 */
template<typename Operator, typename... TArgs>
__global__ void loadBalancingContract(unsigned num_queue_edges, TArgs... args) {
    int     id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    //all warp threads must enter in the loop
    int frontier_round = xlib::upper_approx<xlib::WARP_SIZE>(num_queue_edges);

    for (int i = id; i < frontier_round; i += stride) {
        id_t dst_id;
        degree_t degree;
        if (i < num_queue_edges) {
            auto   item = d_queue2[i];
            auto src_id = item.x;
            auto offset = item.y;

            Vertex src(src_id);
            Edge dst_edge = src.edge(offset);
            auto     pred = Operator()(src, dst_edge, args...);

            dst_id = dst_edge.dst();
            Vertex dst_vertex(dst_id);
            degree    = pred ? dst_vertex.degree() : 0;
        } else
            degree = 0;

        unsigned      ballot = __ballot(degree);
        int num_active_nodes = __popc(ballot);                  // at warp level

        int prefix_sum = degree;
        int total_sum, queue_offset, prefix_sum_old;
        xlib::WarpExclusiveScan<>::Add(prefix_sum, total_sum);

        if (xlib::lane_id() == xlib::WARP_SIZE - 1) {
            int2      info = xlib::make2(num_active_nodes, total_sum);
            auto  to_write = reinterpret_cast<long long unsigned&>(info);
            auto       old = atomicAdd(reinterpret_cast<long long unsigned*>
                                       (&d_queue_counter), to_write);
            auto      old2 = reinterpret_cast<int2&>(old);
            queue_offset   = old2.x;
            prefix_sum_old = old2.y;
        }
        prefix_sum  += __shfl(prefix_sum_old, xlib::WARP_SIZE - 1);
        queue_offset = __shfl(queue_offset, xlib::WARP_SIZE - 1);

        if (degree) {
            queue_offset += __popc(ballot & xlib::LaneMaskLT());
            d_work[queue_offset]   = prefix_sum;
            d_queue1[queue_offset] = dst_id;
        }
    }
}

//==============================================================================

template<typename T, typename Operator>
__global__ void forAllKernel(T* array, int num_items, Operator op) {
    int     id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = id; i < num_items; i += stride)
        op(array[i]);
}

template<typename Operator, typename T>
void forAll(T* array, int num_items, Operator op) {
    forAllKernel <<< xlib::ceil_div<BLOCK_SIZE>(num_items), BLOCK_SIZE >>>
        (array, num_items, op);
}

} // namespace cu_stinger_alg
