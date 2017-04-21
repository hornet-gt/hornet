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

#include "Core/cuStingerConfig.hpp"
#include "Core/cuStingerTypes.cuh"
#include "Support/Numeric.hpp"
#include "Support/Definition.cuh"
#include "Support/PTX.cuh"
#include "Support/WarpScan.cuh"
#include "Support/MergePath.cuh"


/**
 * @brief
 */
namespace cu_stinger_alg {

__constant__ int   d_queue_counter = 0;
__constant__ int*  d_work   = nullptr;
__constant__ id_t* d_queue1 = nullptr;
__constant__ int2* d_queue2 = nullptr;

const int BLOCK_SIZE = 256;

/**
 * @brief
 */
__global__ void LoadBalancingExpand(int work_size) {
    using cu_stinger::id_t;
    using cu_stinger::degree_t;
    __shared__ degree_t smem[xlib::SMemPerBlock<BLOCK_SIZE, degree_t>::value];
    int id = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    const auto lambda = [&](int pos, degree_t offset) {
                            int index = d_work[pos] + offset;
                            assert(id == index);
                            id_t  src = d_queue1[pos];
                            d_queue2[index] = xlib::make2(src, offset);
                        };
    xlib::binarySearchLB<BLOCK_SIZE>(d_work, work_size, smem, lambda);
}

/**
 * @brief
 */
template<typename EdgeOperator, typename... TArgs>
__global__ void LoadBalancingContract(unsigned frontier_size, TArgs... args) {
    using cu_stinger::id_t;
    using cu_stinger::off_t;
    using namespace cu_stinger;

    id_t    id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    int frontier_approx = xlib::upper_approx<xlib::WARP_SIZE>(frontier_size);

    for (off_t e_offset = id; e_offset < frontier_approx; e_offset += stride) {
        id_t src_id;
        degree_t degree;
        if (e_offset < frontier_size) {
            auto     item = d_queue2[e_offset];
            auto edge_idx = item.y;

            src_id = item.x;
            Vertex src(src_id);
            auto pred = EdgeOperator(src, src.edge(edge_idx), args...);
            degree    = pred ? src.degree() : 0;
        } else
            degree = 0;

        unsigned      ballot = __ballot(degree);
        int num_active_nodes = __popc(ballot); // at warp level

        int total_sum, queue_offset, prefix_sum_old;
        xlib::WarpExclusiveScan<>::Add(degree, total_sum);

        if (xlib::lane_id() == xlib::WARP_SIZE - 1) {
            int2      info = xlib::make2(num_active_nodes, total_sum);
            auto  to_write = reinterpret_cast<long long unsigned&>(info);
            auto       old = atomicAdd(reinterpret_cast<long long unsigned*>
                                       (d_queue_counter), to_write);
            int2      old2 = reinterpret_cast<int2&>(old);

            queue_offset   = old2.x;
            prefix_sum_old = old2.y;
        }
        int prefix_sum =  degree + __shfl(prefix_sum_old, xlib::WARP_SIZE - 1);
        queue_offset   = __shfl(queue_offset, xlib::WARP_SIZE - 1);

        if (degree) {
            queue_offset += __popc(ballot & xlib::LaneMaskLT());
            d_work[queue_offset]   = prefix_sum;
            d_queue1[queue_offset] = src_id;
        }
    }
}

template<typename BFSOperator, typename... TArgs>
TraverseAndFilter(TArgs... args) {
    LoadBalancingExpand
        <<< xlib::ceil_div<BLOCK_SIZE>(vertex_frontier_size), BLOCK_SIZE >>>
        (vertex_frontier_size);

    CHECK_CUDA_ERROR

    LoadBalancingContract<BFSOperator>
        <<< xlib::ceil_div<BLOCK_SIZE>(edge_frontier_size), BLOCK_SIZE >>>
        (edge_frontier_size, args...);

    CHECK_CUDA_ERROR
    cudaDeviceSynchronize();
}

} // namespace cu_stinger_alg
