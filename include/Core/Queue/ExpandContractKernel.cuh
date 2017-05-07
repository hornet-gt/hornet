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
#include <Core/cuStingerTypes.cuh>            //Vertex, Edge
#include <Support/Device/Definition.cuh>      //xlib::WARP_SIZE
#include <Support/Device/PTX.cuh>             //xlib::lane_id
#include <Support/Device/WarpScan.cuh>        //xlib::WarpScan
#include <Support/Device/BinarySearchLB.cuh>  //xlib::binarySearchLBAllPosNoDup

namespace custinger_alg {

/**
 * @brief
 */
template<unsigned BLOCK_SIZE, unsigned ITEMS_PER_BLOCK, typename Operator>
__global__
void ExpandContractLBKernel(custinger::cuStingerDevData data,
                            ptr2_t<custinger::vid_t>    d_queue,
                            ptr2_t<int>                 d_work,
                            int2*                       d_queue2_counter,
                            int                         num_vertices,
                            Operator                    op) {
    using namespace custinger;
    __shared__ degree_t smem[ITEMS_PER_BLOCK];

    auto lambda = [&](int pos, degree_t offset) {
        vid_t    dst_id;
        degree_t degree;
        if (pos != -1) {
            auto src_id = d_queue.first[pos];
            Vertex src(data, src_id);
            Edge dst_edge = src.edge(offset);
            bool     pred = op(src, dst_edge);

            dst_id = dst_edge.dst();
            Vertex dst_vertex(data, dst_id);
            degree = pred ? dst_vertex.degree() : 0;
        } else
            degree = 0;

        unsigned      ballot = __ballot(degree);
        int num_active_nodes = __popc(ballot);                  // at warp level

        int prefix_sum = degree;
        int total_sum, queue_offset, prefix_sum_old;
        xlib::WarpExclusiveScan<>::Add(prefix_sum, total_sum);

        if (xlib::lane_id() == xlib::WARP_SIZE - 1) {
            int2      info = make_int2(num_active_nodes, total_sum);
            auto  to_write = reinterpret_cast<long long unsigned&>(info);
            auto       old = atomicAdd(reinterpret_cast<long long unsigned*>
                                       (d_queue2_counter), to_write);
            auto      old2 = reinterpret_cast<int2&>(old);
            queue_offset   = old2.x;
            prefix_sum_old = old2.y;
        }
        prefix_sum  += __shfl(prefix_sum_old, xlib::WARP_SIZE - 1);
        queue_offset = __shfl(queue_offset, xlib::WARP_SIZE - 1);

        if (degree) {
            queue_offset += __popc(ballot & xlib::LaneMaskLT());
            d_work.second[queue_offset]  = prefix_sum;
            d_queue.second[queue_offset] = dst_id;
        }
    };
    xlib::binarySearchLBAllPosNoDup<BLOCK_SIZE>(d_work.first, num_vertices,
                                                 smem, lambda);
}

} // namespace custinger_alg
