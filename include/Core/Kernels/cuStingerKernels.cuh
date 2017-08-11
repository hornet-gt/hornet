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
#include "Support/Device/BinarySearchLB.cuh"

namespace custinger {

__global__ void printKernel(cuStingerDevice custinger) {
    for (vid_t i = 0; i < custinger.nV(); i++) {
        auto vertex = custinger.vertex(i);
        auto degree = vertex.degree();
        //auto field0 = vertex.field<0>();
        printf("%d [%d, %d, 0x%llX]:    ", i, vertex.degree(), -1,
                                            vertex.neighbor_ptr());

        auto ptr = vertex.neighbor_ptr();
        //auto weight_ptr = vertex.edge_weight_ptr();
        for (degree_t j = 0; j < vertex.degree(); j++) {
            //auto   edge = vertex.edge(j);
            //auto weight = edge.weight();
            /*auto  time1 = edge.time_stamp1();
            auto field0 = edge.field<0>();
            auto field1 = edge.field<1>();*/

            //printf("%d    ", edge.dst_id());
            printf("%d    ", vertex.neighbor_id(j));
            //printf("(%d, %d)    ", ptr[j], weight_ptr[j]);
            //printf("[%d, %d]    ", edge.dst(), edge.weight());
            //edge.set_weight(-edge.weight());
        //    d_array[j] = edge.dst();
        }
        printf("\n");
    }
    //printf("\n");
    //from RAW:
    //
    //for (vid_t i = 0; i < d_nV; i++) {
    //  for (degree_t j = 0; j < vertex.degrees(); j++) {
    //       auto edge = vertex.edge(i);
    //----------------------------------------------------
    //to PROPOSED:
    //
    //for (auto v : VertexSet) {
    //  for (auto edge : v) {
}

template<unsigned BLOCK_SIZE>
__global__ void CSRtoCOOKernel(const eoff_t* __restrict__ csr_offsets,
                               vid_t                      nV,
                               vid_t* __restrict__        coo_src) {
    __shared__ int smem[xlib::SMemPerBlock<BLOCK_SIZE, int>::value];

    const auto& lambda = [&](int pos, eoff_t offset) {
                            eoff_t index = csr_offsets[pos] + offset;
                            coo_src[index] = pos;
                        };
    xlib::binarySearchLB<BLOCK_SIZE>(csr_offsets, nV, smem, lambda);
}

__global__ void checkSortedKernel(cuStingerDevice custinger) {
    int    idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (vid_t i = idx; i < custinger.nV(); i += stride) {
        auto vertex = custinger.vertex(i);
        auto    ptr = vertex.neighbor_ptr();

        for (degree_t j = 0; j < vertex.degree() - 1; j++) {
            if (ptr[j] > ptr[j + 1]) {
                printf("Edge %d\t-> %d\t(d: %d)\t(value %d) not sorted \n",
                        i, j, vertex.degree(), ptr[j]);
            }
            else if (ptr[j] == ptr[j + 1]) {
                printf("Edge %d\t-> %d\t(d: %d)\t(value %d) duplicated\n",
                        i, j, vertex.degree(), ptr[j]);
            }
        }
    }
}

__global__
void buildDegreeKernel(cuStingerDevice        custinger,
                       degree_t* __restrict__ d_degrees) {
    int    idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < custinger.nV(); i += stride)
        d_degrees[i] = custinger.vertex(i).degree();
}

} // namespace custinger
