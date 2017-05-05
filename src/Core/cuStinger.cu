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
#include "Core/cuStinger.hpp"
#include "Core/cuStingerTypes.cuh"        //VertexBasicData
#include "Support/Device/CubWrapper.cuh"  //CubSortByValue

namespace custinger {

__device__ int d_array[10];

__global__ void printKernel(cuStingerDevData data) {
    for (vid_t i = 0; i < data.nV; i++) {
        auto vertex = Vertex(data, i);
        auto degree = vertex.degree();
        //auto field0 = vertex.field<0>();
        printf("%d [%d, %d]:    ", i, vertex.degree(), vertex.limit());

        for (degree_t j = 0; j < vertex.degree(); j++) {
            auto   edge = vertex.edge(j);
            /*auto weight = edge.weight();
            auto  time1 = edge.time_stamp1();
            auto field0 = edge.field<0>();
            auto field1 = edge.field<1>();*/

            printf("%d    ", edge.dst());
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

void cuStinger::print() noexcept {
    if (sizeof(degree_t) == 4 && sizeof(vid_t) == 4) {
        printKernel<<<1, 1>>>(device_data());
        CHECK_CUDA_ERROR
    }
    else {
        WARNING("Graph print is enabled only with degree_t/vid_t of size"
                " 4 bytes")
    }
}

void cuStinger::transpose() noexcept {
    mem_manager.clear();

    eoff_t* d_csr_offsets, *d_counts_out;
    vid_t*  d_csr_edges, *d_unique_out, *d_csr_edges_sorted;
    cuMalloc(d_csr_offsets, _nV + 1);
    cuMalloc(d_csr_edges, _nE);
    cuMalloc(d_csr_edges_sorted, _nE);
    cuMalloc(d_counts_out, _nV + 1);
    cuMalloc(d_unique_out, _nV);
    cuMemcpyToDeviceAsync(_csr_offsets, _nV + 1, d_csr_offsets);
    cuMemcpyToDeviceAsync(_csr_edges, _nE, d_csr_edges);
    cuMemcpyToDeviceAsync(0, d_counts_out + _nV);

    xlib::CubSortByValue<vid_t>(d_csr_edges, _nE, d_csr_edges_sorted, _nV - 1);
    xlib::CubRunLengthEncode<vid_t, eoff_t>(d_csr_edges_sorted, _nE,
                                            d_unique_out, d_counts_out);
    cuMemset0x00(d_unique_out, _nV);
    xlib::CubExclusiveSum<eoff_t>(d_counts_out, _nV + 1);

    //transpose_edges(d_csr_offsets, d_csr_edges, d_counts_out, d_unique_out
    //               [](atomicAdd(&, 1));
    cuFree(d_csr_offsets, d_csr_edges, d_csr_edges_sorted, d_counts_out,
           d_unique_out);

    _csr_offsets = new eoff_t[_nV + 1];
    _csr_edges   = new vid_t[_nV + 1];
    cuMemcpyToHostAsync(d_csr_offsets, _nV + 1,
                        const_cast<eoff_t*>(_csr_offsets));
    cuMemcpyToHostAsync(d_csr_edges, _nE, const_cast<vid_t*>(_csr_edges));
    _internal_csr_data = true;
    initialize();
}

} // namespace custinger
