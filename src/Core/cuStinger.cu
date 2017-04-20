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
#include "Core/cuStingerGlobalSpace.cuh"
#include "Core/cuStingerTypes.cuh"

namespace cu_stinger {

void cuStinger::initializeVertexGlobal() noexcept {
    cuMemcpyToSymbol(_nV, d_nV);

    auto vertex_basic_ptr = reinterpret_cast<VertexBasicData*>(_d_vertices);
    cuMemcpyToSymbol(vertex_basic_ptr, d_vertex_basic_ptr);
    //--------------------------------------------------------------------------
    id_t round_nV = xlib::roundup_pow2(_nV);
    byte_t* vertex_data_ptrs[NUM_VTYPES];

    for (int i = 0; i < NUM_EXTRA_VTYPES; i++) {
        vertex_data_ptrs[i] = reinterpret_cast<byte_t*>(_d_vertices) +
                              round_nV * VTYPE_SIZE_PS[i + 1];
    }
    cuMemcpyToSymbol(vertex_data_ptrs, d_vertex_data_ptrs);
}

//==============================================================================

__device__ int d_array[10];

__global__ void printKernel() {
    for (id_t i = 0; i < d_nV; i++) {
        auto vertex = Vertex(i);
        auto degree = vertex.degree();
        //auto field0 = vertex.field<0>();
        printf("%d [%d, %d]:   ", i, vertex.degree(), vertex.limit());

        for (degree_t j = 0; j < vertex.degree(); j++) {
            auto   edge = vertex.edge(j);
            /*auto weight = edge.weight();
            auto  time1 = edge.time_stamp1();
            auto field0 = edge.field<0>();
            auto field1 = edge.field<1>();*/

            printf("%d   ", edge.dst());
        //    d_array[j] = edge.dst();
        }
        printf("\n");
    }
    //printf("\n");
    //from RAW:
    //
    //for (id_t i = 0; i < d_nV; i++) {
    //  for (degree_t j = 0; j < vertex.degrees(); j++) {
    //       auto edge = vertex.edge(i);
    //----------------------------------------------------
    //to PROPOSED:
    //
    //for (auto v : VertexSet) {
    //  for (auto edge : v) {
}

void cuStinger::print() noexcept {
    if (sizeof(degree_t) == 4 && sizeof(id_t) == 4) {
        printKernel<<<1, 1>>>();
        CHECK_CUDA_ERROR
    }
    else
        WARNING("Graph print is enable only with degree_t/id_t of size 4 bytes")
}

} // namespace cu_stinger
