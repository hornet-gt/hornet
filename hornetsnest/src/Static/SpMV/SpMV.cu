/**
 * @author Federico Busato                                                  <br>
 *         Univerity of Verona, Dept. of Computer Science                   <br>
 *         federico.busato@univr.it
 * @date September, 2017
 * @version v2
 *
 * @copyright Copyright © 2017 Hornet. All rights reserved.
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
#include "Static/SpMV/SpMV.cuh"
#include <Graph/GraphWeight.hpp>

namespace hornets_nest {
///////////////
// OPERATORS //
///////////////

struct SpMVOperator {
    int* d_vector;
    int* d_result;

    OPERATOR(Vertex& vertex, Edge& edge) {
        //printf("%d %d      %d\n", vertex.id(), edge.dst_id(), edge.weight());
        auto   col = edge.dst_id();
        auto value = edge.template field<0>();
        auto   sum = value * d_vector[col];
        atomicAdd(d_result + vertex.id(), sum);
    }
};
//------------------------------------------------------------------------------
//////////
// SpMV //
//////////

SpMV::SpMV(HornetGraph& hornet, int* h_vector) :
                                StaticAlgorithm(hornet),
                                load_balancing(hornet),
                                h_vector(h_vector) {
    gpu::allocate(d_vector, hornet.nV());
    gpu::allocate(d_result, hornet.nV());
    host::copyToDevice(h_vector, hornet.nV(), d_vector);
    reset();
}

SpMV::~SpMV() {
    release();
}

void SpMV::reset() {
    gpu::memsetZero(d_result, hornet.nV());
}

void SpMV::run() {
    forAllEdges(hornet, SpMVOperator { d_vector, d_result }, load_balancing);
}

void SpMV::release() {
    gpu::free(d_vector);
    gpu::free(d_result);
    d_vector = nullptr;
    d_result = nullptr;
}

bool SpMV::validate() {
    //auto   n_rows = hornet.nV();
    //auto   h_rows = hornet.csr_offsets();
    //auto   h_cols = hornet.csr_edges();
    //auto  h_value = hornet.edge_field<1>();
    //auto h_result = new int[hornet.nV()];

    //for (auto i = 0; i < n_rows; i++) {
    //    int sum = 0;
    //    for (auto j = h_rows[i]; j < h_rows[i + 1]; j++)
    //        sum += h_value[j] * h_vector[h_cols[j]];
    //    h_result[i] = sum;
    //}
    //bool ret = gpu::equal(h_result, h_result + hornet.nV(), d_result);
    //delete[] h_result;
    //return ret;
    return true;
}

} // namespace hornets_nest
