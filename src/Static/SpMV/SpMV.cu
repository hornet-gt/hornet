/**
 * @author Federico Busato                                                  <br>
 *         Univerity of Verona, Dept. of Computer Science                   <br>
 *         federico.busato@univr.it
 * @date September, 2017
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
#include "Static/SpMV/SpMV.cuh"
#include <GraphIO/GraphWeight.hpp>

namespace hornet_alg {
///////////////
// OPERATORS //
///////////////

struct SpMVOperator {
    int* d_vector;
    int* d_result;

    OPERATOR(Vertex& vertex, Edge& edge) {
        auto dst = edge.dst_id();
        edge.weight();
    }
};
//------------------------------------------------------------------------------
/////////////////
// SpMV //
/////////////////

SpMV::SpMV(HornetGPU& hornet, const int* h_vector) :
                                StaticAlgorithm(hornet),
                                load_balacing(hornet),
                                h_vector(h_vector) {
    gpu::allocate(d_vector, hornet.nV());
    gpu::allocate(d_result, hornet.nV());
    host::copyToDevice(h_vector, hornet.nV(), d_vector);
    reset();
}

SpMV::~SpMV() {
    gpu::free(d_vector, d_result);
}

void SpMV::reset() {
    gpu::memset0x00(d_result, hornet.nV());
}

void SpMV::run() {
    forAllEdges(hornet, SpMVOperator { d_vector, d_result }, load_balacing);
    //segmented reduce
}

void SpMV::release() {
    gpu::free(d_vector, d_result);
    d_vector = nullptr;
    d_result = nullptr;
}

bool SpMV::validate() {
    using namespace graph;
    GraphWeight<vid_t, eoff_t, int> graph(hornet.csr_offsets(), hornet.nV(),
                                          hornet.csr_edges(), hornet.nE());
    //HOST SpMV
    auto h_result = new int[hornet.nV()]();
    for (auto i = 0; i < hornet.nV(); i++) {
        const auto& vertex = hornet.vertex(i);
        int sum = 0;
        for (auto j = 0; j < vertex.degree(); j++) {
            const auto& edge = vertex.edge(j);
            sum += edge.weight() * h_vector[edge.dst_id()];
        }
        h_result[i] = sum;
    }
    bool ret = gpu::equal(h_result, h_result + graph.nV(), d_result);
    delete[] h_result;
    return ret;
}

} // namespace hornet_alg
