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
#include "Static/ShortestPath/SSSP.cuh"
#include <GraphIO/GraphWeight.hpp>
#include <GraphIO/BellmanFord.hpp>

namespace hornets_nest {

const weight_t INF = std::numeric_limits<weight_t>::max();

//------------------------------------------------------------------------------
///////////////
// OPERATORS //
///////////////

struct SSSPOperator {
    weight_t*            d_distances;
    TwoLevelQueue<vid_t> queue;

    OPERATOR(Vertex& vertex, Edge& edge) {
        auto       src = vertex.id();
        auto       dst = edge.dst_id();
        auto    weight = edge.weight();
        auto tentative = d_distances[src] + weight;
        if (tentative < d_distances[dst]) {
            d_distances[dst] = tentative;
            queue.insert(dst);
        }
    }
};
//------------------------------------------------------------------------------
/////////////////
// SSSP //
/////////////////

SSSP::SSSP(HornetGraph& hornet) : StaticAlgorithm(hornet),
                                queue(hornet),
                                load_balacing(hornet) {
    gpu::allocate(d_distances, hornet.nV());
    reset();
}

SSSP::~SSSP() {
    gpu::free(d_distances);
}

void SSSP::reset() {
    queue.clear();
    auto distances = d_distances;
    forAllnumV(hornet, [=] __device__ (int i){ distances[i] = INF; } );
}

void SSSP::set_parameters(vid_t source) {
    sssp_source = source;
    queue.insert(sssp_source);
    host::copyToDevice(weight_t(0), d_distances + sssp_source);
}

void SSSP::run() {
    while (queue.size() > 0) {
        forAllEdges(hornet, queue, SSSPOperator { d_distances, queue },
                    load_balacing);
        queue.swap();
    }
}

void SSSP::release() {
    gpu::free(d_distances);
    d_distances = nullptr;
}

bool SSSP::validate() {
    using namespace graph;
    GraphWeight<vid_t, eoff_t, weight_t>
        graph(hornet.csr_offsets(), hornet.nV(),
              hornet.csr_edges(), hornet.nE(), hornet.edge_field<1>());
    BellmanFord<vid_t, eoff_t, weight_t> sssp(graph);
    sssp.run(sssp_source);

    auto h_distances = sssp.result();
    return gpu::equal(h_distances, h_distances + graph.nV(), d_distances);
}

} // namespace hornets_nest
