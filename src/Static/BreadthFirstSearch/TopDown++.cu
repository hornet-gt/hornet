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
#include "Static/BreadthFirstSearch/TopDown++.cuh"

namespace custinger_alg {

const dist_t INF = std::numeric_limits<dist_t>::max();

//------------------------------------------------------------------------------
///////////////
// OPERATORS //
///////////////

struct BFSOperatorNoAtomic {
    dist_t* d_distances;
    dist_t  current_level;
    BFSOperatorNoAtomic(dist_t* d_distances_, dist_t current_level_) :
                                d_distances(d_distances_),
                                current_level(current_level_) {}

    __device__ __forceinline__
    bool operator()(const Vertex& src, const Edge& edge) {
        auto dst = edge.dst();
        if (d_distances[dst] == INF) {
            d_distances[dst] = current_level;
            return true;             // the vertex dst is active
        }
        return false;                // the vertex dst is not active
    }

};
//------------------------------------------------------------------------------
/////////////////
// BfsTopDown2 //
/////////////////

BfsTopDown2::BfsTopDown2(custinger::cuStinger& custinger) :
                                 StaticAlgorithm(custinger),
                                 queue(custinger, true) {
    cuMalloc(d_distances, custinger.nV());
    reset();
}

BfsTopDown2::~BfsTopDown2() {
    cuFree(d_distances);
}

void BfsTopDown2::reset() {
    current_level = 1;
    queue.clear();

    auto distances = d_distances;
    forAllnumV(custinger, [=] __device__ (int i){ distances[i] = INF; } );
}

void BfsTopDown2::set_parameters(vid_t source) {
    bfs_source = source;
    queue.insert(bfs_source);
    cuMemcpyToDevice(0, d_distances + bfs_source);
}

void BfsTopDown2::run() {
    while (queue.size() > 0) {
        queue.traverse_edges( BFSOperatorNoAtomic(d_distances, current_level) );
        current_level++;
    }
}

void BfsTopDown2::release() {
    cuFree(d_distances);
    d_distances = nullptr;
}

bool BfsTopDown2::validate() {
    using namespace graph;
    GraphStd<vid_t, eoff_t> graph(custinger.csr_offsets(), custinger.nV(),
                                  custinger.csr_edges(), custinger.nE());
    BFS<vid_t, eoff_t> bfs(graph);
    bfs.run(bfs_source);

    auto h_distances = bfs.distances();
    return cu::equal(h_distances, h_distances + graph.nV(),  d_distances);
}

} // namespace custinger_alg
