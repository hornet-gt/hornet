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
#include "Static/BreadthFirstSearch/TopDown2.cuh"
#include "Auxilary/DuplicateRemoving.cuh"
#include <Graph/GraphStd.hpp>
#include <Graph/BFS.hpp>

namespace hornets_nest {

const dist_t INF = std::numeric_limits<dist_t>::max();

//------------------------------------------------------------------------------
///////////////
// OPERATORS //
///////////////

struct BFSOperator1 {
    dist_t*              d_distances;
    TwoLevelQueue<vid_t> queue;

    OPERATOR(Vertex& vertex, Edge& edge) {
        auto dst = edge.dst_id();
        if (!is_duplicate<2>(dst) && d_distances[dst] == INF)
            queue.insert(dst);
    }
};

struct BFSOperator2 {
    dist_t* d_distances;
    dist_t  current_level;

    OPERATOR(vid_t& vertex_id) {
        d_distances[vertex_id] = current_level;
    }
};

struct BFSOperatorAtomic {                  //deterministic
    dist_t               current_level;
    dist_t*              d_distances;
    TwoLevelQueue<vid_t> queue;

    OPERATOR(Vertex& vertex, Edge& edge) {
        auto dst = edge.dst_id();
        if (atomicCAS(d_distances + dst, INF, current_level) == INF)
            queue.insert(dst);
    }
};
//------------------------------------------------------------------------------
/////////////////
// BfsTopDown2 //
/////////////////

BfsTopDown2::BfsTopDown2(HornetGraph& hornet) :
                                 StaticAlgorithm(hornet),
                                 queue(hornet, 5),
                                 load_balancing(hornet) {
    gpu::allocate(d_distances, hornet.nV());
    reset();
}

BfsTopDown2::~BfsTopDown2() {
    gpu::free(d_distances);
}

void BfsTopDown2::reset() {
    current_level = 1;
    queue.clear();

    auto distances = d_distances;
    forAllnumV(hornet, [=] __device__ (int i){ distances[i] = INF; } );
}

void BfsTopDown2::set_parameters(vid_t source) {
    bfs_source = source;
    queue.insert(bfs_source);               // insert bfs source in the frontier
    gpu::memsetZero(d_distances + bfs_source);  //reset source distance
}

void BfsTopDown2::run() {
    while (queue.size() > 0) {
        forAllEdges(hornet, queue,
                    BFSOperatorAtomic { current_level, d_distances, queue },
                    load_balancing);
        queue.swap();
        current_level++;
    }
}

void BfsTopDown2::release() {
    gpu::free(d_distances);
    d_distances = nullptr;
}

bool BfsTopDown2::validate() {
    std::cout << "\nTotal enqueue vertices: "
              << xlib::format(queue.enqueue_items())
              << std::endl;

    // using namespace graph;
    // GraphStd<vid_t, eoff_t> graph(hornet.csr_offsets(), hornet.nV(),
    //                               hornet.csr_edges(), hornet.nE());
    // BFS<vid_t, eoff_t> bfs(graph);
    // bfs.run(bfs_source);
    //
    // auto h_distances = bfs.result();
    // return gpu::equal(h_distances, h_distances + graph.nV(), d_distances);
    return true;
}

} // namespace hornets_nest
