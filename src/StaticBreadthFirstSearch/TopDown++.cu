#include "StaticBreadthFirstSearch/TopDown++.cuh"

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
        return false;
    }

};
//------------------------------------------------------------------------------
////////////////
// BfsTopDown2 //
////////////////

BfsTopDown2::BfsTopDown2(custinger::cuStinger& custinger) :
                         StaticAlgorithm(custinger), queue(custinger, true) {

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
    cuMemcpyToDevice(0, d_distances + bfs_source);
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
