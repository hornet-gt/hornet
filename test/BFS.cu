///@files
#include "Kernels/Traverse.cuh"
#include "Core/cuStinger.hpp"
#include "GraphIO/GraphStd.hpp" //GraphStd
#include "Support/Timer.cuh"    //Timer

using namespace cu_stinger;
using namespace cu_stinger_alg;
using namespace timer;

struct VertexInitialization;
struct BFSOperator;

using dist_t = int;

/**
 * @brief Example tester for cuSTINGER.
 * Loads an input graph, creates a batches of edges, inserts them into the
 * graph, and then removes them from the graph.
 */
int main(int argc, char* argv[]) {
    // INIT //
    graph::GraphStd<cu_stinger::id_t, cu_stinger::off_t> graph;
    graph.read(argv[1]);

    cuStingerInit custinger_init(graph.nV(), graph.nE(),
                                 graph.out_offsets_array(),
                                 graph.out_edges_array());
    cuStinger custiger_graph(custinger_init);
    id_t bfs_source = 0;
    //--------------------------------------------------------------------------
    // BFS INIT //
    ForAll<VertexInitialization>(d_distance, graph.nV());
    Queue queue;
    queue.insert(bfs_source);
    //queue.insert(bfs_sources, num_sources);

    dist_t level = 1;
    //--------------------------------------------------------------------------
    // BFS ALGORITHM //
    while (queue.size() > 0) {
        TraverseAndFilter<BFSOperator>(level);
        level++;
    }
}

const dist_t INFINITY = std::numeric_limits<dist_t>::max();

struct VertexInitialization {
    __device__ __forceinline__
    VertexInitialization(id_t vertex_id) {
        d_distance[vertex_id] = INFINITY;
    }
};

struct BFSOperator {
    __device__ __forceinline__
    bool operator()(Vertex src, Edge edge, dist_t level) {
        auto dst = edge.dst();
        if (d_distance[dst] == INFINITY) {
            d_distance[dst] = level;
            return true;            // the vertex dst is active
        }
        return false;               // the vertex dst is not active
    }
};
