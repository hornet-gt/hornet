///@files
#include "Core/cuStinger.hpp"     //cuStingerInit, cuStinger
#include "cuStingerAlg/Queue.cuh" //Queue
#include "GraphIO/GraphStd.hpp"   //GraphStd
#include "Support/Host/Timer.hpp"      //Timer

using namespace cu_stinger;
using namespace cu_stinger_alg;
using namespace timer;

using     dist_t = int;
const dist_t INF = std::numeric_limits<dist_t>::max();

struct BFSOperator;
struct BFSOperatorNoAtomic;

struct VertexInitialization {
    __device__ __forceinline__
    void operator()(dist_t& vertex_distance) {
        vertex_distance = INF;
    }
};

int main(int argc, char* argv[]) {
    using cu_stinger::id_t;
    using cu_stinger::off_t;
    //--------------------------------------------------------------------------
    // INIT //
    graph::GraphStd<id_t, off_t> graph;
    graph.read(argv[1]);

    cuStingerInit custinger_init(graph.nV(), graph.nE(),
                                 graph.out_offsets_array(),
                                 graph.out_edges_array());
    cuStinger custiger_graph(custinger_init);
    id_t    bfs_source = 0;
    dist_t* d_distance;        //d_distance can accessed on both host and device
    Allocate alloc(d_distance, graph.nV());
    //--------------------------------------------------------------------------
    // BFS INIT //
    forAll(d_distance, graph.nV(), VertexInitialization());
    Queue queue(custinger_init);
    queue.insert(bfs_source);
    //queue.insert(bfs_sources, num_sources);               // Multi-sources BFS
    dist_t level = 1;
    //--------------------------------------------------------------------------
    // BFS ALGORITHM //
    while (queue.size() > 0) {
        queue.traverseAndFilter<BFSOperator>(d_distance, level);
        level++;
        std::cout << "---------------------------------------------------" << std::endl;
    }
    //--------------------------------------------------------------------------
    //auto is_correct = std::equal(h_distance, h_distance + nV, d_distance);
    //std::cout << (is_correct ? "Correct <>\n" : "! Not Correct\n");
}

//------------------------------------------------------------------------------

struct BFSOperator {
    __device__ __forceinline__
    bool operator()(Vertex src, Edge edge, dist_t* d_distance, dist_t level) {
        auto dst = edge.dst();
        auto old = atomicCAS(d_distance + dst, INF, level);
        printf("src %d\t dest %d\t dist %d\ta: %d\n", src.id(), edge.dst(), old, old == INF);
        return old == INF;
    }
};

struct BFSOperatorNoAtomic {
    __device__ __forceinline__
    bool operator()(Vertex src, Edge edge, dist_t* d_distance, dist_t level) {
        auto dst = edge.dst();
        if (d_distance[dst] == INF) {
            d_distance[dst] = level;
            return true;    // the vertex dst is active
        }
        return false;       // the vertex dst is not active
    }
};
