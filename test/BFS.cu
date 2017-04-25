#include "Core/cuStinger.hpp"           //cuStingerInit, cuStinger
#include "cuStingerAlg/Queue2.cuh"      //Queue
#include "GraphIO/BFS.hpp"              //BFS
#include "GraphIO/GraphStd.hpp"         //GraphStd
#include "Support/Device/Algorithm.hpp" //cu::equal
#include "Support/Host/Timer.hpp"       //Timer

using namespace cu_stinger;
using namespace cu_stinger_alg;
using namespace timer;

using     dist_t = int;
const dist_t INF = std::numeric_limits<dist_t>::max();

struct BFSOperatorAtomic;
struct BFSOperatorNoAtomic;

struct VertexInitialization {
    __device__ __forceinline__
    void operator()(dist_t& vertex_distance) {
        vertex_distance = INF;
    }
};

//==============================================================================

int main(int argc, char* argv[]) {
    using cu_stinger::id_t;
    using cu_stinger::off_t;
    id_t    bfs_source = 0;
    //--------------------------------------------------------------------------
    // HOST BFS //
    graph::GraphStd<id_t, off_t> graph;
    graph.read(argv[1]);
    graph::BFS<id_t, off_t> bfs(graph);
    bfs.run(bfs_source);

    auto h_distances = bfs.distances();
    //--------------------------------------------------------------------------
    // DEVICE INIT //
    cuStingerInit custinger_init(graph.nV(), graph.nE(),
                                 graph.out_offsets_array(),
                                 graph.out_edges_array());
    cuStinger custiger_graph(custinger_init);

    dist_t* d_distances;
    Allocate alloc(d_distances, graph.nV());
    //--------------------------------------------------------------------------
    // BFS INIT //
    forAll(d_distances, graph.nV(), VertexInitialization());
    Queue queue(custinger_init);
    queue.insert(bfs_source);
    //queue.insert(bfs_sources, num_sources);               // Multi-sources BFS
    cuMemcpyToDevice(0, d_distances + bfs_source);
    dist_t level = 1;

    Timer<DEVICE> TM;
    TM.start();
    //--------------------------------------------------------------------------
    // BFS ALGORITHM //
    while (queue.size() > 0) {
        queue.traverseAndFilter<BFSOperatorNoAtomic>(d_distances, level);
        level++;
    }
    //--------------------------------------------------------------------------
    // BFS VALIDATION //
    TM.stop();
    TM.print("BFS");

    auto is_correct = cu::equal(h_distances, h_distances + graph.nV(),
                                d_distances);
    std::cout << (is_correct ? "\nCorrect <>\n\n" : "\n! Not Correct\n\n");
}

//------------------------------------------------------------------------------

struct BFSOperatorAtomic {
    __device__ __forceinline__
    bool operator()(Vertex src, Edge edge, dist_t* d_distances, dist_t level) {
        auto dst = edge.dst();
        auto old = atomicCAS(d_distances + dst, INF, level);
        return old == INF;
    }
};

struct BFSOperatorNoAtomic {
    __device__ __forceinline__
    bool operator()(Vertex src, Edge edge, dist_t* d_distances, dist_t level) {
        auto dst = edge.dst();
        if (d_distances[dst] == INF) {
            d_distances[dst] = level;
            return true;    // the vertex dst is active
        }
        return false;       // the vertex dst is not active
    }
};

//------------------------------------------------------------------------------
//#include <cuda_profiler_api.h>
//cudaProfilerStart();
//cudaProfilerStop();
//xlib::printArray(h_distances, graph.nV(), "Host\n");
//xlib::printArray(tmp_distance, graph.nV(), "Device\n");

/*auto statistics = bfs.statistics(0);
int l = 1;
for (const auto& it : statistics)
    std::cout << l++ << "\t" << it[2] << std::endl;*/
