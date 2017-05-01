


#include "cuStingerAlg/DeviceQueue.cuh"
#include "cuStingerAlg/Operator.cuh"    //Operator



#include "cuStingerAlg/cuStingerAlg.cuh"      //cuStingerAlg
#include "cuStingerAlg/TwoLevelQueue.cuh"       //Queue
#include "cuStingerAlg/MultiLevelQueue.cuh"       //Queue


#include "GraphIO/BFS.hpp"              //BFS
#include "GraphIO/GraphStd.hpp"         //GraphStd
#include "Support/Device/Algorithm.cuh" //cu::equal
#include "Support/Host/Timer.hpp"       //Timer
#include "cuStingerAlg/LoadBalancing/BinarySearch.cuh"

//#include "Core/cuStinger.hpp"           //cuStingerInit, cuStinger
#include "Csr/Csr.hpp"           //cuStingerInit, cuStinger
#include "Csr/CsrTypes.cuh"           //cuStingerInit, cuStinger

//using namespace csr;
using namespace cu_stinger_alg;
using namespace timer;
using namespace load_balacing;

using dist_t = int;

struct VertexInit;
struct BFSOperatorAtomic;
struct BFSOperatorNoAtomic;

//==============================================================================

int main(int argc, char* argv[]) {
    using namespace cu_stinger;
    cudaSetDevice(1);
    vid_t bfs_source = 0;
    //--------------------------------------------------------------------------
    //////////////
    // HOST BFS //
    //////////////
    graph::GraphStd<vid_t, eoff_t> graph;
    graph.read(argv[1]);
    graph::BFS<vid_t, eoff_t> bfs(graph);
    bfs.run(bfs_source);

    auto h_distances = bfs.distances();
    //--------------------------------------------------------------------------
    /////////////////
    // DEVICE INIT //
    /////////////////
    cuStingerInit custinger_init(graph.nV(), graph.nE(),
                                 graph.out_offsets_array(),
                                 graph.out_edges_array());

    cuStinger custiger_graph(custinger_init);

    dist_t* d_distances;
    Allocate alloc(d_distances, graph.nV());
    //--------------------------------------------------------------------------
    //////////////
    // BFS INIT //
    //////////////
    forAllnumV<VertexInit>(d_distances);
    cuMemcpyToDevice(0, d_distances + bfs_source);

    dist_t level = 1;
    TwoLevelQueue<vid_t> queue(graph.nV() * 2);
    queue.insert(bfs_source);
    //queue.insert(bfs_sources, num_sources);               // Multi-sources BFS
    load_balacing::BinarySearch lb(custinger_init, graph.nV() * 2);

    Timer<DEVICE> TM;
    TM.start();
    //--------------------------------------------------------------------------
    ///////////////////
    // BFS ALGORITHM //
    ///////////////////
    while (queue.size() > 0) {
        //lb.traverse_edges<BFSOperatorNoAtomic>(queue.device_ptr(), queue.size(),
        //                                       d_distances);
        level++;
    }
    //--------------------------------------------------------------------------
    ////////////////////
    // BFS VALIDATION //
    ////////////////////
    TM.stop();
    TM.print("BFS");

    auto is_correct = cu::equal(h_distances, h_distances + graph.nV(),
                                d_distances);
    std::cout << (is_correct ? "\nCorrect <>\n\n" : "\n! Not Correct\n\n");
}

//------------------------------------------------------------------------------

using cu_stinger::Vertex;
using cu_stinger::Edge;

const dist_t INF = std::numeric_limits<dist_t>::max();

struct VertexInit {
    __device__ __forceinline__
    static void apply(dist_t& vertex_distance) {
        vertex_distance = INF;
    }
};


__device__ __forceinline__
static void BFSOperatorAtomic(Vertex src, Edge edge, void* optional_field) {
    /*auto dst = edge.dst();
    auto old = atomicCAS(d_distances + dst, INF, level);
    if (old == INF)
        queue.insert(src.id());     // the vertex dst is active*/
}

__device__ __forceinline__
static void BFSOperatorNoAtomic(Vertex src, Edge edge, void* optional_field) {
    /*auto dst = edge.dst();
    if (d_distances[dst] == INF) {
        d_distances[dst] = level;
        queue.insert(src.id());    // the vertex dst is active
    }*/
}
