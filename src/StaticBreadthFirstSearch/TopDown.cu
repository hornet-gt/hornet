#include "StaticBreadthFirstSearch/TopDown.cuh"

namespace custinger_alg {

const dist_t INF = std::numeric_limits<dist_t>::max();

//------------------------------------------------------------------------------
__device__ __forceinline__
void VertexInit(vid_t index, void* optional_field) {
    auto bfs_data = *reinterpret_cast<BfsData*>(optional_field);
    bfs_data.distances[index] = INF;
}

__device__ __forceinline__
void BFSOperatorAtomic(Vertex src, Edge edge, void* optional_field) {
    auto bfs_data = *reinterpret_cast<BfsData*>(optional_field);
    auto dst = edge.dst();
    auto old = atomicCAS(bfs_data.distances + dst, INF, bfs_data.current_level);
    if (old == INF)
        bfs_data.queue.insert(src.id());     // the vertex dst is active*/
}

__device__ __forceinline__
void BFSOperatorNoAtomic(Vertex src, Edge edge, void* optional_field) {
    auto bfs_data = *reinterpret_cast<BfsData*>(optional_field);
    auto dst = edge.dst();
    if (bfs_data.distances[dst] == INF) {
        bfs_data.distances[dst] = bfs_data.current_level;
        bfs_data.queue.insert(src.id());    // the vertex dst is active
    }
}
//------------------------------------------------------------------------------

BfsTopDown::BfsTopDown(const custinger::cuStinger& custinger) :
                       StaticAlgorithm(custinger) {

    cuMalloc(bfs_data.distances, custinger.nV());
    reset();
}

BfsTopDown::~BfsTopDown() {
    cuFree(bfs_data.distances);
}

void BfsTopDown::reset() {
    bfs_data.queue.clear();
    forAllnumV<VertexInit>(custinger_graph, bfs_data.distances);
    cuMemcpyToDevice(0, bfs_data.distances + bfs_source);
}

void BfsTopDown::run() {
    while (bfs_data.queue.size() > 0) {
        load_balacing.traverse_edges<BFSOperatorNoAtomic>(bfs_data);
        bfs_data.queue.swap();
        bfs_data.current_level++;
    }
}

void BfsTopDown::release() {
    cuFree(bfs_data.distances);
    bfs_data.distances = nullptr;
}

bool BfsTopDown::validate() {
    using namespace graph;
    GraphStd<vid_t, eoff_t> graph(custinger.csr_offsets(), custinger.nV(),
                                   custinger.csr_edges(), custinger.nE());
    BFS<vid_t, eoff_t> bfs(graph);
    bfs.run(bfs_source);

    auto h_distances = bfs.distances();
    //auto  is_correct = cu::equal(h_distances, h_distances + graph.nV(),
    //                             bfs_data.distances);
    //std::cout << (is_correct ? "\nCorrect <>\n\n" : "\n! Not Correct\n\n");
    return cu::equal(h_distances, h_distances + graph.nV(), bfs_data.distances);
}

} // namespace custinger_alg
