#include "Static/BreadthFirstSearch/TopDown.cuh"

namespace custinger_alg {

const dist_t INF = std::numeric_limits<dist_t>::max();

//------------------------------------------------------------------------------
///////////////
// OPERATORS //
///////////////

__device__ __forceinline__
void VertexInit(vid_t index, void* optional_field) {
    auto bfs_data = reinterpret_cast<BfsData*>(optional_field);
    bfs_data->distances[index] = INF;
}

__device__ __forceinline__
void BFSOperatorAtomic(Edge& edge, void* optional_field) {
    auto bfs_data = reinterpret_cast<BfsData*>(optional_field);
    auto dst = edge.dst_id();
    auto old = atomicCAS(bfs_data->distances + dst, INF,
                         bfs_data->current_level);
    if (old == INF)
        bfs_data->queue.insert(dst);         // the vertex dst is active
}

__device__ __forceinline__
void BFSOperatorNoAtomic(Edge& edge, void* optional_field) {
    auto bfs_data = reinterpret_cast<BfsData*>(optional_field);
    auto dst = edge.dst_id();
    if (bfs_data->distances[dst] == INF) {
        bfs_data->distances[dst] = bfs_data->current_level;
        bfs_data->queue.insert(dst);         // the vertex dst is active
    }
}
//------------------------------------------------------------------------------
////////////////
// BfsTopDown //
////////////////

BfsTopDown::BfsTopDown(custinger::cuStinger& custinger) :
                                       StaticAlgorithm(custinger),
                                       load_balacing(custinger),
                                       host_bfs_data(custinger) {
    gpu::allocate(host_bfs_data.distances, custinger.nV());
    device_bfs_data = register_data(host_bfs_data);
    reset();
}

BfsTopDown::~BfsTopDown() {
    gpu::free(host_bfs_data.distances);
}

void BfsTopDown::reset() {
    host_bfs_data.current_level = 1;
    host_bfs_data.queue.clear();
    syncDeviceWithHost();

    forAllnumV<VertexInit>(custinger, device_bfs_data);
}

void BfsTopDown::set_parameters(vid_t source) {
    bfs_source = source;
    host_bfs_data.queue.insert(bfs_source); // insert bfs source in the frontier
    //reset source distance
    cuMemcpyToDevice(0, host_bfs_data.distances + bfs_source);
}

void BfsTopDown::run() {
    while (host_bfs_data.queue.size() > 0) {
        forAllEdges<BFSOperatorAtomic>(host_bfs_data.queue, device_bfs_data,
                                       load_balacing);
        syncHostWithDevice();
        host_bfs_data.queue.swap();
        host_bfs_data.current_level++;
        syncDeviceWithHost();
    }
}

void BfsTopDown::release() {
    gpu::free(host_bfs_data.distances);
    host_bfs_data.distances = nullptr;
}

bool BfsTopDown::validate() {
    using namespace graph;
    GraphStd<vid_t, eoff_t> graph(custinger.csr_offsets(), custinger.nV(),
                                  custinger.csr_edges(), custinger.nE());
    BFS<vid_t, eoff_t> bfs(graph);
    bfs.run(bfs_source);
    /*auto vector = bfs.statistics(bfs_source);
    for (const auto& it : vector) {
        auto sum = it[0] + it[1] + it[2] + it[3];
        std::cout << it[2] << "\t" << sum << std::endl;
    }*/

    auto h_distances = bfs.distances();
    return cu::equal(h_distances, h_distances + graph.nV(),
                     host_bfs_data.distances);
}

} // namespace custinger_alg
