#pragma once

#include "HornetAlg.hpp"
#include "Core/HostDeviceVar.cuh"
#include "Core/LoadBalancing/VertexBased.cuh"
#include "Core/LoadBalancing/ScanBased.cuh"
#include "Core/LoadBalancing/BinarySearch.cuh"
#include <Core/GPUCsr/Csr.cuh>
#include <Core/GPU/Hornet.cuh>

using triangle_t = int;
using  HornetGPU = gpu::Hornet<EMPTY, TypeList<triangle_t>>;

namespace hornet_alg {

struct KTrussData {
    int max_K;

    int tsp;
    int nbl;
    int shifter;
    int blocks;
    int sps;

    int* is_active;
    int* offset_array;
    int* triangles_per_edge;
    int* triangles_per_vertex;

    vid_t* src;
    vid_t* dst;
    int    counter;
    int    active_vertices;

    TwoLevelQueue<vid_t> active_queue; // Stores all the active vertices

    int full_triangle_iterations;

    vid_t nv;
    off_t ne;                  // undirected-edges
    off_t num_edges_remaining; // undirected-edges
};

//==============================================================================

// Label propogation is based on the values from the previous iteration.
class KTruss : public StaticAlgorithm<HornetGPU> {
public:
    KTruss(HornetGPU& hornet);
    ~KTruss();

    void reset()    override;
    void run()      override;
    void release()  override;
    bool validate() override { return true; }

    //--------------------------------------------------------------------------
    void setInitParameters(int tsp, int nbl, int shifter,
                           int blocks, int sps);
    void init();

    bool findTrussOfK(bool& stop);
    void runForK(int max_K);

    void runDynamic();
    bool findTrussOfKDynamic(bool& stop);
    void runForKDynamic(int max_K);

    void copyOffsetArrayHost(vid_t* host_offset_array);
    void copyOffsetArrayDevice(vid_t* device_offset_array);
    void resetEdgeArray();
    void resetVertexArray();

    vid_t getIterationCount();
    vid_t getMaxK();

private:
    HostDeviceVar<KTrussData> hd_data;

    //load_balacing::BinarySearch load_balacing;
    //load_balacing::VertexBased1 load_balacing;
};

//==============================================================================

void callDeviceDifferenceTriangles(const HornetGPU& hornet,
                                   const gpu::BatchUpdate& batch_update,
                                   triangle_t* __restrict__ output_triangles,
                                   int threads_per_intersection,
                                   int num_intersec_perblock,
                                   int shifter,
                                   int thread_blocks,
                                   int blockdim,
                                   bool deletion);

} // namespace hornet_alg
