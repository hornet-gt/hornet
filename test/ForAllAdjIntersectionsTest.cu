/**
 * @brief
 * @file
 */

#include "HornetAlg.hpp"
#include "Core/LoadBalancing/VertexBased.cuh"
#include "Core/LoadBalancing/ScanBased.cuh"
#include "Core/LoadBalancing/BinarySearch.cuh"
#include <Core/GPUCsr/Csr.cuh>
#include <Core/GPUHornet/Hornet.cuh>

#include <GraphIO/GraphStd.hpp>
#include <Util/CommandLineParam.hpp>
#include <cuda_profiler_api.h> //--profile-from-start off

using namespace timer;
using namespace hornets_nest;

using HornetGraph = gpu::Hornet<EMPTY, EMPTY>;

struct GetAttr {
    OPERATOR(Vertex& src, Edge& edge) {
        if (edge.src_id() == 2170) {
            printf("edge %d -> %d : \n", edge.src_id(), edge.dst_id());
        }
    }
};

/*
 * Naive intersection operator
 * Assumption: access to entire adjacencies of v1 and v2 required
 */
struct OPERATOR_AdjIntersectionCount {
    int* vTriangleCounts;

    OPERATOR(Vertex& v1, Vertex& v2) {
        int equalCount = 0;
        int outDeg1 = v1.out_degree();
        int outDeg2 = v2.out_degree();
        int eoff1 = 0;
        int eoff2 = 0;
        vid_t vid_curr1;
        vid_t vid_curr2;
        int comp;
        while (eoff1 < outDeg1 && eoff2 < outDeg2) {
            vid_curr1 = v1.edge(eoff1).dst_id();
            vid_curr2 = v2.edge(eoff2).dst_id();
            comp = vid_curr1 - vid_curr2;
            equalCount += (comp == 0);
            eoff1 += (comp <= 0 && !0);
            eoff2 += (comp >= 0 && !0);
        }
        atomicAdd(vTriangleCounts+v1.id(), equalCount);
    }
};

/*
 * Intersect operator optimized for better work-balance opportunities.
 * Assumption: operates on subarrays of adjacencies of v1 and v2
 * @input FLAG: indicates specific information regarding this operation
 */
struct OPERATOR_AdjIntersectionCount2 {
    int* vTriangleCounts;

    OPERATOR(EdgeIt& eit1, Edge& end1, EdgeIt& eit2, Edge& end2, int FLAG) {
        int equalCount = 0;
        vid_t vid_curr1;
        vid_t vid_curr2;
        int comp;
        while (*eit1 <= end1 && *eit2 <= end2) {
            vid_curr1 = eit1->dst_id();
            vid_curr2 = eit2->dst_id();
            comp = vid_curr1 - vid_curr2;
            equalCount += (comp == 0);
            if (comp <= 0)
                eit1++;
            if (comp <= 0)
                eit2++;
        }
        atomicAdd(vTriangleCounts+v1.id(), equalCount);
    }
};

int main(int argc, char* argv[]) {

    graph::GraphStd<vid_t, eoff_t> graph;
    CommandLineParam cmd(graph, argc, argv);
    //graph.print();

    HornetInit hornet_init(graph.nV(), graph.nE(), graph.csr_out_offsets(),
                           graph.csr_out_edges());

    HornetGraph hornet_graph(hornet_init);
    //hornet_graph.print();

    Timer<DEVICE> TM(5);
    cudaProfilerStart();
    TM.start();

    // Get the edge values
    //load_balacing::VertexBased1 load_balancing { hornet_graph };
    //forAllEdges(hornet_graph, GetAttr { }, load_balancing);
    forAllAdjUnions(hornet_graph, GetAttr { }); 

    TM.stop();
    cudaProfilerStop();
    TM.print("ForAllAdjUnions Time");

    return 0;
}
