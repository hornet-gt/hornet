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

    OPERATOR(AoSData<vid_t>* eit1, AoSData<vid_t>* end1, AoSData<vid_t>* eit2, AoSData<vid_t>* end2, int FLAG) {
        int equalCount = 0;
        vid_t vid_curr1;
        vid_t vid_curr2;
        int comp;
        //printf("comparing: %p %p\n", eit1, eit2);
        while (eit1 != end1 && eit2 != end2) {
            vid_curr1 = (*eit1).get<0>();
            vid_curr2 = (*eit2).get<0>();
            comp = vid_curr1 - vid_curr2;
            equalCount += (comp == 0);
            if (comp <= 0)
                eit1++;
            if (comp >= 0)
                eit2++;
        }
        //atomicAdd(vTriangleCounts+v1.id(), equalCount);
    }
};

/*
 * Intersect operator optimized for better work-balance opportunities.
 * Assumption: operates on subarrays of adjacencies of v1 and v2
 * @input FLAG: indicates specific information regarding this operation
 */
struct OPERATOR_AdjIntersectionCount3 {
    int* vTriangleCounts;

    OPERATOR(Vertex &u, Vertex& v, vid_t* ui_begin, vid_t* ui_end, vid_t* vi_begin, vid_t* vi_end, int FLAG) {
        int count = 0;
        int comp_equals, comp1, comp2, ui_bound, vi_bound;
        //printf("Intersecting %d, %d: %d -> %d, %d -> %d\n", u.id(), v.id(), *ui_begin, *ui_end, *vi_begin, *vi_end);
        while (vi_begin <= vi_end && ui_begin <= ui_end) {
            comp_equals = (*ui_begin == *vi_begin);
            count += comp_equals;
            comp1 = (*ui_begin >= *vi_begin);
            comp2 = (*ui_begin <= *vi_begin);
            ui_bound = (ui_begin == ui_end);
            vi_bound = (vi_begin == vi_end);
            // early termination
            if ((ui_bound && comp2) || (vi_bound && comp1))
                break;
            if ((comp1 && !vi_bound) || ui_bound)
                vi_begin += 1;
            if ((comp2 && !ui_bound) || vi_bound)
                ui_begin += 1;
        }
        atomicAdd(vTriangleCounts+u.id(), count);
        atomicAdd(vTriangleCounts+v.id(), count);
    }
};

int main(int argc, char* argv[]) {

    using namespace graph::structure_prop;
    using namespace graph::parsing_prop;

    graph::GraphStd<vid_t, eoff_t> graph(UNDIRECTED);
    graph.read(argv[1], SORT | PRINT_INFO);
    HornetInit hornet_init(graph.nV(), graph.nE(), graph.csr_out_offsets(),
                           graph.csr_out_edges());

    HornetGraph hornet_graph(hornet_init);
    // hornet_graph.print();

    Timer<DEVICE> TM(5);
    cudaProfilerStart();
    TM.start();

    //forAllAdjUnions(hornet_graph, OPERATOR_AdjIntersectionCount3 { NULL });

    TM.stop();
    cudaProfilerStop();
    TM.print("ForAllAdjUnions Time");

    return 0;
}
